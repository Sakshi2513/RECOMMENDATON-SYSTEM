"""Train the model used by the app and write it to models/.

The vocabulary is limited to plain ASCII words (no digits) that appear in at least --min-df training areas, so names,
unit numbers and rare one-off terms from customer reports are not stored in the model file (the repository is public).
Outputs: models/model.joblib, models/typo_map.json, models/vocabulary_review.txt (single words, for a privacy review).

Run: python -m pipeline.build_app_model [--min-df 25] [--out-dir models]
"""
import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from pipeline.action_classifier import ACTIONS, best_thresholds, labels, q_of
from pipeline.common import OUT, jl_read
from recsys.config import BASE_DIR

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{2,}\b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-df", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=BASE_DIR / "models")
    args = ap.parse_args()

    v2 = OUT / "v2"
    tr, va, te = (jl_read(v2 / f"{n}.jsonl") for n in ("train", "val", "test"))
    names = list(ACTIONS)
    ytr, yva, yte = labels(tr), labels(va), labels(te)
    keep = [j for j in range(len(names)) if ytr[:, j].sum() >= 100 and yva[:, j].sum() >= 20 and yte[:, j].sum() >= 20]
    names_kept = [names[j] for j in keep]
    ytr, yva, yte = ytr[:, keep], yva[:, keep], yte[:, keep]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=args.min_df, token_pattern=TOKEN_PATTERN)
    xtr = vectorizer.fit_transform([q_of(r) for r in tr])
    xva, xte = vectorizer.transform([q_of(r) for r in va]), vectorizer.transform([q_of(r) for r in te])
    models = [LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(xtr, ytr[:, j]) for j in range(len(keep))]
    scores = lambda x: np.column_stack([m.predict_proba(x)[:, 1] for m in models])
    thresholds = best_thresholds(yva, scores(xva))   # tuned on validation only
    f1 = precision_recall_fscore_support(yte, scores(xte) >= thresholds, average=None, zero_division=0)[2]
    print(f"vocabulary {len(vectorizer.vocabulary_)} terms; test macro F1 {f1.mean():.3f}")
    for name, value in zip(names_kept, f1):
        print(f"  {name:22s} {value:.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tf": vectorizer, "models": models, "th": thresholds, "names": names_kept}, args.out_dir / "model.joblib")
    shutil.copy(OUT / "typo_map.json", args.out_dir / "typo_map.json")
    words = sorted(w for w in vectorizer.vocabulary_ if " " not in w)
    (args.out_dir / "vocabulary_review.txt").write_text(" ".join(words) + "\n", encoding="utf-8")
    print(f"saved to {args.out_dir} ({len(words)} single words listed in vocabulary_review.txt)")


if __name__ == "__main__":
    main()
