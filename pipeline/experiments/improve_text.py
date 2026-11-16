"""Second round: separate negative/positive text blocks, and train on train+val with cross-validated thresholds.
Test scored once per setting.  Run: python -m pipeline.experiments.improve_text"""

import json

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import GroupKFold

from pipeline.action_classifier import ACTIONS, best_thresholds, labels
from pipeline.common import OUT, jl_read
from recsys.classifier import model_text


def main():
    V2 = OUT / "v2"
    tr, va, te = (jl_read(V2 / f"{n}.jsonl") for n in ("train", "val", "test"))
    names = list(ACTIONS)
    Ytr, Yva, Yte = labels(tr), labels(va), labels(te)
    keep = [j for j in range(len(names)) if Ytr[:, j].sum() >= 100 and Yva[:, j].sum() >= 20 and Yte[:, j].sum() >= 20]
    Ytr, Yva, Yte = Ytr[:, keep], Yva[:, keep], Yte[:, keep]
    TP = r"(?u)\b[a-zA-Z]{2,}\b"

    def joint(r):
        return model_text(r["negative"], r["positive"])

    def blocks(fit_rows, rows_list, mode, min_df=25):
        vs = []
        if mode in ("joint", "joint+sep"):
            vs.append((TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, token_pattern=TP), joint))
        if mode in ("sep", "joint+sep"):
            vs.append(
                (
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, token_pattern=TP),
                    lambda r: r["negative"],
                )
            )
            vs.append(
                (
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, token_pattern=TP),
                    lambda r: r["positive"],
                )
            )
        for v, f in vs:
            v.fit([f(r) for r in fit_rows])
        return [hstack([v.transform([f(r) for r in rows]) for v, f in vs]).tocsr() for rows in rows_list]

    def fit_scores(Xtr, Ytr_, Xs, C=4.0):
        out = [np.zeros((X.shape[0], Ytr_.shape[1])) for X in Xs]
        for j in range(Ytr_.shape[1]):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, Ytr_[:, j])
            for o, X in zip(out, Xs, strict=True):
                o[:, j] = m.predict_proba(X)[:, 1]
        return out

    res = []
    for mode in ("joint", "sep", "joint+sep"):
        # 1) as before: fit on train, thresholds on val
        Xtr, Xva, Xte = blocks(tr, [tr, va, te], mode)
        Sva, Ste = fit_scores(Xtr, Ytr, [Xva, Xte])
        th = best_thresholds(Yva, Sva)
        f1 = prfs(Yte, Ste >= th, average=None, zero_division=0)[2]
        r1 = {
            "mode": mode,
            "setup": "train only, thresholds on val",
            "val": round(float(prfs(Yva, Sva >= th, average=None, zero_division=0)[2].mean()), 3),
            "test": round(float(f1.mean()), 3),
        }
        # 2) train+val, thresholds from 5-fold CV grouped by inspection
        al = tr + va
        Yal = np.vstack([Ytr, Yva])
        groups = [r["inspection_id"] for r in al]
        oof = np.zeros(Yal.shape)
        for a, b in GroupKFold(5).split(al, groups=groups):
            Xa, Xb = blocks([al[i] for i in a], [[al[i] for i in a], [al[i] for i in b]], mode)
            oof[b] = fit_scores(Xa, Yal[a], [Xb])[0]
        th2 = best_thresholds(Yal, oof)
        Xal, Xte2 = blocks(al, [al, te], mode)
        Ste2 = fit_scores(Xal, Yal, [Xte2])[0]
        f2 = prfs(Yte, Ste2 >= th2, average=None, zero_division=0)
        r2 = {
            "mode": mode,
            "setup": "train+val, CV thresholds",
            "cv": round(float(prfs(Yal, oof >= th2, average=None, zero_division=0)[2].mean()), 3),
            "test": round(float(f2[2].mean()), 3),
            "test_per_action_F1": f2[2].round(2).tolist(),
        }
        print(r1)
        print(r2)
        res += [r1, r2]
    (OUT / "stage4_improve2_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
