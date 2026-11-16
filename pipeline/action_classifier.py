"""Stage 4: multi-label repair-action classifier.
Action tags are derived from the v2 recommendation text by keyword rules (weak labels).
Model reads negative + positive text and predicts which actions apply.
Compares: TF-IDF + logistic regression, MiniLM embeddings + logistic regression, kNN vote from BM25 top-5.
Thresholds tuned per action on val; test scored once.  Run: python -m pipeline.action_classifier
"""

import json

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from pipeline.common import OUT, jl_read
from pipeline.experiments.keyword_search import BM25, topk
from recsys.actions import ACTION_RULES, action_flags
from recsys.classifier import model_text

V2 = OUT / "v2"
V2 = OUT / "v2"
ACTIONS = ACTION_RULES  # action -> keyword rule (see recsys/actions.py)


def labels(rows):
    return np.array([action_flags(r["recommendation"]) for r in rows])


def q_of(r):
    return model_text(r["negative"], r["positive"])


def prf(Y, P):
    p, r, f, s = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    return p, r, f, s


def best_thresholds(Y, S):
    th = []
    for j in range(Y.shape[1]):
        best, bt = -1, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f = precision_recall_fscore_support(Y[:, j], S[:, j] >= t, average="binary", zero_division=0)[2]
            if f > best:
                best, bt = f, float(t)
        th.append(bt)
    return np.array(th)


def main():
    tr, va, te = (jl_read(V2 / f"{n}.jsonl") for n in ("train", "val", "test"))
    Ytr, Yva, Yte = labels(tr), labels(va), labels(te)
    names = list(ACTIONS)
    support = Ytr.sum(0)
    keep = [j for j in range(len(names)) if support[j] >= 100 and Yva[:, j].sum() >= 20 and Yte[:, j].sum() >= 20]
    names_k = [names[j] for j in keep]
    Ytr, Yva, Yte = Ytr[:, keep], Yva[:, keep], Yte[:, keep]
    print("actions kept:", names_k, "train support:", Ytr.sum(0).tolist())

    texts = {n: [q_of(r) for r in rows] for n, rows in (("tr", tr), ("va", va), ("te", te))}
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = {"tr": tf.fit_transform(texts["tr"]), "va": tf.transform(texts["va"]), "te": tf.transform(texts["te"])}

    def fit_lr(Xtr, Xs):
        S = {k: np.zeros((Xv.shape[0], len(keep))) for k, Xv in Xs.items()}
        for j in range(len(keep)):
            m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xtr, Ytr[:, j])
            for k, Xv in Xs.items():
                S[k][:, j] = m.predict_proba(Xv)[:, 1]
        return S

    res = {"actions": names_k, "train_support": Ytr.sum(0).tolist(), "val": {}, "test": {}}
    S_tfidf = fit_lr(X["tr"], {"va": X["va"], "te": X["te"]})

    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = {k: m.encode(texts[k], batch_size=64, normalize_embeddings=True, show_progress_bar=False) for k in texts}
    S_emb = fit_lr(E["tr"], {"va": E["va"], "te": E["te"]})
    Xc = {k: hstack([X[k], csr_matrix(E[k])]).tocsr() for k in X}
    S_comb = fit_lr(Xc["tr"], {"va": Xc["va"], "te": Xc["te"]})

    # kNN vote baseline from BM25 top-5 (stage 2 retrieval)
    bm = BM25(texts["tr"])

    def knn(k):
        top = topk(bm.scores(texts[k]), 5)
        return np.stack([Ytr[t].mean(0) for t in top])

    S_knn = {"va": knn("va"), "te": knn("te")}

    models = {"tfidf_lr": S_tfidf, "minilm_lr": S_emb, "tfidf+minilm_lr": S_comb, "bm25_knn5": S_knn}
    for name, S in models.items():
        th = best_thresholds(Yva, S["va"])
        res["val"][name] = {"macro_f1": round(float(prf(Yva, S["va"] >= th)[2].mean()), 3)}
        models[name] = dict(S, th=th)
        print("val", name, res["val"][name])
    # 'always predict the most common action' reference
    best = max(res["val"], key=lambda k: res["val"][k]["macro_f1"])
    res["best_on_val"] = best
    for name in models:
        S = models[name]
        P = S["te"] >= S["th"]
        p, r, f, s = prf(Yte, P)
        res["test"][name] = {
            "macro_f1": round(float(f.mean()), 3),
            "micro_f1": round(float(precision_recall_fscore_support(Yte, P, average="micro", zero_division=0)[2]), 3),
        }
        if name == best:
            res["test_per_action"] = {
                a: {
                    "precision": round(float(p[i]), 3),
                    "recall": round(float(r[i]), 3),
                    "f1": round(float(f[i]), 3),
                    "support_test": int(s[i]),
                    "share_of_test": round(float(Yte[:, i].mean()), 3),
                }
                for i, a in enumerate(names_k)
            }
            # exact-set match and 'at least the right main action'
            res["test_exact_set_match"] = round(float((Yte.astype(bool) == P).all(1).mean()), 3)
            res["test_mean_actions_true_vs_pred"] = [
                round(float(Yte.sum(1).mean()), 2),
                round(float(P.sum(1).mean()), 2),
            ]
            np.save(OUT / "stage4_test_scores.npy", S["te"])
    (OUT / "stage4_report.json").write_text(json.dumps(res, indent=1))
    print("best on val:", best)
    print("test:", res["test"])
    for a, v in res["test_per_action"].items():
        print(f"{a:22s}", v)
    print("exact set match", res["test_exact_set_match"], res["test_mean_actions_true_vs_pred"])


if __name__ == "__main__":
    main()
