"""Stage 2: keyword search baseline (TF-IDF and BM25) scored by top-k hit rate.
A retrieved case is a 'hit' for a query when its recommendation is close to the true one:
  text hit : TF-IDF cosine between the two recommendations >= 0.5
  tag hit  : the two recommendations share the same set of repair-action tags (Jaccard >= 0.5)
Tune on val, score test once.  Run: python -m pipeline.experiments.keyword_search
"""

import json
import random
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from pipeline.common import OUT, jl_read

TAGS = {
    "grouting": r"grout",
    "chemical_coat": r"chemical|coating|coat\b|brush",
    "plumbing": r"plumb|pipe|nahani|nahin|trap|drain",
    "brick_bat_coba": r"brick\s*bat|coba|bat coba",
    "epoxy": r"epoxy",
    "sealant": r"sealant|sealent|silicon|caulk|pu\b",
    "patch": r"patch|plaster|crack|chipping",
    "waterproof": r"waterproof|membrane|app\b|sbs",
    "tile": r"tile|tiling|retil|flooring",
    "injection": r"inject|pressure",
    "paint": r"paint|putty|primer",
    "skirting": r"skirting",
    "fillet": r"fillet|filet|watta",
    "inspection": r"further|inspect|check|monitor|observe",
}
TAGRX = {k: re.compile(v, re.I) for k, v in TAGS.items()}


def tags(t):
    return frozenset(k for k, rx in TAGRX.items() if rx.search(t))


def jacc(a, b):
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        self.n = tf.shape[0]
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean()
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x docs

    def scores(self, queries):
        q = self.cv.transform(queries)
        q.data[:] = 1
        return (q @ self.W).toarray()


def topk(S, k):
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(S, idx, 1), axis=1), 1)


def main():
    train, val, test = (jl_read(OUT / f"{n}.jsonl") for n in ("train", "val", "test"))
    rec_vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit([r["recommendation"] for r in train])
    R_tr = rec_vec.transform([r["recommendation"] for r in train])
    tags_tr = [tags(r["recommendation"]) for r in train]

    def evaluate(qrows, field_fn, method, ks=(1, 5, 10)):
        docs = [field_fn(r) for r in train]
        qs = [field_fn(r) for r in qrows]
        if method == "tfidf":
            v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
            D = v.fit_transform(docs)
            S = (v.transform(qs) @ D.T).toarray()
        else:
            S = BM25(docs).scores(qs)
        top = topk(S, max(ks))
        R_q = rec_vec.transform([r["recommendation"] for r in qrows])
        out = {}
        sim = np.asarray([[R_q[i].multiply(R_tr[j]).sum() for j in top[i]] for i in range(len(qrows))])
        tg_q = [tags(r["recommendation"]) for r in qrows]
        tagj = np.asarray([[jacc(tg_q[i], tags_tr[j]) for j in top[i]] for i in range(len(qrows))])
        for k in ks:
            out[f"text_hit@{k}"] = round(float((sim[:, :k] >= 0.5).any(1).mean()), 3)
            out[f"tag_hit@{k}"] = round(float((tagj[:, :k] >= 0.5).any(1).mean()), 3)
        out["mean_best_text_sim@5"] = round(float(sim[:, :5].max(1).mean()), 3)
        return out, top

    # random baseline (same metric) for context
    rng = np.random.default_rng(0)
    rtop = rng.integers(0, len(train), size=(len(val), 10))
    R_v = rec_vec.transform([r["recommendation"] for r in val])
    tg_v = [tags(r["recommendation"]) for r in val]
    rb = {
        f"{m}_hit@5": round(
            float(
                np.mean(
                    [
                        any(
                            (m == "text" and R_v[i].multiply(R_tr[j]).sum() >= 0.5)
                            or (m == "tag" and jacc(tg_v[i], tags_tr[j]) >= 0.5)
                            for j in rtop[i, :5]
                        )
                        for i in range(len(val))
                    ]
                )
            ),
            3,
        )
        for m in ("text", "tag")
    }

    variants = {
        "neg+pos": lambda r: f"{r['negative']} {r['positive']}",
        "pos only": lambda r: r["positive"],
        "neg only": lambda r: r["negative"],
        "neg+2xpos": lambda r: f"{r['negative']} {r['positive']} {r['positive']}",
    }
    res = {"random_val": rb, "val": {}}
    for name, fn in variants.items():
        for m in ("tfidf", "bm25"):
            res["val"][f"{m} | {name}"] = evaluate(val, fn, m)[0]
            print(f"val {m:5s} {name:10s}", res["val"][f"{m} | {name}"])
    best = max(res["val"], key=lambda k: res["val"][k]["tag_hit@5"] + res["val"][k]["text_hit@5"])
    m, name = [x.strip() for x in best.split("|")]
    res["best_on_val"] = best
    tres, top = evaluate(test, variants[name], m)
    res["test_best"] = tres
    # per property type
    by = {}
    for pt in sorted({r["property_type"] or "unknown" for r in test}):
        ids = [i for i, r in enumerate(test) if (r["property_type"] or "unknown") == pt]
        if len(ids) >= 30:
            sub = [test[i] for i in ids]
            by[pt] = {"n": len(ids), **evaluate(sub, variants[name], m)[0]}
    res["test_by_property_type"] = by
    # sample outputs for eyeballing
    random.seed(11)
    samples = []
    for i in random.sample(range(len(test)), 10):
        samples.append(
            {
                "negative": test[i]["negative"],
                "positive": test[i]["positive"],
                "true": test[i]["recommendation"][:200],
                "top1": train[top[i][0]]["recommendation"][:200],
            }
        )
    res["samples"] = samples
    (OUT / "stage2_report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("BEST on val:", best)
    print("TEST:", tres)
    print("random (val):", rb)
    print("by property:", by)


if __name__ == "__main__":
    main()
