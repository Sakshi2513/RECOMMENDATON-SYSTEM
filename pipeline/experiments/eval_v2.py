"""Re-score the BM25 baseline on the repaired labels (v2). Same metric as stage 2."""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.common import OUT, jl_read
from pipeline.experiments.keyword_search import BM25, jacc, tags, topk


def main():
    V2 = OUT / "v2"
    tr, va, te = (jl_read(V2 / f"{n}.jsonl") for n in ("train", "val", "test"))

    def q(r):
        return f"{r['negative']} {r['positive']} {r['positive']}"

    rv = TfidfVectorizer(sublinear_tf=True, min_df=2).fit([r["recommendation"] for r in tr])
    Rt = rv.transform([r["recommendation"] for r in tr])
    tt = [tags(r["recommendation"]) for r in tr]
    bm = BM25([q(r) for r in tr])

    def ev(rows, top):
        Rq = rv.transform([r["recommendation"] for r in rows])
        tg = [tags(r["recommendation"]) for r in rows]
        sim = np.asarray([[Rq[i].multiply(Rt[j]).sum() for j in top[i]] for i in range(len(rows))])
        tj = np.asarray([[jacc(tg[i], tt[j]) for j in top[i]] for i in range(len(rows))])
        o = {}
        for k in (1, 5, 10):
            o[f"text_hit@{k}"] = round(float((sim[:, :k] >= 0.5).any(1).mean()), 3)
            o[f"tag_hit@{k}"] = round(float((tj[:, :k] >= 0.5).any(1).mean()), 3)
        return o

    res = {
        "val_bm25": ev(va, topk(bm.scores([q(r) for r in va]), 10)),
        "test_bm25": ev(te, topk(bm.scores([q(r) for r in te]), 10)),
    }
    rng = np.random.default_rng(0)
    res["test_random"] = ev(te, rng.integers(0, len(tr), size=(len(te), 10)))
    res["sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
    (OUT / "stage2_v2_report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
