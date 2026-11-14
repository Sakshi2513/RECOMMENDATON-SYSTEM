"""Stage 3: semantic search (sentence embeddings, exact cosine search, hybrid with BM25, cross-encoder rerank).
Same hit metric as stage 2. Tune on val, score test once.  Run: python -m pipeline.experiments.semantic_search
(faiss is not installed; with ~8k vectors an exact matrix product is instant and gives the same result.)
"""
import json, random, time, numpy as np
from pipeline.common import *
from pipeline.experiments.keyword_search import BM25, tags, jacc, topk
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
q_of = lambda r: f'{r["negative"]} . {r["positive"]} . {r["positive"]}'


def z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def main():
    train, val, test = (jl_read(OUT / f"{n}.jsonl") for n in ("train", "val", "test"))
    rec_vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit([r["recommendation"] for r in train])
    R_tr = rec_vec.transform([r["recommendation"] for r in train])
    tg_tr = [tags(r["recommendation"]) for r in train]

    def metrics(rows, top, ks=(1, 5, 10)):
        R_q = rec_vec.transform([r["recommendation"] for r in rows])
        tg = [tags(r["recommendation"]) for r in rows]
        sim = np.asarray([[R_q[i].multiply(R_tr[j]).sum() for j in top[i]] for i in range(len(rows))])
        tj = np.asarray([[jacc(tg[i], tg_tr[j]) for j in top[i]] for i in range(len(rows))])
        o = {}
        for k in ks:
            if k <= top.shape[1]:
                o[f"text_hit@{k}"] = round(float((sim[:, :k] >= .5).any(1).mean()), 3)
                o[f"tag_hit@{k}"] = round(float((tj[:, :k] >= .5).any(1).mean()), 3)
        return o

    docs = [q_of(r) for r in train]
    bm = BM25(docs)
    S_bm = {"val": bm.scores([q_of(r) for r in val]), "test": bm.scores([q_of(r) for r in test])}
    res = {"val": {}}
    emb = {}
    for mn in MODELS:
        t0 = time.time()
        m = SentenceTransformer(mn)
        enc = lambda xs: m.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        E = enc(docs)
        emb[mn] = {"val": enc([q_of(r) for r in val]) @ E.T, "test": enc([q_of(r) for r in test]) @ E.T}
        short = mn.split("/")[-1]
        res["val"][f"dense {short}"] = metrics(val, topk(emb[mn]["val"], 10))
        print("val", f"dense {short}", res["val"][f"dense {short}"], f"{time.time()-t0:.0f}s")
        for a in (0.3, 0.5, 0.7):
            S = a * z(emb[mn]["val"]) + (1 - a) * z(S_bm["val"])
            res["val"][f"hybrid a={a} {short}"] = metrics(val, topk(S, 10))
            print("val", f"hybrid a={a} {short}", res["val"][f"hybrid a={a} {short}"])
    score = lambda k: res["val"][k]["text_hit@5"] + res["val"][k]["tag_hit@5"]
    best = max(res["val"], key=score)
    res["best_first_stage_on_val"] = best
    mn = next(x for x in MODELS if x.split("/")[-1] in best)
    if best.startswith("dense"):
        S = {s: emb[mn][s] for s in ("val", "test")}
    else:
        a = float(best.split("a=")[1].split()[0])
        S = {s: a * z(emb[mn][s]) + (1 - a) * z(S_bm[s]) for s in ("val", "test")}

    # cross-encoder rerank of the top 20 (zero-shot, not trained on this data)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    def rerank(rows, Sx):
        top = topk(Sx, 20)
        pairs = [(q_of(r), docs[j]) for r, t in zip(rows, top) for j in t]
        sc = ce.predict(pairs, batch_size=128, show_progress_bar=False).reshape(len(rows), 20)
        order = np.argsort(-sc, axis=1)
        return np.take_along_axis(top, order, 1), top
    t0 = time.time()
    rv, _ = rerank(val, S["val"])
    res["val"]["first stage + cross-encoder rerank"] = metrics(val, rv)
    print("val rerank", res["val"]["first stage + cross-encoder rerank"], f"{time.time()-t0:.0f}s")

    use_rerank = score("first stage + cross-encoder rerank") > score(best)
    res["rerank_used"] = bool(use_rerank)
    if use_rerank:
        top_test, _ = rerank(test, S["test"])
    else:
        top_test = topk(S["test"], 10)
    res["test_final"] = metrics(test, top_test)
    res["test_bm25_baseline_stage2"] = metrics(test, topk(S_bm["test"], 10))
    by = {}
    for pt in sorted({r["property_type"] or "unknown" for r in test}):
        ids = [i for i, r in enumerate(test) if (r["property_type"] or "unknown") == pt]
        if len(ids) >= 30:
            by[pt] = {"n": len(ids), **metrics([test[i] for i in ids], top_test[ids])}
    res["test_by_property_type"] = by
    (OUT / "stage3_report.json").write_text(json.dumps(res, indent=1))
    print("TEST final:", res["test_final"])
    print("TEST bm25 :", res["test_bm25_baseline_stage2"])
    print(by)


if __name__ == "__main__":
    main()
