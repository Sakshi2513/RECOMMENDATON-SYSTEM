"""Try to improve the action classifier. Pick the best setting on validation, report test once for every tried setting.
Feature groups:  text (as in the app) | + property/checklist type | + all checklist answers from inspection.json
Run: python -m pipeline.experiments.improve_features
"""
import csv, json, collections, numpy as np
from pipeline.common import *
from pipeline.action_classifier import ACTIONS, labels, q_of, best_thresholds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as prfs
from scipy.sparse import hstack, csr_matrix

V2 = OUT / "v2"
tr, va, te = (jl_read(V2 / f"{n}.jsonl") for n in ("train", "val", "test"))
names = list(ACTIONS)
Ytr, Yva, Yte = labels(tr), labels(va), labels(te)
keep = [j for j in range(len(names)) if Ytr[:, j].sum() >= 100 and Yva[:, j].sum() >= 20 and Yte[:, j].sum() >= 20]
Ytr, Yva, Yte = Ytr[:, keep], Yva[:, keep], Yte[:, keep]

# inspection id -> folder
folder = {r["inspection_id"]: r["origin_folder"] for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8"))}
need = {r["inspection_id"] for rows in (tr, va, te) for r in rows}
feat = {}
for iid in need:
    j = json.loads((ROOT / folder[iid] / "inspection.json").read_text(encoding="utf-8"))
    base, chk = [], []
    for k in ("property_type", "checklist_type", "previous_repair_work", "previous_structural_audit"):
        if isinstance(j.get(k), str):
            base.append(f"{k}={j[k]}")
    f = j.get("floors")
    if isinstance(f, (int, float)):
        base.append(f"floors={min(int(f), 6)}")
    for k, v in j.items():
        if k.startswith("checklist_") and k != "checklist_type" and isinstance(v, str) and 0 < len(v) <= 40:
            chk.append(f"{k}={v}")
    feat[iid] = (base, chk)
cnt_b, cnt_c = collections.Counter(), collections.Counter()
for iid in {r["inspection_id"] for r in tr}:
    cnt_b.update(feat[iid][0]); cnt_c.update(feat[iid][1])
vb = {t: i for i, t in enumerate(t for t, c in cnt_b.items() if c >= 30)}
vc = {t: i for i, t in enumerate(t for t, c in cnt_c.items() if c >= 30)}
print("base tokens", len(vb), "checklist tokens", len(vc))


def onehot(rows, which, vocab):
    m = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, r in enumerate(rows):
        for t in feat[r["inspection_id"]][which]:
            if t in vocab:
                m[i, vocab[t]] = 1
    return csr_matrix(m)


tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=25, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
T = {"tr": tf.fit_transform([q_of(r) for r in tr]), "va": tf.transform([q_of(r) for r in va]), "te": tf.transform([q_of(r) for r in te])}
B = {k: onehot(rows, 0, vb) for k, rows in (("tr", tr), ("va", va), ("te", te))}
C = {k: onehot(rows, 1, vc) for k, rows in (("tr", tr), ("va", va), ("te", te))}


def run(name, blocks, Creg=4.0, cw="balanced"):
    X = {k: hstack([b[k] * w for b, w in blocks]).tocsr() for k in ("tr", "va", "te")}
    S = {k: np.zeros((X[k].shape[0], len(keep))) for k in ("va", "te")}
    for j in range(len(keep)):
        mdl = LogisticRegression(C=Creg, max_iter=3000, class_weight=cw).fit(X["tr"], Ytr[:, j])
        for k in S:
            S[k][:, j] = mdl.predict_proba(X[k])[:, 1]
    th = best_thresholds(Yva, S["va"])
    v = prfs(Yva, S["va"] >= th, average=None, zero_division=0)[2]
    t = prfs(Yte, S["te"] >= th, average=None, zero_division=0)
    res = {"name": name, "val_macroF1": round(float(v.mean()), 3), "test_macroF1": round(float(t[2].mean()), 3), "test_per_action_F1": t[2].round(2).tolist()}
    print(res["name"].ljust(44), "val", res["val_macroF1"], "test", res["test_macroF1"])
    return res


results = []
results.append(run("A text only (current app)", [(T, 1)]))
results.append(run("B text + property/checklist type", [(T, 1), (B, 1)]))
results.append(run("C text + type + all checklist answers", [(T, 1), (B, 1), (C, 1)]))
results.append(run("C2 same, checklist weight 0.5", [(T, 1), (B, 1), (C, 0.5)]))
for Creg in (1.0, 16.0):
    results.append(run(f"C with C={Creg}", [(T, 1), (B, 1), (C, 1)], Creg=Creg))
results.append(run("C without class balancing", [(T, 1), (B, 1), (C, 1)], cw=None))
results.append(run("checklist answers only (no text)", [(B, 1), (C, 1)]))
best = max(results, key=lambda r: r["val_macroF1"])
print("best on validation:", best["name"], "-> test", best["test_macroF1"])
(OUT / "stage4_improve_report.json").write_text(json.dumps({"actions": [names[j] for j in keep], "results": results, "best_on_val": best["name"]}, indent=1))
