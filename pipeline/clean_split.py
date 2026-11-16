"""Stage 1, steps 4-8: clean, dedupe, split, write jsonl + review samples.
Input: _built/areas_raw.jsonl.  Output: train/val/test/unlabelled.jsonl, review CSVs, clean_report.json
Run: python -m pipeline.clean_split
"""

import collections
import csv
import difflib
import hashlib
import json
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.common import OUT, jl_read, jl_write
from recsys.cleaning import clean

NEAR_DUP = 0.95
ACTION_WORDS = re.compile(
    r"(?i)\b(grout|coat|patch|repair|replac|seal|fill|inject|waterproof|apply|remov|fix|plumbing|"
    r"checking|brick\s*bat|coba|plaster|paint|re-?do|chipping|treat|install|clean|fil+et|"
    r"crack|gap|joint|epoxy|pu\b|membrane)"
)
TYPO_DENY = {
    "clearing",
    "connecting",
    "constructing",
    "indication",
    "injecting",
    "interval",
    "seeing",
    "sports",
    "filing",
    "tilling",
    "damgaurd",
    "paining",
    "curring",
}
TYPO_ADD = {"lekage": "leakage", "lekaage": "leakage", "leakaed": "leaked"}


def build_typo_map(rows):
    cnt = collections.Counter()
    for r in rows:
        for f in ("negative_raw", "positive_raw"):
            cnt.update(re.findall(r"[a-z]{4,}", (r[f] or "").lower()))
        for s in r["rec_steps_raw"]:
            cnt.update(re.findall(r"[a-z]{4,}", s.lower()))
    common = [w for w, c in cnt.items() if c >= 80]
    by_first = collections.defaultdict(list)
    for w in common:
        by_first[w[0]].append(w)
    typo = {}
    for w, c in cnt.items():
        if c > 4 or len(w) < 6:
            continue
        best = difflib.get_close_matches(w, by_first[w[0]], n=1, cutoff=0.86)
        if not best:
            continue
        b = best[0]
        if cnt[b] < 100 * c or b == w or abs(len(b) - len(w)) > 1:
            continue
        if w.startswith(b) or b.startswith(w):  # plural / inflection, not a typo
            continue
        if w[:-1] == b[:-1] or w[:3] != b[:3]:  # different ending word or different stem
            continue
        typo[w] = b
    for w in TYPO_DENY:
        typo.pop(w, None)
    typo.update(TYPO_ADD)
    return typo


def main():
    rows = jl_read(OUT / "areas_raw.jsonl")
    rep = {}
    rep["areas_in"] = len(rows)

    typo = build_typo_map(rows)
    (OUT / "typo_map.json").write_text(json.dumps(typo, indent=1, sort_keys=True))
    rep["typo_fixes"] = len(typo)

    out = []
    for r in rows:
        neg, pos = clean(r["negative_raw"], typo), clean(r["positive_raw"], typo)
        rec = clean("; ".join(r["rec_steps_raw"]), typo)
        out.append(
            {
                "id": r["id"],
                "inspection_id": r["inspection_id"],
                "inspection_date": r["inspection_date"],
                "source_pdf_sha256": r["source_pdf_sha256"],
                "negative": neg,
                "positive": pos,
                "recommendation": rec,
                "checklist_type": r["checklist_type"],
                "property_type": r["property_type"],
            }
        )

    # drop unusable
    keep = [o for o in out if o["negative"] or o["positive"]]
    rep["dropped_no_text"] = len(out) - len(keep)
    out = keep
    unl = [o for o in out if len(o["recommendation"]) < 3]
    lab = [o for o in out if len(o["recommendation"]) >= 3]
    rep["labelled_before_dedupe"], rep["unlabelled_before_dedupe"] = len(lab), len(unl)

    def exact(rs):
        seen, res = set(), []
        for o in rs:
            k = (o["negative"], o["positive"], o["recommendation"])
            if k in seen:
                continue
            seen.add(k)
            res.append(o)
        return res

    lab2, unl2 = exact(lab), exact(unl)
    rep["exact_dupes_removed_labelled"], rep["exact_dupes_removed_unlabelled"] = (
        len(lab) - len(lab2),
        len(unl) - len(unl2),
    )

    # near-duplicates (TF-IDF cosine >= NEAR_DUP) on labelled rows
    texts = [f"{o['negative']} || {o['positive']} || {o['recommendation']}" for o in lab2]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    drop = set()
    n = X.shape[0]
    for s in range(0, n, 1000):
        sim = (X[s : s + 1000] @ X.T).tocoo()
        for i, j, v in zip(sim.row + s, sim.col, sim.data, strict=True):
            if j < i and v >= NEAR_DUP and j not in drop:
                drop.add(int(i))
    lab3 = [o for i, o in enumerate(lab2) if i not in drop]
    rep["near_dupes_removed_labelled"] = len(lab2) - len(lab3)

    # inputs that map to several different targets
    m = collections.defaultdict(set)
    for o in lab3:
        m[(o["negative"], o["positive"])].add(o["recommendation"])
    rep["inputs_with_multiple_targets"] = sum(1 for v in m.values() if len(v) > 1)

    # split by inspection id (80/10/10)
    def split_of(iid):
        h = int(hashlib.sha1(iid.encode()).hexdigest(), 16) % 10
        return "val" if h == 0 else "test" if h == 1 else "train"

    for o in lab3 + unl2:
        o["split"] = split_of(o["inspection_id"])
    for name in ("train", "val", "test"):
        jl_write(OUT / f"{name}.jsonl", [o for o in lab3 if o["split"] == name])
        rep[f"{name}_rows"] = sum(1 for o in lab3 if o["split"] == name)
        rep[f"{name}_inspections"] = len({o["inspection_id"] for o in lab3 if o["split"] == name})
    jl_write(OUT / "unlabelled.jsonl", unl2)
    rep["unlabelled_rows"] = len(unl2)
    rep["labelled_final"] = len(lab3)
    rep["rec_starts_with_observed"] = sum(1 for o in lab3 if o["recommendation"].startswith("observed"))
    rep["rec_without_action_word"] = sum(1 for o in lab3 if not ACTION_WORDS.search(o["recommendation"]))
    ids = {n: {o["inspection_id"] for o in lab3 if o["split"] == n} for n in ("train", "val", "test")}
    rep["leakage_train_test"] = len(ids["train"] & ids["test"])
    rep["leakage_train_val"] = len(ids["train"] & ids["val"])
    rep["distinct_recommendations"] = len({o["recommendation"] for o in lab3})
    L = sorted(len(o["recommendation"]) for o in lab3)
    rep["rec_len_chars_median_p90_max"] = [L[len(L) // 2], L[int(len(L) * 0.9)], L[-1]]

    # review samples for hand-checking
    random.seed(7)

    def dump(path, sample, extra):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "negative", "positive", "recommendation"] + extra)
            for o in sample:
                w.writerow([o["id"], o["negative"], o["positive"], o["recommendation"]] + [""] * len(extra))

    dump(OUT / "review_100_action_split.csv", random.sample(lab3, 100), ["is_action_only (y/n)", "notes"])
    tests = [o for o in lab3 if o["split"] == "test"]
    dump(
        OUT / "test_check_200.csv", random.sample(tests, min(200, len(tests))), ["label (good/wrong/unclear)", "notes"]
    )

    (OUT / "clean_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
