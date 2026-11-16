"""Label repair: keep only the action part of each recommendation.
Rules (applied per recommendation, in order):
 1. Split into clauses on ; and . (and on a comma before 'need to / should / ...').
 2. Drop junk clauses: photo lists, 'refer to ...', non-English text, very short clauses.
 3. Drop observation clauses (observed / present / detected ...) that contain no action wording.
 4. If no clause is left, the row has no usable recommendation and moves to unlabelled.
Input: train/val/test/unlabelled.jsonl   Output: v2/*.jsonl, label_fix_report.json, label_fix_samples.csv
Run: python -m pipeline.fix_labels
"""

import csv
import json
import random
import re

from pipeline.common import OUT, jl_read, jl_write

V2 = OUT / "v2"
ACTION = re.compile(
    r"\b(grout\w*|appl(?:y|ication|ied)|coat\w*|patch\w*|repair\w*|replac\w*|seal\w*|fill\w*|inject\w*|waterproof\w*|"
    r"install\w*|remov\w*|provid\w*|plaster\w*|paint\w*|treat\w*|rectif\w*|fix\w*|re-?do|dismantl\w*|clean\w*|"
    r"need\w*|requir\w*|should|shall|must|has to|have to|to be|recommend\w*|suggest\w*|do(?:ne)? the|"
    r"checking|check|chip\w*|break\w*|cut|make|renew\w*|rebuild\w*|re-?lay\w*|re-?tile\w*|connect\w*|"
    r"brick\s*bat|coba|epoxy|pressure|membrane|waterproofing|monitor\w*|further)\b"
)
OBS = re.compile(
    r"\b(observed|observation|observations|present|presence|detected|noticed|found|seen|appears?|visible|noted|"
    r"identified|was|were|are|is)\b"
)
JUNK = re.compile(r"(?:^|\s)photo\s*\d+|\brefer\b|kindly refer|impacted area[- ]?\d|as mentioned|same as|\bsft\b|^\W*$")
NONASCII = re.compile(r"[^\x00-\x7f]")


def fix(rec):
    clauses = [
        c.strip(" ,-:")
        for c in re.split(r"[;.]|,\s+(?=(?:and\s+)?(?:need|should|shall|must|solution|treatment|to be))", rec)
    ]
    keep = []
    for c in clauses:
        if len(c) < 8 or JUNK.search(c):
            continue
        if len(NONASCII.findall(c)) > len(c) * 0.3:
            continue
        if not ACTION.search(c):
            continue
        # observation clause: mentions an observation verb and the only action wording is weak
        if OBS.search(c) and not re.search(
            r"\b(grout|appl|coat|patch|repair|replac|seal|fill|inject|waterproof|install|remov|provid|"
            r"plaster|paint|rectif|need|requir|should|shall|must|to be|recommend|check|monitor|treat)",
            c,
        ):
            continue
        keep.append(c)
    return "; ".join(keep)


def main():
    V2.mkdir(exist_ok=True)
    rep = {}
    data = {n: jl_read(OUT / f"{n}.jsonl") for n in ("train", "val", "test", "unlabelled")}
    unl = data["unlabelled"]
    moved, changed, seen = 0, 0, set()
    out = {n: [] for n in ("train", "val", "test")}
    samples, dropped_dupes = [], 0
    for n in ("train", "val", "test"):
        for r in data[n]:
            new = fix(r["recommendation"])
            if not new:
                moved += 1
                unl.append(dict(r, recommendation="", label_fix="no_action_found", rec_original=r["recommendation"]))
                if len(samples) < 400:
                    samples.append((r["id"], "MOVED", r["recommendation"], ""))
                continue
            if len(new) < 0.9 * len(r["recommendation"]):
                changed += 1
                if len(samples) < 400:
                    samples.append((r["id"], "CHANGED", r["recommendation"], new))
            k = (r["negative"], r["positive"], new)
            if k in seen:
                dropped_dupes += 1
                continue
            seen.add(k)
            out[n].append(dict(r, recommendation=new, rec_original=r["recommendation"]))
    for n in out:
        jl_write(V2 / f"{n}.jsonl", out[n])
        rep[f"{n}_rows"] = len(out[n])
    jl_write(V2 / "unlabelled.jsonl", unl)
    rep["labelled_in"] = sum(len(data[n]) for n in ("train", "val", "test"))
    rep["labelled_out"] = sum(len(v) for v in out.values())
    rep["rows_kept_unchanged"] = rep["labelled_out"] - changed
    rep["rows_changed_to_action_part"] = changed  # includes small formatting changes
    rep["rows_moved_to_unlabelled"] = moved
    rep["dupes_removed_after_fix"] = dropped_dupes
    L = sorted(len(r["recommendation"]) for v in out.values() for r in v)
    rep["rec_len_chars_median_p90_max"] = [L[len(L) // 2], L[int(len(L) * 0.9)], L[-1]]
    rep["rec_starts_with_observed_after"] = sum(
        1 for v in out.values() for r in v if r["recommendation"].startswith("observed")
    )
    (OUT / "label_fix_report.json").write_text(json.dumps(rep, indent=2))
    random.seed(9)
    pick = random.sample(samples, min(100, len(samples)))
    with open(OUT / "label_fix_samples.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "before", "after", "correct? (y/n)"])
        for s in pick:
            w.writerow(list(s) + [""])
    for k, v in rep.items():
        print(k, v)
    for s in pick[:14]:
        print(
            s[1],
            "|",
            s[2][:170].encode("ascii", "replace").decode(),
            "=>",
            s[3][:170].encode("ascii", "replace").decode(),
        )


if __name__ == "__main__":
    main()
