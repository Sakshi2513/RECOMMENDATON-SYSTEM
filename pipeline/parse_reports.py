"""Stage 1, steps 1-3: collect reports, dedupe by PDF hash, parse areas from raw text, merge JSON.
Output: _built/areas_raw.jsonl, _built/manifest.csv, _built/parse_report.json
Run: python -m pipeline.parse_reports
"""
import csv, json, re, collections
from pipeline.common import *

MARK = re.compile(r"(?i)\b(need\s+to\s+do|solution\s*[:\-])")
AREA = re.compile(r"(?m)^Impacted Area (\d+)\s*$")


def lines_until_end(block):
    """Cut a block at the first photo list / heading garbage (two single-char lines in a row)."""
    out, singles = [], 0
    for ln in block.split("\n"):
        s = ln.strip()
        if re.match(r"(?i)^(positive|negative) side photographs$", s) or re.match(r"(?i)^photo \d+$", s):
            break
        singles = singles + 1 if len(s) == 1 else 0
        if singles >= 2:
            out = out[:-1]
            break
        out.append(s)
    return " ".join(x for x in out if x)


def section(chunk, start, ends):
    m = re.search(start, chunk)
    if not m:
        return None
    rest = chunk[m.end():]
    if start.startswith("Negative"):
        rest = rest.split("Positive side Description")[0]
    return lines_until_end(rest)


def parse_raw(text):
    ms = list(AREA.finditer(text))
    areas = {}
    for i, m in enumerate(ms):
        chunk = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)]
        n = int(m.group(1))
        neg = section(chunk, r"Negative side Description\s*\n", [])
        pos = section(chunk, r"Positive side Description\s*\n", [])
        areas[n] = (neg or "", pos or "")
    return areas


def split_action(pos):
    """Return (observation, [action steps]). Action = text from first 'Need to do' / 'Solution:'."""
    m = MARK.search(pos)
    if not m:
        return pos.strip(), []
    obs, act = pos[: m.start()].strip(), pos[m.start():]
    steps = [re.sub(r"(?i)^(need\s+to\s+do|solution\s*[:\-])\s*", "", s).strip(" ,;.")
             for s in re.split(r"(?i)(?=\bneed\s+to\s+do\b)", act)]
    steps = [s for s in steps if s]
    return obs, steps


FOOT = re.compile(r"(?i)private\s*&\s*confidential\s*\d*\s*/?\s*\d*|page\s+\d+\s+of\s+\d+")
LEAD = re.compile(r"(?i)^\s*(need\s+to\s+do|solution\s*[:\-]?|treatment\s*[:\-]?|recommendations?\s*[:\-]?|note\s*:)\s*")


def cut_after_label(raw, label):
    """If raw starts with the JSON label (ignoring punctuation/case), return (label_part, remainder)."""
    ln = re.sub(r"[^a-z0-9]", "", (label or "").lower())
    if not ln:
        return None
    seen, idx = "", None
    for i, ch in enumerate(raw):
        c = ch.lower()
        if c.isalnum() and c.isascii():
            seen += c
            if len(seen) == len(ln):
                idx = i + 1
                break
            if seen != ln[: len(seen)]:
                return None
    if idx is None:
        return None
    return raw[:idx].strip(), raw[idx:].strip(" ,;.")


def split_pos(raw_pos, jlabel, jtxt):
    """Return (observation, recommendation_text)."""
    raw_pos = FOOT.sub(" ", raw_pos)
    raw_pos = re.sub(r"\s+", " ", raw_pos).strip()
    cut = cut_after_label(raw_pos, jlabel)
    if cut:
        obs, rec = cut
        # JSON text may be a fragment; keep the longer when raw remainder is shorter
        if jtxt and len(re.sub(r"\W", "", jtxt)) > len(re.sub(r"\W", "", rec)):
            rec = jtxt
        return obs, rec
    obs, steps = split_action(raw_pos)
    if steps:
        return obs, " Need to do ".join(steps) if len(steps) > 1 else steps[0]
    return (jlabel or raw_pos), (jtxt or "")


def to_steps(rec):
    rec = re.sub(r"\s+", " ", rec or "").strip()
    parts = re.split(r"(?i)(?=need\s+to\s+do)", rec)
    out = []
    for p in parts:
        p = LEAD.sub("", p).strip(" ,;.")
        if p:
            out.append(p)
    return out


def norm_cmp(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    stats = collections.Counter()
    seen_hash = {}
    rows, manifest = [], []
    for d in inspection_dirs():
        stats["folders"] += 1
        jp, tp = d / "inspection.json", d / "raw_text.txt"
        if not (jp.exists() and tp.exists()):
            stats["missing_file"] += 1
            continue
        j = json.loads(jp.read_text(encoding="utf-8"))
        h = j.get("source_pdf_sha256")
        if h in seen_hash:
            stats["dup_pdf_hash_dropped"] += 1
            continue
        seen_hash[h] = d.name
        text = tp.read_text(encoding="utf-8", errors="replace")
        raw_areas = parse_raw(text)
        n_json = j.get("area_count") or 0
        manifest.append([j["inspection_id"], j.get("source_pdf"), h, j.get("inspection_date"), d.name])
        stats["reports"] += 1
        if len(raw_areas) != n_json:
            stats["area_count_mismatch"] += 1
        for n in range(1, max(n_json, max(raw_areas, default=0)) + 1):
            k = f"area_{n:02d}_"
            jneg, jpos = j.get(k + "negative_label"), j.get(k + "positive_label")
            jtxt = j.get(k + "positive_text")
            if n not in raw_areas and jneg is None and jpos is None:
                continue
            rneg, rpos_full = raw_areas.get(n, ("", ""))
            obs, rec_text = split_pos(rpos_full, jpos, jtxt)
            steps = to_steps(rec_text)
            # compare raw vs json
            if jneg and rneg and norm_cmp(jneg) != norm_cmp(rneg):
                stats["neg_raw_vs_json_differ"] += 1
            if not rneg and jneg:
                stats["neg_only_in_json"] += 1
            if jtxt and not steps:
                stats["json_has_rec_but_none_parsed"] += 1
            neg = rneg or jneg or ""
            pos_obs = obs or jpos or ""
            if not steps and jtxt:
                steps = to_steps(jtxt)
                stats["rec_from_json_fallback"] += 1
            stats["areas"] += 1
            rows.append({
                "id": f"{j['inspection_id']}_a{n:02d}", "inspection_id": j["inspection_id"],
                "inspection_date": j.get("inspection_date"), "source_pdf_sha256": h,
                "extraction_status": None,
                "negative_raw": neg, "positive_raw": pos_obs, "rec_steps_raw": steps,
                "negative_json": jneg, "positive_json": jpos,
                "checklist_type": j.get("checklist_type"), "property_type": j.get("property_type"),
            })
    jl_write(OUT / "areas_raw.jsonl", rows)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inspection_id", "source_pdf", "source_pdf_sha256", "inspection_date", "origin_folder"])
        w.writerows(manifest)
    stats["areas_with_rec"] = sum(1 for r in rows if r["rec_steps_raw"])
    stats["areas_empty_negative"] = sum(1 for r in rows if not r["negative_raw"])
    (OUT / "parse_report.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
