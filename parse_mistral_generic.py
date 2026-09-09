#!/usr/bin/env python3
"""GENERIC Mistral-markdown-table -> our oneshot JSON converter.

Unlike parse_mistral_md (hardcoded for the Table VII civil-condition grid), this infers
the column groups/subkeys from the markdown's multi-row header rather than assuming a
fixed layout. Handles Tables VIII / IX / XIII / XVII (Education/Literacy by Religion&Age,
Civil Condition by Religion, Occupations, Subsidiary, Infirmities, ...).

Structure learned from real Mistral output:
  - A markdown table block: 1 header row ABOVE the `|---|` separator, then below it more
    group rows, the innermost SUB row (Males/Females/Total/Persons) closest to the data,
    optionally a printed column-number row (17|18|19...), then the data rows.
  - Columns carry a (group, subkey) identity: groups forward-filled top->bottom across the
    super-group/group header rows; subkey from the LAST (innermost) header row.
  - Leading column(s) may be a text row-label (occupations) or absent (religions/districts
    not captured -> synthetic "Row N").
  - A block with NO real header (first row is numeric data) is a CONTINUATION (e.g. the
    MARRIED/WIDOWED sub-blocks of Table VIII) -> reuse the previous block's column map.

Output section shape matches what detect_misfires + the oneshot consumers expect:
  [{"name": <section>, "rows": [{<rowkey>: <label>, <group>: {<subkey>: int}, ...}]}]
"""
import os, re, glob, json, collections

NUM = re.compile(r"^-?[\d,]+([.·'ʼ’`][\d]+)?$")  # int OR decimal ('.', '·' mid-dot, or "'" apostrophe)
DOTS = re.compile(r"^[.…\s]*$")             # '...' / '…' / blank = nil (>=2 dots, no digit)
SUB_MALE = re.compile(r"\bmales?\b", re.I)
SUB_FEMALE = re.compile(r"\bfemales?\b", re.I)
SUB_PERSON = re.compile(r"\bpersons?\b|\btotal\b|\bboth sexes\b", re.I)
ROWLABEL_DIGIT = re.compile(r"\d")
JUNK_LINE = re.compile(r"this content downloaded|jstor|all use subject|^\s*\d+\s*$|"
                       r"^\s*[\divx]+[-.\s]*cp\.?\s*$|^\s*added specially|^\s*page\b",
                       re.I)


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_sep(line):
    s = line.strip().strip("|")
    return bool(s) and set(s.replace("|", "").replace("-", "").replace(":", "").strip()) <= {" ", ""} and "-" in s


def _is_sep_cells(cells):
    """A markdown header/body separator row: every cell is dashes (e.g. '---', ':--:')."""
    cc = [c.strip() for c in cells if c.strip()]
    return bool(cc) and all(set(c) <= set("-:") for c in cc)


def _is_num(c):
    c = c.strip()
    return bool(NUM.match(c)) and c != ""


def _to_num(c):
    """Parse a numeric cell -> int (whole) or float (decimal; '·' mid-dot, '.', or "'")."""
    c = c.strip().replace(",", "")
    c = re.sub(r"[·'ʼ’`]", ".", c)
    try:
        return int(c)
    except Exception:
        pass
    try:
        return float(c)
    except Exception:
        return None


def _to_int(c):
    v = _to_num(c)
    return v if isinstance(v, int) else None


def _consecutive_ints(cells):
    """True for a printed column-number header row like 17|18|19|20 (consecutive ints)."""
    vals = [_to_int(c) for c in cells if c.strip()]
    if len(vals) < 3 or any(v is None for v in vals):
        return False
    return all(0 < v < 200 for v in vals) and all(vals[i + 1] == vals[i] + 1 for i in range(len(vals) - 1))


def _norm_sub(txt):
    if SUB_MALE.search(txt) and not SUB_FEMALE.search(txt):
        return "males"
    if SUB_FEMALE.search(txt):
        return "females"
    if SUB_PERSON.search(txt):
        return "persons"
    t = re.sub(r"\s+", " ", txt.strip().strip(".")).lower()
    return t or None


def _clean_group(txt):
    return re.sub(r"\s+", " ", txt.strip()).strip(".").strip() or None


def _split_blocks(md_path):
    """Yield (section_name, list-of-pipe-lines) blocks, threading the latest standalone
    text label as the section name."""
    lines = open(md_path).read().splitlines()
    blocks, cur_name, i = [], "", 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            blocks.append((cur_name, block))
            continue
        # standalone text -> candidate section label
        if s and not JUNK_LINE.search(s) and len(s) < 60 and re.search(r"[A-Za-z]", s):
            cur_name = s.strip().strip(".").strip()
        i += 1
    return blocks


def _build_colmap(header_rows, ncol):
    """header_rows = ordered header cell-lists (super-group ... group ... innermost SUB row).
    Return (colgroup: {c->group str}, colsub: {c->subkey})."""
    if not header_rows:
        return {}, {}
    sub_row = header_rows[-1]
    group_rows = header_rows[:-1]
    # forward-fill each group row across its empty spans, top->bottom combine
    colgroup = {}
    for c in range(ncol):
        parts = []
        for gr in group_rows:
            # forward-fill: find nearest non-empty NON-numeric label at or before c in this row
            lab = None
            for cc in range(c, -1, -1):
                if cc < len(gr) and gr[cc].strip():
                    if _is_num(gr[cc]):
                        break              # a stray numeric cell in a header row is data, not a label
                    lab = _clean_group(gr[cc]); break
            if lab:
                parts.append(lab)
        if parts:
            colgroup[c] = ". ".join(parts) + "."
    colsub = {}
    for c in range(ncol):
        if c < len(sub_row) and sub_row[c].strip():
            sk = _norm_sub(sub_row[c])
            if sk:
                colsub[c] = sk
    return colgroup, colsub


def _label_col_count(data_rows, ncol, colsub):
    """How many leading columns are row-labels (not data)?
    Preferred signal: a data column always has a subkey (Males/Females/persons/...), so leading
    columns with NO subkey are labels — this correctly treats a numeric Year/Age first column as
    a label. Falls back to the non-numeric-majority heuristic when no subkeys were detected."""
    if colsub:
        n = 0
        for c in range(ncol):
            if c in colsub:
                break
            n += 1
        # guard: never swallow every column as a label
        return n if n < ncol else 0
    n = 0
    for c in range(min(ncol, 3)):
        col = [r[c] for r in data_rows if c < len(r) and r[c].strip()]
        if not col:
            break
        nonnum = sum(1 for v in col if not _is_num(v))
        if nonnum > 0.5 * len(col):
            n += 1
        else:
            break
    return n


def parse_sections(md_path, rowkey="row"):
    blocks = _split_blocks(md_path)
    sections = []
    prev = None  # (colgroup, colsub, ncol, nlabel)
    row_counter = 0
    for name, block in blocks:
        # sequence of cell-rows in reading order; markdown separators dropped (noise)
        seq = [g for g in (_cells(b) for b in block) if not _is_sep_cells(g)]
        if not seq:
            continue

        def content_cells(cells):
            return [c for c in cells if c.strip() and not DOTS.match(c)]

        def n_nonnumeric(cells):
            return sum(1 for c in content_cells(cells) if not _is_num(c))

        def n_numeric(cells):
            return sum(1 for c in content_cells(cells) if _is_num(c))

        # data starts at the first non-colnum row with >=2 numeric cells (wide tables);
        # fall back to >=1 for genuinely narrow (1-value) tables when no row reaches 2.
        def first_data(thresh):
            for k, r in enumerate(seq):
                if _consecutive_ints(r):
                    continue
                if n_numeric(r) >= thresh:
                    return k
            return None
        def last_spanning(cand):
            for k in range(len(cand) - 1, -1, -1):
                if n_nonnumeric(cand[k]) >= 2:
                    return k
            return None

        data_start = first_data(2)
        if data_start is None:
            data_start = first_data(1)
        if data_start is None:
            # header-only block: the data is in a FOLLOWING block (a standalone text line
            # split the markdown table). Build & stash the colmap so the next data-only block
            # can reuse it; emit no rows here.
            cand = [r for r in seq if not _consecutive_ints(r)]
            si = last_spanning(cand)
            if si is not None:
                header_rows = cand[:si + 1]
                ncol = max(len(r) for r in seq)
                cg, cs = _build_colmap(header_rows, ncol)
                if cs:
                    prev = (cg, cs, ncol, _label_col_count([], ncol, cs))
            continue

        header_cand = [r for r in seq[:data_start] if not _consecutive_ints(r)]
        # the innermost SUB header = the last spanning (>=2 non-numeric) row before the data
        sub_idx = last_spanning(header_cand)
        data_rows = [r for r in seq[data_start:] if not _consecutive_ints(r)]

        if sub_idx is not None:
            header_rows = header_cand[:sub_idx + 1]   # super-group(s) ... group(s) ... SUB
            ncol = max(len(r) for r in seq)
            colgroup, colsub = _build_colmap(header_rows, ncol)
            nlabel = _label_col_count(data_rows, ncol, colsub)
            prev = (colgroup, colsub, ncol, nlabel)
        else:
            if not prev:
                continue                          # no header here and nothing to reuse
            colgroup, colsub, prev_ncol, nlabel = prev
            ncol = max([prev_ncol] + [len(r) for r in seq])

        rows = []
        for r in data_rows:
            if not r or all(DOTS.match(c) for c in r):
                continue
            # row label
            if nlabel > 0:
                lab = " ".join(r[c] for c in range(nlabel) if c < len(r) and r[c].strip()).strip()
                if not lab:
                    row_counter += 1; lab = f"Row {row_counter}"
            else:
                row_counter += 1; lab = f"Row {row_counter}"
            rec = {rowkey: lab}
            cells_with_num = 0
            for c in range(nlabel, ncol):
                if c >= len(r):
                    continue
                g, sk = colgroup.get(c), colsub.get(c)
                if not g and not sk:
                    continue
                cell = r[c].strip()
                if DOTS.match(cell) or not _is_num(cell):
                    continue
                v = _to_num(cell)
                if v is None:
                    continue
                cells_with_num += 1
                gk = g or "value"
                skk = sk or "value"
                slot = rec.get(gk)
                if not isinstance(slot, dict):
                    if gk in rec:            # group name collides with the row-label key
                        gk = gk + " (col)"
                        if not isinstance(rec.get(gk), dict):
                            rec[gk] = {}
                    else:
                        rec[gk] = {}
                rec[gk][skk] = v
            if cells_with_num >= 1:
                rows.append(rec)
        if len(rows) >= 1:
            sections.append({"name": name or "", "rows": rows})
    return sections


def to_oneshot(sections, title, src, method="mistral_ocr_generic"):
    groups = set()
    data = []
    for s in sections:
        for r in s["rows"]:
            for k, v in r.items():
                if isinstance(v, dict):
                    groups.add(k)
        if s["rows"]:
            data.append({"name": s["name"], "rows": s["rows"]})
    return {"source_image": src, "extraction_method": method,
            "metadata": {"title": title, "region": "", "year": 0, "data_type": "absolute",
                         "column_groups": sorted(groups)},
            "data": data,
            "acceptance": {"version": method, "decision": "accepted_high_confidence", "trust_score": 85}}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        secs = parse_sections(p)
        nrows = sum(len(s["rows"]) for s in secs)
        print(f"{os.path.basename(p)}: {len(secs)} sections, {nrows} rows")
        for s in secs[:6]:
            print(f"  [{s['name'][:30]}] {len(s['rows'])} rows; keys={list(s['rows'][0].keys()) if s['rows'] else []}")
            if s["rows"]:
                print("    ", json.dumps(s["rows"][0])[:220])
