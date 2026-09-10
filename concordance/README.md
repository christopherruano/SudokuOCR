# District concordance pipeline

Deterministic, re-runnable pipeline mapping the historical census geographies to
modern administrative areas. No output file is ever hand-edited; every manual
decision passes through `overrides.csv` with a justification and reviewer.

## Phase 1 — classification (this directory)
- `classify_geo.py` — classifies all 615 catalog geographies. Rule provenance is
  recorded per row (`class_rule`, `class_evidence`).
- `overrides.csv` — reviewed manual decisions (authoritative; applied by the script).
- `proposals_claude.csv` — LLM-proposed classifications for the 73 names no rule
  resolves, each with justification + confidence, `status=pending_review`.
  Promote a row to `overrides.csv` only after human review.
- `out/geo_classified.csv`, `out/district_universe.csv`, `out/unresolved_queue.csv`.

Entity classes: the geo_reference v1 types plus `urban_section` (city census
wards), `aggregate` (multi-unit rows as printed), `natural_division` (census
natural divisions), `not_geography`.

## Evidence tiers (used from Phase 2 on)
A exact/contiguous successor · B documented rename/split · C many-to-many with
area shares · LLM-proposed rows are never authoritative until reviewed.
