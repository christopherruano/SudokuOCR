#!/usr/bin/env python3
"""Refresh the human-facing provenance workbook (xlsx) from the live CSV master.

page_number_status.csv is the machine source of truth (one row per page: where it is, its source PDF
+ pdf_page, Table-VII flag, role, which assembly run / sub-group, review verdict). This mirrors it into
page_provenance_workbook.xlsx with a Summary sheet, so the workbook reflects the latest OCR/assembly
work. Best-effort; the overnight orchestrator tolerates failure.
  python3 refresh_provenance_workbook.py
"""
import os, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/page_number_status.csv")
XLSX = os.path.join(ROOT, "export_final/page_provenance/page_provenance_workbook.xlsx")


def main():
    df = pd.read_csv(CSV, dtype=str).fillna("")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    vii = df[df["is_tableVII"] == "1"]
    in_asm = df[df["assembly_run"] != ""]
    reviewed = df[df["review_verdict"] != ""]
    summary = pd.DataFrame({
        "metric": ["generated", "total pages", "Table VII pages", "pages in an assembly run",
                   "distinct assembly runs", "pages with a review verdict",
                   "pages with a PDF page number", "verdict: correct", "verdict: wrong/boundary"],
        "value": [stamp, len(df), len(vii), len(in_asm),
                  in_asm["assembly_run"].nunique(), len(reviewed),
                  int((df["page_status"] != "unknown").sum()),
                  int((reviewed["review_verdict"] == "correct").sum()),
                  int(reviewed["review_verdict"].isin(["wrong", "boundary"]).sum())],
    })

    os.makedirs(os.path.dirname(XLSX), exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        df.to_excel(xl, sheet_name="pages", index=False)
        vii.to_excel(xl, sheet_name="tableVII", index=False)
        in_asm.sort_values(["assembly_run", "run_position"]).to_excel(xl, sheet_name="assemblies", index=False)
    print(f"workbook refreshed -> {XLSX}  ({len(df)} pages, {len(vii)} Table VII, {in_asm['assembly_run'].nunique()} runs) @ {stamp}")


if __name__ == "__main__":
    main()
