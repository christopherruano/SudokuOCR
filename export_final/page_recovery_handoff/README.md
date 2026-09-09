# PDF Page Recovery Handoff

This bundle is for the machine that can access the original census PDFs.

Goal:

1. Resolve local image hashes that do not yet have source PDF / PDF page provenance.
2. Render missing expected PDF pages into PNGs using the same deterministic image-hash naming convention as the existing pipeline.
3. Bring back small manifests plus any rendered PNGs so the local provenance workbook can be rebuilt.

## Files In This Bundle

- `cluster_page_recovery.py`: portable script to run on the PDF-access machine.
- `page_provenance_dictionary.csv`: current local image dictionary, including unresolved images.
- `table_vii_missing_pages.csv`: focused missing-page request for likely Table VII / age-sex-civil-condition pages.
- `missing_pdf_pages.csv`: broader missing-page request for all resolved table ranges.
- `unresolved_source_pdf_doc_hashes.csv`: doc hashes whose source PDFs are not yet known locally.
- `table_vii_accessibility.csv`: summary of Table VII candidate source groups and current status.

## Important Path Rule

Image hashes were generated as:

```text
sha1(canonical_source_pdf_path + ":" + pdf_page_number)
```

If you run this on FASRC and the PDFs are under `/n/home11/cruano/census_pdfs`, use that path directly.

If you run this on another computer where the same PDFs live somewhere else, you must provide both:

- `--pdf-root`: actual local directory containing PDFs.
- `--canonical-pdf-root`: the original path prefix used by the pipeline, usually `/n/home11/cruano/census_pdfs`.

The script renders from the actual path, but computes hashes from the canonical path.

## Best First Run: FASRC / Cluster

From a shell on the cluster:

```bash
cd ~/SudokuOCR
mkdir -p ~/page_recovery_handoff
```

Put this bundle into `~/page_recovery_handoff` somehow, then:

```bash
cd ~/page_recovery_handoff
python3 cluster_page_recovery.py \
  --pdf-root /n/home11/cruano/census_pdfs \
  --canonical-pdf-root /n/home11/cruano/census_pdfs \
  --page-provenance-csv page_provenance_dictionary.csv \
  --missing-pages-csv table_vii_missing_pages.csv \
  --resolve-unplaced \
  --render-missing \
  --out-dir page_recovery_table_vii
```

This does the focused Table VII pass.

Expected outputs:

- `page_recovery_table_vii/pdf_inventory.csv`
- `page_recovery_table_vii/resolved_unplaced_images.csv`
- `page_recovery_table_vii/rendered_missing_pages_manifest.csv`
- `page_recovery_table_vii/rendered_missing_pages/digitized_tables/.../*.png`

## Broader Run

Only do this after the focused Table VII pass looks sensible:

```bash
python3 cluster_page_recovery.py \
  --pdf-root /n/home11/cruano/census_pdfs \
  --canonical-pdf-root /n/home11/cruano/census_pdfs \
  --page-provenance-csv page_provenance_dictionary.csv \
  --missing-pages-csv missing_pdf_pages.csv \
  --resolve-unplaced \
  --render-missing \
  --out-dir page_recovery_all_ranges
```

This renders the broad worklist. Some pages may be blank/narrative/non-target because resolved table ranges can be broad.

## If Running On A Different Computer

Example:

```bash
python3 cluster_page_recovery.py \
  --pdf-root /Volumes/ExternalDrive/census_pdfs \
  --canonical-pdf-root /n/home11/cruano/census_pdfs \
  --page-provenance-csv page_provenance_dictionary.csv \
  --missing-pages-csv table_vii_missing_pages.csv \
  --resolve-unplaced \
  --render-missing \
  --out-dir page_recovery_table_vii
```

This assumes the relative layout under `/Volumes/ExternalDrive/census_pdfs` matches `/n/home11/cruano/census_pdfs`.

## Windows / PowerShell Example

Your current PDF tree appears to be:

```text
D:\census_pdfs\1871\...
D:\census_pdfs\1881\...
...
D:\census_pdfs\1951\...
D:\census_pdfs\census_1961\...
```

That mirrors the cluster layout for the historical census PDFs, so use:

```powershell
cd D:\page_recovery_handoff

py -3 .\cluster_page_recovery.py `
  --pdf-root D:\census_pdfs `
  --canonical-pdf-root /n/home11/cruano/census_pdfs `
  --page-provenance-csv .\page_provenance_dictionary.csv `
  --missing-pages-csv .\table_vii_missing_pages.csv `
  --resolve-unplaced `
  --render-missing `
  --out-dir .\page_recovery_table_vii
```

If `py -3` is not available, use:

```powershell
python .\cluster_page_recovery.py `
  --pdf-root D:\census_pdfs `
  --canonical-pdf-root /n/home11/cruano/census_pdfs `
  --page-provenance-csv .\page_provenance_dictionary.csv `
  --missing-pages-csv .\table_vii_missing_pages.csv `
  --resolve-unplaced `
  --render-missing `
  --out-dir .\page_recovery_table_vii
```

Before rendering, check that Poppler is available:

```powershell
pdftoppm -h
pdfinfo -h
```

If those fail, install Poppler for Windows and add its `bin` folder to `PATH`.

## What To Bring Back

Bring back the whole output folder:

```text
page_recovery_table_vii/
```

At minimum, bring back:

- `pdf_inventory.csv`
- `resolved_unplaced_images.csv`
- `rendered_missing_pages_manifest.csv`
- `rendered_missing_pages/`

If email size is limited, start by emailing only the three CSVs. The CSVs tell us:

- whether the unresolved images can be mapped to PDFs/pages,
- which missing pages rendered successfully,
- how big the rendered PNG folder is likely to be.

Then send the rendered PNGs in smaller zip chunks if needed.

## Packaging Commands

On the PDF machine:

```bash
tar -czf page_recovery_table_vii_manifests.tgz \
  page_recovery_table_vii/pdf_inventory.csv \
  page_recovery_table_vii/resolved_unplaced_images.csv \
  page_recovery_table_vii/rendered_missing_pages_manifest.csv
```

For PNGs:

```bash
tar -czf page_recovery_table_vii_pngs.tgz \
  page_recovery_table_vii/rendered_missing_pages
```

If too large for email, split into 40 MB chunks:

```bash
split -b 40m page_recovery_table_vii_pngs.tgz page_recovery_table_vii_pngs.tgz.part_
```

Recombine locally:

```bash
cat page_recovery_table_vii_pngs.tgz.part_* > page_recovery_table_vii_pngs.tgz
tar -xzf page_recovery_table_vii_pngs.tgz
```

## After Bringing Outputs Back Here

Place returned outputs somewhere like:

```text
export_final/page_recovery_returned/page_recovery_table_vii/
```

Then rebuild the provenance workbook locally:

```bash
python3 build_page_provenance_workbook.py
```

If rendered PNGs should be folded into the canonical local image tree, we can copy them into a new explicit root and rerun the workbook with:

```bash
python3 build_page_provenance_workbook.py \
  --image-root export/digitized_tables_referenced/digitized_tables \
  --image-root export/never_ocrd_images/digitized_tables \
  --image-root export_final/page_recovery_returned/page_recovery_table_vii/rendered_missing_pages/digitized_tables
```
