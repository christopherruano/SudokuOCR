# Flash-drive task: render every missing Table VII page to PNG

Run Claude Code on the computer that has the flash drive mounted, in (or pointed at)
the drive. Paste the brief below. Goal: produce PNGs of every Table VII page we're
missing, plus a tiny manifest, to copy to the main machine so there are NO gaps.

## Set up Claude Code on that computer
```bash
npm install -g @anthropic-ai/claude-code      # or: curl -fsSL https://claude.ai/install.sh | sh
cd /path/to/flash-drive                        # the drive with the census PDFs (+ maybe digitized_tables/)
claude
```

## Paste this as the task
> I have a flash drive mounted here with the original Indian census volume PDFs
> (named like `saoa.crl.NNNNNNNN.pdf`), and possibly the digitization outputs
> (`digitized_tables/<dochash>_t*/metadata.json`). I need to render every **Table VII
> (Age, Sex & Civil Condition)** page so none are missing, and package them to copy to
> my main machine.
>
> Do this:
> 1. **Inventory the drive.** Find all the `*.pdf` files and any `digitized_tables/*/metadata.json`.
>    Tell me which you found (PDFs only, or PDFs + metadata).
> 2. **Find the Table VII pages by CONTENT (authoritative — do NOT just trust the TOC).**
>    The page ranges we have came from a TOC parse that may be incomplete, so verify
>    against the actual pages:
>    - For each PDF, extract per-page text: `pdftotext -f P -l P -layout "<pdf>" -` page
>      by page (or `pdftotext -layout "<pdf>" out.txt` which form-feed-splits pages).
>    - A page IS Table VII if its text (esp. the top/header) matches any of:
>      `TABLE\s*VII\b`, `Age,?\s*Sex\s*(and|&)?\s*Civil\s*Condition`, `Civil\s*Condition\s*by\s*Age`,
>      `\bC[-\s.]*III\b` (the post-1931 C-III form). Watch for roman/garbled headers.
>    - **Include continuation pages:** Table VII spans consecutive pages and continuation
>      pages may lack a clear header. So once you find the Table VII run, include every
>      page from the first match to the last match **inclusive** (fill interior pages),
>      plus a ±1 page margin.
>    - If a PDF has NO text layer (pdftotext returns blank), fall back to the ranges in
>      `render_targets.csv` (columns `source_pdf, render_from, render_to`) and render the
>      whole padded range — we'll classify on the main machine.
>    - Cross-check: if `metadata.json` exists, compare your content-detected pages to its
>      `resolved_start..end` and to `rendered_pages[].pdf_page`; render anything you found
>      that wasn't already rendered. Report any volume where content disagrees with the TOC
>      range (that's a volume the first pass got wrong).
> 3. **Render the pages.** For every page in each volume's Table VII range that is NOT
>    already rendered, render it to PNG with:
>    `pdftoppm -png -r 300 -f P -l P "<pdf>" out/<dochash>_p<P>`
>    (300 DPI; one page at a time). Name each file `<dochash>_p<pagenumber>.png`.
>    If `metadata.json` told you which pages already exist, SKIP those (render only gaps);
>    otherwise render the whole range — duplicates are harmless, the main machine dedups.
> 4. **Write `render_manifest.csv`** with columns: `filename, dochash, pdf_page,
>    source_pdf, year, province` — one row per rendered PNG.
> 5. **Package** `out/` + `render_manifest.csv` into one folder (or a .tar.gz) for me to copy.
>
> Be thorough — the whole point is ZERO missing Table VII pages. Report how many pages
> you rendered, across how many volumes, and any PDFs you couldn't open.

## Copy back to the main machine
Put the rendered `out/` folder + `render_manifest.csv` somewhere I can pick them up
(USB to the main machine, or a cloud direct-download link). On the main machine I'll
ingest them: dedup against what we already have (by volume+page), OCR the new pages
with Gemini, and assemble — completing every volume's Table VII.

## Notes
- 300 DPI is good for OCR; the table digits stay crisp.
- `render_targets.csv` (181 volumes, the authoritative Table VII ranges) ships alongside
  this file — use it if the drive has no `metadata.json`.
- Don't render entire PDFs unless a volume has no range info — that wastes time; the
  Table VII range is what matters.
