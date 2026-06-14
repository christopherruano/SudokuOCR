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
> 2. **Determine the Table VII page range per volume:**
>    - If `metadata.json` exists: for each table dir whose title/topic mentions VII /
>      "Age, Sex and Civil Condition" / C-III, read `queue_row.resolved_start_page` and
>      `resolved_end_page`. The volume's Table VII range = min(start)..max(end) over
>      those tables. Also collect the pages ALREADY rendered (`rendered_pages[].pdf_page`).
>    - If no metadata: use the file `render_targets.csv` I'll give you (columns
>      `source_pdf, render_from, render_to`) — render that whole range per PDF.
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
