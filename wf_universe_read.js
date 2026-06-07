export const meta = {
  name: 'universe-pageread-haiku',
  description: 'Haiku reads printed page numbers off 2,816 un-read scans in gap-bearing docs, to tell which MISSING pages we actually have',
  phases: [{ title: 'ReadPages', detail: 'haiku per-candidate page#' }, { title: 'Synthesize' }],
}

const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/universe_candidates.json'
const N = 2816, BATCH = 16
const nb = Math.ceil(N / BATCH)

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['reads'],
  properties: {
    reads: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['imghash', 'page_number', 'page_visible'],
        properties: {
          imghash: { type: 'string', description: 'the 40-hex imghash field of the entry, copied exactly' },
          page_number: { type: 'string', description: 'printed page number exactly as shown, or "" ' },
          page_visible: { type: 'boolean' },
          table_number: { type: 'string' },
          confidence: { enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const INSTR = (s, e) => `Read ${FILE} (JSON list of scanned census table images). Process entries index ${s}..${e - 1}.
Each entry has: imghash (40-hex id), img (absolute path), missing_pages (pages we are hunting for in this document).
For EACH entry: open its img with the Read tool and report:
- imghash: copy the entry's imghash field EXACTLY (do not alter).
- page_number: the PRINTED page number in the page margin (top/bottom corners), exactly as printed; "" if none visible.
- page_visible: true only if a printed page number is actually visible on the scan.
- table_number: table identity from the heading if shown (e.g. "VII"), else "".
- confidence.
Be fast and literal. Many scans won't show a page number — set page_visible=false. Return all via StructuredOutput.`

phase('ReadPages')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i * BATCH, Math.min((i + 1) * BATCH, N)),
    { label: `u:${i}`, phase: 'ReadPages', schema: SCHEMA, model: 'haiku' })
)).then(rs => rs.filter(Boolean))

phase('Synthesize')
const reads = res.flatMap(r => r.reads || [])
const visible = reads.filter(r => r.page_visible).length
log(`universe reads: ${reads.length} / ${N} · ${visible} show a page number`)
return { n_cands: N, n_reads: reads.length, page_visible: visible, reads }
