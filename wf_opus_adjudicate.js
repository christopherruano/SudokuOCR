export const meta = {
  name: 'opus-adjudicate-keeper-badreads',
  description: 'Opus vision-checks 413 flagged keeper tables (scan vs extracted matrix): completeness, column alignment, values-match -> keep vs re-OCR',
  phases: [{ title: 'Adjudicate', detail: 'opus scan-vs-matrix per flagged table' }, { title: 'Synthesize' }],
}

const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/opus_adjudicate_candidates.json'
const N = 413, BATCH = 3
const nb = Math.ceil(N / BATCH)

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['idx', 'completeness', 'columns_ok', 'values_match', 'pattern_real', 'verdict'],
        properties: {
          idx: { type: 'integer', description: 'the entry idx, copied exactly' },
          legible: { type: 'boolean', description: 'is the scan itself legible enough to read?' },
          completeness: { enum: ['complete', 'dropped_rows', 'dropped_cols', 'severely_incomplete'],
            description: 'did the extraction capture every row & column present on the scan?' },
          columns_ok: { type: 'boolean', description: 'are the columns aligned to the correct headers (no shifted/mislabeled columns)?' },
          values_match: { enum: ['match', 'partial', 'mismatch'],
            description: 'do the extracted numbers loosely match what the scan shows?' },
          pattern_real: { type: 'boolean', description: 'is the flagged pattern (all-zero / constant column) what the source ACTUALLY shows? true=real, false=misread' },
          verdict: { enum: ['good', 'reocr', 'uncertain'],
            description: 'good=keep on explorer as-is; reocr=bad read, needs re-OCR; uncertain=cannot tell' },
          note: { type: 'string', description: 'one short phrase of evidence' },
        },
      },
    },
  },
}

const INSTR = (s, e) => `Read ${FILE} (JSON list of flagged census table extractions). Process entries index ${s}..${e - 1}.
Each entry has: idx, img (absolute path to the SOURCE SCAN), title, reasons (why it was flagged), detail, and extracted_matrix (what our OCR pulled out, as a compact grid).
These were flagged as all-zero, single-value, or having a constant/repeated column — classic signs of a BAD READ, BUT some may be genuinely correct (a sparse table really is mostly zero, or a column really is constant).
For EACH entry: open its img with the Read tool and compare the SCAN against the extracted_matrix. Judge:
- legible: can you read the scan at all?
- completeness: did we capture every row and column on the scan? (dropped_rows / dropped_cols / severely_incomplete if not)
- columns_ok: are columns aligned to the right headers (not shifted/swapped)?
- values_match: do the extracted numbers loosely match the scan (match / partial / mismatch)?
- pattern_real: is the flagged all-zero/constant pattern what the source ACTUALLY shows (true), or did the OCR misread real numbers as zeros/repeats (false)?
- verdict: 'good' if the extraction faithfully matches the scan (keep it); 'reocr' if it's a bad read that needs re-OCR; 'uncertain' if you truly cannot tell.
- note: one short phrase.
Report idx EXACTLY as given. Be quick but look carefully at whether numbers were dropped or misread. Return all via StructuredOutput.`

phase('Adjudicate')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i * BATCH, Math.min((i + 1) * BATCH, N)),
    { label: `adj:${i}`, phase: 'Adjudicate', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))

phase('Synthesize')
const verdicts = res.flatMap(r => r.verdicts || [])
const good = verdicts.filter(v => v.verdict === 'good').length
const reocr = verdicts.filter(v => v.verdict === 'reocr').length
const unc = verdicts.filter(v => v.verdict === 'uncertain').length
log(`adjudicated ${verdicts.length}/${N}: good=${good} reocr=${reocr} uncertain=${unc}`)
return { n: N, n_verdicts: verdicts.length, good, reocr, uncertain: unc, verdicts }
