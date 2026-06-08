export const meta = {
  name: 'tableVII-visual-qa',
  description: 'Opus checks Table VII matrices against their scans: do perfect ones truly match? are imperfect flags real?',
  phases: [{ title: 'QA' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/tableVII_qa_candidates.json'
const N = 48, BATCH = 1
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'matches_scan', 'verdict'],
  properties: {
    idx: { type: 'integer' },
    matches_scan: { type: 'boolean', description: 'do the extracted numbers match what the scan shows (spot a few cells)?' },
    has_all_civil_cols: { type: 'boolean', description: 'does the extraction capture all civil-condition columns the scan shows (Unmarried/Married/Widowed)?' },
    age_complete: { type: 'boolean', description: 'are all age bands on the scan captured?' },
    verdict: { enum: ['good', 'needs_reocr', 'uncertain'], description: 'good=faithful & complete; needs_reocr=misread or columns/rows dropped' },
    note: { type: 'string' },
  },
}
const INSTR = (d) => `Read ${FILE}; take ONLY entry index ${d}. It has scans (absolute paths to the source scan image(s)), the extracted matrix, and flagged_issues (why our validator flagged it, if any).
Open each scan with the Read tool and compare to the matrix. Judge:
- matches_scan: do the extracted numbers match the scan (check several cells)?
- has_all_civil_cols: are ALL civil-condition columns present (Unmarried, Married, Widowed) that the scan shows? (some scans are age x sex only with no civil condition — then this is true by default)
- age_complete: are all age bands captured?
- verdict: 'good' if faithful and complete; 'needs_reocr' if numbers are misread or columns/rows were dropped; 'uncertain' if unreadable.
Report idx exactly. Use StructuredOutput.`
phase('QA')
const res = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(INSTR(i), { label: `qa:${i}`, phase: 'QA', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
log(`Table VII QA ${v.length}/${N}: good=${v.filter(x=>x.verdict==='good').length} reocr=${v.filter(x=>x.verdict==='needs_reocr').length}`)
return { n: N, verdicts: v }
