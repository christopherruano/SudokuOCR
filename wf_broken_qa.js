export const meta = {
  name: 'broken-tableVII-severity',
  description: 'Opus classifies HOW broken the structurally-failing Table VIIs are: truncated / misread / partial-multipage / not-a-table / bad-scan',
  phases: [{ title: 'Classify' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/broken_qa_candidates.json'
const N = 28
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'severity', 'fixable_by_reocr'],
  properties: {
    idx: { type: 'integer' },
    scan_quality: { enum: ['clear', 'faint', 'rotated', 'low_res', 'illegible'] },
    is_real_tableVII: { type: 'boolean', description: 'is the scan actually an Age/Sex/Civil-Condition table (vs a diagram/text/other table)?' },
    severity: { enum: ['minor', 'moderate', 'severe', 'not_a_table'],
      description: 'minor=mostly right, few cells/cols off; moderate=a column or section dropped; severe=most data wrong/missing; not_a_table=scan is not a Table VII' },
    whats_wrong: { type: 'string', description: 'one phrase: what is actually broken vs the scan' },
    fixable_by_reocr: { type: 'boolean', description: 'would re-OCR of THIS crop likely fix it (true), or is it a bad/cut crop or non-table that re-OCR cannot fix (false)?' },
  },
}
const INSTR = (d) => `Read ${FILE}; take ONLY entry index ${d}. It has scans (paths to the source scan) and the extracted matrix. Our validator flagged this Table VII as structurally broken (missing age bands / sex / values).
Open each scan and judge HOW broken it is vs the scan:
- scan_quality, is_real_tableVII
- severity: minor / moderate / severe / not_a_table
- whats_wrong: one phrase
- fixable_by_reocr: would re-OCR of this same crop likely fix it, or is the crop cut-off / rotated / not-a-table (re-OCR won't help)?
Report idx exactly. Use StructuredOutput.`
phase('Classify')
const res = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(INSTR(i), { label: `bq:${i}`, phase: 'Classify', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
log(`broken-severity ${v.length}/${N}`)
return { n: N, verdicts: v }
