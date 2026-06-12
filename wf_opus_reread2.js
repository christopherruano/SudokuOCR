export const meta = {
  name: 'opus-reread-tableVII-default',
  description: 'Opus re-reads the 340 non-templated imperfect Table VII tables; classifies the table TYPE from its title (count/distribution/rate/age-sex) and extracts the full grid accordingly',
  phases: [{ title: 'Reread' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/opus_reread_candidates.json'
const START = 134, COUNT = 340   // the 'default' (non-templated) candidates
const ROW = {
  type: 'object', additionalProperties: false, required: ['age'],
  properties: {
    age: { type: 'string' },
    pop_m: { type: ['number', 'null'] }, pop_f: { type: ['number', 'null'] },
    unm_m: { type: ['number', 'null'] }, unm_f: { type: ['number', 'null'] },
    mar_m: { type: ['number', 'null'] }, mar_f: { type: ['number', 'null'] },
    wid_m: { type: ['number', 'null'] }, wid_f: { type: ['number', 'null'] },
  },
}
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'is_tableVII', 'table_kind', 'sections'],
  properties: {
    idx: { type: 'integer' },
    legible: { type: 'boolean' },
    is_tableVII: { type: 'boolean', description: 'is this an age table (age x sex, with or without civil condition)?' },
    table_kind: { enum: ['count', 'distribution', 'rate', 'ratio', 'percentage', 'age_sex_only', 'other'],
      description: 'classify from the TITLE/HEADERS: count=raw counts (U+M+W=Population); distribution=per 1,000/10,000 "of each sex" (U+M+W sums to that base); rate/ratio/percentage=do not sum; age_sex_only=age x sex counts with NO civil condition' },
    base: { type: ['integer', 'null'], description: 'for distribution: the printed base (1000/10000), else null' },
    sections: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['name', 'rows'],
      properties: { name: { type: 'string' }, rows: { type: 'array', items: ROW } } } },
  },
}
const INSTR = (d) => `Read ${FILE}; take ONLY entry index ${d}. It has scans (paths) and issues (why our prior OCR was incomplete).
Open EVERY scan and FIRST classify the table by READING ITS TITLE/HEADERS:
- table_kind: count (raw counts — Unmarried+Married+Widowed = Population) | distribution (per 1,000 / per 10,000 "of each sex" — U+M+W sums to that base, set base) | rate | ratio | percentage | age_sex_only (age x sex counts, NO civil-condition columns) | other.
Then transcribe FULLY and accurately:
- Capture EVERY age-band row. Capture all columns the table actually has: for civil-condition tables that's Population/Total, Unmarried, Married, Widowed each x Males/Females; for age_sex_only it's just Population/Total x Males/Females (leave unm/mar/wid null).
- Fill the numeric fields you can read; null only for genuinely absent cells (a dash/nil = 0).
- If the page stacks religion/community/territory blocks, return EACH as its own section.
- Do NOT force civil-condition columns onto a table that doesn't have them, and do NOT assume sums to 1000 unless it's a distribution table.
Report idx exactly. If not an age table at all, is_tableVII=false, sections=[]. Use StructuredOutput.`
phase('Reread')
const res = await parallel(Array.from({ length: COUNT }, (_, i) => () =>
  agent(INSTR(START + i), { label: `rr2:${START + i}`, phase: 'Reread', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
const cells = v.reduce((a, x) => a + (x.sections || []).reduce((b, s) => b + (s.rows || []).length, 0), 0)
log(`re-read ${v.length}/${COUNT}; ${cells} age-rows`)
return { n: COUNT, n_done: v.length, reads: v }
