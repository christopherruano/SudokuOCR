export const meta = {
  name: 'opus-reread-tableVII',
  description: 'Opus vision re-reads imperfect Table VII tables to the FULL grid (all civil-condition cols x M/F, every religion section), validated by U+M+W=Population',
  phases: [{ title: 'Reread' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/opus_reread_candidates.json'
const N = 134   // templated batch first
const ROW = {
  type: 'object', additionalProperties: false, required: ['age'],
  properties: {
    age: { type: 'string', description: 'age band exactly as printed, e.g. "0-1","5-10","60 and over","Total"' },
    pop_m: { type: ['number', 'null'] }, pop_f: { type: ['number', 'null'] },
    unm_m: { type: ['number', 'null'] }, unm_f: { type: ['number', 'null'] },
    mar_m: { type: ['number', 'null'] }, mar_f: { type: ['number', 'null'] },
    wid_m: { type: ['number', 'null'] }, wid_f: { type: ['number', 'null'] },
  },
}
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'is_tableVII', 'sections'],
  properties: {
    idx: { type: 'integer' },
    legible: { type: 'boolean' },
    is_tableVII: { type: 'boolean', description: 'is this actually an Age/Sex/Civil-Condition table?' },
    sections: {
      type: 'array',
      description: 'one per religion/community/territory block on the page (or one "All" section)',
      items: {
        type: 'object', additionalProperties: false, required: ['name', 'rows'],
        properties: { name: { type: 'string' }, rows: { type: 'array', items: ROW } },
      },
    },
  },
}
const INSTR = (d) => `Read ${FILE}; take ONLY entry index ${d}. It has scans (absolute paths to the source scan image(s)), expected_columns (the civil-condition column groups this table SHOULD have), and expected_age_rows (the age bands a sibling page had). Our previous OCR of this table was INCOMPLETE (see "issues").
Open EVERY scan with the Read tool and transcribe the table FULLY and accurately:
- Capture EVERY age-band row (use expected_age_rows as a checklist; read what's actually printed).
- Capture ALL FOUR civil-condition columns — Population/Total, Unmarried, Married, Widowed — each split into Males and Females. Fill pop_m/pop_f/unm_m/unm_f/mar_m/mar_f/wid_m/wid_f. Use null only when a cell is genuinely absent/blank on the scan; a dash/nil means 0.
- If the page stacks multiple religion/community/territory blocks, return EACH as its own section with its name.
- Read digits carefully; do not invent or duplicate columns. Population should equal Unmarried+Married+Widowed per cell.
Report idx exactly. If the scan is not actually an Age/Sex/Civil-Condition table, set is_tableVII=false and sections=[]. Use StructuredOutput.`
phase('Reread')
const res = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(INSTR(i), { label: `rr:${i}`, phase: 'Reread', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
const cells = v.reduce((a, x) => a + (x.sections || []).reduce((b, s) => b + (s.rows || []).length, 0), 0)
log(`re-read ${v.length}/${N} tables, ${cells} age-rows extracted`)
return { n: N, n_done: v.length, reads: v }
