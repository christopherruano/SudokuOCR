export const meta = {
  name: 'topup-pageread-haiku',
  description: 'Haiku reads page numbers off the 40 crops dropped from the main pass (100% coverage)',
  phases: [{ title: 'ReadPages' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/topup_candidates.json'
const N = 40, BATCH = 16
const nb = Math.ceil(N / BATCH)
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['reads'],
  properties: { reads: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['imghash', 'page_number', 'page_visible'],
    properties: {
      imghash: { type: 'string', description: 'copy the entry imghash exactly' },
      page_number: { type: 'string' }, page_visible: { type: 'boolean' },
      table_number: { type: 'string' }, confidence: { enum: ['high', 'medium', 'low'] },
    } } } },
}
const INSTR = (s, e) => `Read ${FILE} (JSON list). Process entries index ${s}..${e - 1}.
Each has imghash and img (path). Open each img (Read tool) and report imghash (copied exactly),
page_number (printed margin number exactly, or ""), page_visible, table_number, confidence.
Return all via StructuredOutput.`
phase('ReadPages')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i * BATCH, Math.min((i + 1) * BATCH, N)),
    { label: `t:${i}`, phase: 'ReadPages', schema: SCHEMA, model: 'haiku' })
)).then(rs => rs.filter(Boolean))
const reads = res.flatMap(r => r.reads || [])
return { n_reads: reads.length, reads }
