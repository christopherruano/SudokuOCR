export const meta = {
  name: 'newcrop-pageread-haiku',
  description: 'Haiku reads printed page number + table number off 1,914 newly-OCR\'d Table VII crops, to fold them into the multi-page stitch groups',
  phases: [{ title: 'ReadPages' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/newcrop_pageread.json'
const N = 1914, BATCH = 16
const nb = Math.ceil(N / BATCH)
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['reads'],
  properties: { reads: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['imghash', 'page_number', 'page_visible'],
    properties: {
      imghash: { type: 'string', description: 'copy the entry imghash exactly' },
      page_number: { type: 'string', description: 'printed margin page number exactly, or ""' },
      page_visible: { type: 'boolean' },
      table_number: { type: 'string', description: 'table identity from heading (e.g. VII), or ""' },
      confidence: { enum: ['high', 'medium', 'low'] },
    } } } },
}
const INSTR = (s, e) => `Read ${FILE} (JSON list of scanned census table crops). Process entries index ${s}..${e - 1}.
Each has imghash and img (absolute path). Open each img (Read tool) and report:
- imghash: copy exactly. - page_number: printed margin page number (top/bottom), exactly, or "". - page_visible. - table_number (heading identity, e.g. VII). - confidence.
Many won't show a page number — set page_visible=false. Return all via StructuredOutput.`
phase('ReadPages')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i * BATCH, Math.min((i + 1) * BATCH, N)),
    { label: `np:${i}`, phase: 'ReadPages', schema: SCHEMA, model: 'haiku' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const reads = res.flatMap(r => r.reads || [])
log(`new-crop page reads: ${reads.length} / ${N} · ${reads.filter(r => r.page_visible).length} visible`)
return { n: N, n_reads: reads.length, reads }
