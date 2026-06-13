export const meta = {
  name: 'vision-place-read',
  description: 'Haiku reads the geographic place (taluk/district/division) printed on each scan header, to split over-grouped Table VII tables per place',
  phases: [{ title: 'Read' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/vision_place_candidates.json'
const N = 1205, BATCH = 16
const nb = Math.ceil(N / BATCH)
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['reads'],
  properties: { reads: { type: 'array', items: {
    type: 'object', additionalProperties: false, required: ['imghash', 'place'],
    properties: {
      imghash: { type: 'string', description: 'copy the entry imghash exactly' },
      place: { type: 'string', description: 'the geographic unit printed on this page (taluk/district/division/state/city), e.g. "Neyyattinkara Taluk", "Kolar District", "Eastern Division". "" if none visible.' },
      confidence: { enum: ['high', 'medium', 'low'] },
    } } } },
}
const INSTR = (s, e) => `Read ${FILE} (JSON list of census Table VII page crops). Process entries index ${s}..${e - 1}.
Each has imghash and img (path). Open each img (Read tool) and report the GEOGRAPHIC PLACE the page covers — the taluk / district / division / state / city named in the page heading or running header (e.g. "Neyyattinkara Taluk", "Kolar District", "Bangalore City", "Eastern Division", "State Summary"). Copy imghash exactly. If no place is visible on the page, set place="". Be concise and literal. Return all via StructuredOutput.`
phase('Read')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i * BATCH, Math.min((i + 1) * BATCH, N)),
    { label: `vp:${i}`, phase: 'Read', schema: SCHEMA, model: 'haiku' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const reads = res.flatMap(r => r.reads || [])
log(`vision place reads: ${reads.length}/${N} · ${reads.filter(r => r.place).length} with a place`)
return { n: N, reads }
