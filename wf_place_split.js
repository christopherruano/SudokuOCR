export const meta = {
  name: 'place-split-assembled',
  description: 'AI assigns each member page of an over-grouped Table VII to its geographic place (taluk/district), carrying the place across continuation pages, so the assembly can split per place',
  phases: [{ title: 'Split' }, { title: 'Synthesize' }],
}
const DIR = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/place_split_docs'
const INPUT = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/place_split_input.json'
const N = 35
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['gid', 'assignments'],
  properties: {
    gid: { type: 'integer' },
    assignments: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['i', 'place'],
        properties: {
          i: { type: 'integer', description: 'the member index' },
          place: { type: 'string', description: 'the geographic place this page belongs to (e.g. "Vilavankod Taluk", "Bombay City"), normalized; same string for all pages of one place' },
        },
      },
    },
  },
}
const INSTR = (k) => `Read ${INPUT} (JSON array of multi-page table groups). Take the entry at ARRAY INDEX ${k} (the ${k}-th element, 0-based). Use its "gid" as the gid in your output.
Its "members" are page-ordered table pages, each with i, page, title. The pages belong to DIFFERENT geographic places (taluks/districts) — but a CONTINUATION page often omits the place name (it was printed on an earlier page's header).
Assign each member i to its geographic PLACE:
- Read the place from the title when present (e.g. "(6) Neyyattinkara Taluk", "Bombay City", "Hazara District").
- When a page's title has NO place (continuation/sub-block like "B-HINDU" or "Age Returns"), it belongs to the SAME place as the most recent preceding page that did name one.
- Use ONE consistent normalized place string per place (e.g. "Neyyattinkara Taluk").
Return gid and an assignment {i, place} for EVERY member. Use StructuredOutput.`
phase('Split')
// gids are not 0..N-1; read them from the input order
const res = await parallel(Array.from({ length: N }, (_, k) => () =>
  agent(INSTR(k), { label: `ps:${k}`, phase: 'Split', schema: SCHEMA, model: 'sonnet' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
const places = new Set()
v.forEach(x => (x.assignments || []).forEach(a => places.add(x.gid + '|' + a.place)))
log(`place-split ${v.length}/${N} groups -> ${places.size} (group,place) pairs`)
return { n: N, results: v }
