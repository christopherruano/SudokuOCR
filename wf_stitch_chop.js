export const meta = {
  name: 'stitch-chop-multipage',
  description: 'AI chops each volume\'s page-ordered table titles into logical multi-page tables (Imperial Table VII across districts, etc.)',
  phases: [{ title: 'Chop', detail: 'one agent per volume' }, { title: 'Synthesize' }],
}

const DIR = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/stitch_chop_docs'
const N = 209, BATCH = 1
const nb = Math.ceil(N / BATCH)

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['dochash', 'groups'],
  properties: {
    dochash: { type: 'string' },
    groups: {
      type: 'array',
      description: 'logical tables: each a run of consecutive same-subject pages',
      items: {
        type: 'object', additionalProperties: false,
        required: ['subject', 'anchor_index', 'indices'],
        properties: {
          subject: { type: 'string', description: 'the table identity, e.g. "Table VII - Age, Sex and Civil Condition"' },
          table_number: { type: 'string', description: 'roman/arabic table number if clear, else ""' },
          anchor_index: { type: 'integer', description: 'the i of the FIRST page of this table (the one that introduces it, WITHOUT a continued/concluded marker)' },
          indices: { type: 'array', items: { type: 'integer' }, description: 'all member i values incl the anchor, in page order' },
          multipage: { type: 'boolean', description: 'true if this logical table spans >1 page/member' },
          confidence: { enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const INSTR = (d) => `Read ${DIR}/doc_${d}.json (ONE census volume).
It has dochash and a "sequence": the volume's tables in PRINTED-PAGE order, each with i (index), page, tno (OCR'd table number, often MIS-read), fam (family), title.
Your job: CHOP this sequence into LOGICAL multi-page tables. A logical table is a run of CONSECUTIVE pages that are the SAME table continued across pages/districts (e.g. "Table VII. Age, Sex and Civil Condition" repeated for district (6), (7), (8)... is ONE logical table).
Rules:
- Group by SUBJECT (the title), NOT by tno — tno is frequently mis-OCR'd (VII read as VI/VIII). Trust the title text.
- CONTINUATION MARKERS are decisive: a title containing "continued", "contd", "concluded", "concld", or "(contd.)" is NOT a new table — it is a CONTINUATION of the most recent earlier table with the same base subject. Attach it to THAT group; never let a "continued"/"concluded" page start a new group.
- The ANCHOR (first page) of a group is the page that INTRODUCES the subject WITHOUT a continuation marker. Set anchor_index to it. "Concluded" marks the LAST member of a group.
- Members must share the subject and run in page order. A genuinely different subject (e.g. "Table VI. Religion") starts a new group even if interleaved between continuation pages of another table.
- Use the district/part number in the title (e.g. "(6)","(7)") and page numbers to order members.
- Every row index i must land in exactly one group. Singletons are their own group (anchor_index = that i, multipage=false).
Return dochash and the groups (subject, table_number if clear, anchor_index, member indices in page order, multipage, confidence). Use StructuredOutput.`

phase('Chop')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i), { label: `chop:${i}`, phase: 'Chop', schema: SCHEMA, model: 'sonnet' })
)).then(rs => rs.filter(Boolean))

phase('Synthesize')
const docs = res.filter(Boolean)
const mp = docs.flatMap(d => d.groups || []).filter(g => g.multipage)
log(`chopped ${docs.length} volumes -> ${mp.length} multi-page logical tables`)
return { n_docs: docs.length, n_multipage: mp.length, results: docs }
