export const meta = {
  name: 'spotverify-stitch-groups',
  description: 'Opus verifies sampled multi-page stitch groups against member scans: same subject? correct grouping? page order?',
  phases: [{ title: 'Verify' }, { title: 'Synthesize' }],
}

const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/spotverify_candidates.json'
const N = 64, BATCH = 1
const nb = Math.ceil(N / BATCH)

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'same_subject', 'verdict'],
  properties: {
    idx: { type: 'integer' },
    same_subject: { type: 'boolean', description: 'do ALL the shown scans show the same table/subject?' },
    same_row_structure: { type: 'boolean', description: 'do they share the same row scheme (e.g. same age bands)?' },
    page_order_ok: { type: 'boolean', description: 'are the members consistent with one continuous table?' },
    verdict: { enum: ['correct', 'over_grouped', 'wrong_members', 'uncertain'],
      description: 'correct=one true table; over_grouped=distinct tables wrongly merged (needs split); wrong_members=some pages do not belong' },
    note: { type: 'string', description: 'one short phrase of evidence' },
  },
}

const INSTR = (d) => `Read ${FILE} (JSON list). Take ONLY the entry at index ${d}.
It is a proposed MULTI-PAGE TABLE group: a "subject", the member "pages", and "member_images" (absolute paths to the source scans of the anchor + sample member pages).
Open EACH member image with the Read tool and judge whether they truly form ONE logical multi-page table:
- same_subject: do all scans show the SAME table/subject (e.g. all "Table VII. Age, Sex and Civil Condition" for the same taluk/district series)?
- same_row_structure: do they share the same row scheme (same age bands down the side, or same row stubs)?
- page_order_ok: are they consistent with one table continued across pages?
- verdict: 'correct' if they're one table; 'over_grouped' if these are actually DIFFERENT tables wrongly merged; 'wrong_members' if some pages clearly don't belong; 'uncertain' if you can't tell.
Report idx exactly. Look at the titles and the row/column structure in the scans. Use StructuredOutput.`

phase('Verify')
const res = await parallel(Array.from({ length: nb }, (_, i) => () =>
  agent(INSTR(i), { label: `sv:${i}`, phase: 'Verify', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))

phase('Synthesize')
const v = res.filter(Boolean)
const correct = v.filter(x => x.verdict === 'correct').length
log(`spot-verified ${v.length}/${N}: correct=${correct} over_grouped=${v.filter(x=>x.verdict==='over_grouped').length} wrong=${v.filter(x=>x.verdict==='wrong_members').length}`)
return { n: N, n_verdicts: v.length, correct, verdicts: v }
