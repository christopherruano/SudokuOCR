export const meta = {
  name: 'pilot-verify-reread',
  description: 'Opus verifies accepted re-reads against scans: is the NEW matrix accurate, and did it fix the OLD one\'s problem areas?',
  phases: [{ title: 'Verify' }, { title: 'Synthesize' }],
}
const FILE = '/Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/pilot_verify_candidates.json'
const N = 16
const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'new_matches_scan', 'verdict'],
  properties: {
    idx: { type: 'integer' },
    new_matches_scan: { enum: ['matches', 'mostly', 'mismatch'], description: 'does the NEW (re-read) matrix match the scan on spot-checked cells?' },
    new_more_complete: { type: 'boolean', description: 'does the NEW matrix capture columns/rows the OLD one was missing?' },
    old_problem: { type: 'string', description: 'what the OLD matrix got wrong/dropped (one phrase)' },
    verdict: { enum: ['reread_better', 'reread_accurate_same', 'reread_has_errors', 'old_was_fine'],
      description: 'reread_better=fixed the old problems AND accurate; reread_has_errors=new matrix has wrong digits; old_was_fine=no real improvement needed' },
    note: { type: 'string' },
  },
}
const INSTR = (d) => `Read ${FILE}; take ONLY entry index ${d}. It has scans (paths to the source scan), new_matrix (our Opus RE-READ), and old_matrix (the ORIGINAL OCR we are trying to improve).
Open the scan(s) and judge by comparing BOTH matrices to what the scan actually shows:
- new_matches_scan: spot-check several cells of the NEW matrix against the scan — matches / mostly / mismatch.
- new_more_complete: does NEW capture columns or rows that OLD dropped?
- old_problem: one phrase on what OLD got wrong (dropped Widowed, truncated ages, misread digits...).
- verdict: reread_better (NEW fixed OLD's problems and its digits are right) | reread_accurate_same (both fine) | reread_has_errors (NEW has wrong digits) | old_was_fine.
Be a strict checker — actually read digits off the scan. Report idx exactly. Use StructuredOutput.`
phase('Verify')
const res = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(INSTR(i), { label: `pv:${i}`, phase: 'Verify', schema: SCHEMA, model: 'opus' })
)).then(rs => rs.filter(Boolean))
phase('Synthesize')
const v = res.filter(Boolean)
const better = v.filter(x => x.verdict === 'reread_better').length
const errs = v.filter(x => x.verdict === 'reread_has_errors').length
log(`pilot ${v.length}/${N}: better=${better} errors=${errs}`)
return { n: N, verdicts: v }
