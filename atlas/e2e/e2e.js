const puppeteer = require('puppeteer-core');
const OUT = process.argv[2], BASE = 'https://heb91r-5a20dde1.pages.dev';
const wait = ms => new Promise(r => setTimeout(r, ms));
const results = []; const ok = (name, pass, detail='') => { results.push({name, pass, detail}); };
(async () => {
  const b = await puppeteer.launch({executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new'});
  const p = await b.newPage(); await p.setViewport({width: 1400, height: 950});
  const errs = []; p.on('pageerror', e => errs.push(p.url() + ' :: ' + e.message));
  p.on('requestfailed', r => { if (!r.url().includes('nel.cloudflare')) errs.push('REQFAIL ' + r.url()); });
  p.on('response', r => { if (r.status() >= 400) errs.push(r.status() + ' ' + r.url()); });
  const shot = n => p.screenshot({path: `${OUT}/${n}.png`});

  // 1 home + cards
  await p.goto(BASE + '/', {waitUntil: 'networkidle0'}); await shot('01_home');
  const cards = await p.$$eval('#homecards a', a => a.map(x => x.getAttribute('href')));
  ok('home: 4 cards', cards.length === 4, cards.join(','));
  const navs = await p.$$eval('nav.nav a', a => a.map(x => x.getAttribute('href')));
  for (const h of [...new Set(navs)]) { const r = await p.goto(BASE + '/' + h, {waitUntil: 'networkidle0'}); ok('nav link ' + h, r.status() === 200 && !(await p.title()).includes('— map') || h === 'index.html', await p.title()); }

  // 2 atlas: every year x layer
  await p.goto(BASE + '/atlas.html', {waitUntil: 'networkidle0'});
  for (const L of ['coverage', 'child', 'old', 'sexratio']) for (const y of ['all','1872','1881','1891','1901','1911','1921','1931','1941','1951']) {
    await p.click(`#layers button[data-k="${L}"]`); await p.click(`#years button[data-k="${y}"]`); await wait(120);
    const st = await p.evaluate(() => ({paths: document.querySelectorAll('path.u').length + document.querySelectorAll('path.h').length, hist: document.querySelectorAll('path.h').length, fills: new Set([...document.querySelectorAll('path.u,path.h')].map(x => x.getAttribute('fill'))).size, sum: document.getElementById('summary').innerText.slice(0, 60)}));
    const empty = /\b0 historical/.test(st.sum);   // years with no self-checked district data draw one fill
    ok(`atlas ${L} ${y}`, st.hist > 50 && (st.fills >= 2 || empty), `fills=${st.fills} ${st.sum.replace(/\n/g,' ')}`);
    if (['1881','1901','1931'].includes(y)) await shot(`02_atlas_${L}_${y}`);
  }
  // 2b historical units: click, subdivisions
  for (const [y, pid] of [['1901', 'gurgaon-district'], ['1891', 'poona-district'], ['1911', 'chanda-district'], ['1931', 'travancore-state']]) {
    await p.goto(BASE + '/atlas.html?y=' + y, {waitUntil: 'networkidle0'});
    const h = await p.$(`path.h[data-p="${pid}"]`); if (!h) { ok(`hist ${pid} ${y}`, false, 'not drawn'); continue; }
    await h.click(); await wait(200); const det = await p.$eval('#detail', e => e.innerText);
    await p.click('#detail button.btn'); await wait(1500);
    const sub = await p.evaluate(() => ({u: document.querySelectorAll('#subsvg path').length, l: document.querySelectorAll('#subsvg text').length}));
    ok(`hist ${pid} ${y}`, det.includes('today:') && sub.u > 0, `${det.split('\n')[0]} | subdiv units ${sub.u}, labels ${sub.l}`);
  }
  await p.goto(BASE + '/atlas.html?y=1901&b=modern', {waitUntil: 'networkidle0'});
  ok('modern mode draws 944 districts', (await p.$$eval('path.u', a => a.length)) === 944);
  // 3 click districts across regions -> inventory -> table -> scan + nav strip
  for (const name of ['Gurgaon', 'Chennai', 'Pune', 'Sylhet', 'Mandalay', 'Kanpur Nagar', 'Peshawar', 'Thiruvananthapuram']) {
    await p.goto(BASE + '/atlas.html?y=1901&b=modern', {waitUntil: 'networkidle0'});
    const gid = await p.evaluate(n => { const u = G.units.find(u => u.name.replace(/([a-z])([A-Z])/g,'$1 $2') === n || u.name === n.replace(/ /g,'')); return u && u.gid; }, name);
    if (!gid) { ok('atlas find ' + name, false, 'no unit'); continue; }
    await (await p.$(`path[data-g="${gid}"]`)).click(); await wait(200);
    const det = await p.$eval('#detail', e => e.innerText);
    const link = await p.$('#detail a[href^="place.html"]');
    ok(`click ${name}`, det.includes('modern district'), det.split('\n').slice(0, 4).join(' / '));
    if (!link) { ok(`${name}: inventory link`, false, 'no historical unit linked'); continue; }
    await Promise.all([p.waitForNavigation({waitUntil: 'networkidle0'}), link.click()]);
    const inv = await p.evaluate(() => ({h1: document.querySelector('h1')?.innerText, complete: !!document.querySelector('.complete'), rows: document.querySelectorAll('main table tr').length}));
    ok(`${name}: inventory page`, inv.complete && inv.rows > 1, `${inv.h1}, ${inv.rows} rows`);
    if (name === 'Gurgaon' || name === 'Sylhet') await shot(`03_place_${name}`);
    const t = await p.$('main a[href^="tables/"]');
    if (t) {
      await Promise.all([p.waitForNavigation({waitUntil: 'networkidle0'}), t.click()]);
      const tp = await p.evaluate(() => ({img: [...document.images].filter(i => i.src.includes('/img/')).map(i => i.complete && i.naturalWidth > 0), ctx: document.querySelector('.ctx')?.innerText || '', cells: document.querySelectorAll('table.mx td').length, csv: !!document.querySelector('a.dl')}));
      ok(`${name}: table page`, tp.img.length > 0 && tp.img.every(Boolean) && tp.cells > 0 && tp.csv, `imgs=${tp.img} cells=${tp.cells} ctx="${tp.ctx.slice(0, 90)}"`);
      if (name === 'Gurgaon') await shot('04_table_page');
      const nx = await p.$('.ctx a[href$=".html"]:not([href*="volume"]):not([href*="place"])');
      if (nx) { const r = await Promise.all([p.waitForNavigation({waitUntil: 'networkidle0'}), nx.click()]); ok(`${name}: prev/next table`, r[0].status() === 200 && !!(await p.$('table.mx')), p.url().slice(-60)); }
      const cov = await p.$('.ctx a[href*="place.html"]');
      if (cov) { await Promise.all([p.waitForNavigation({waitUntil: 'networkidle0'}), cov.click()]); ok(`${name}: covers -> place`, !!(await p.$('.complete')), await p.$eval('h1', e => e.innerText)); }
    }
  }
  // 4 census order
  await p.goto(BASE + '/volume.html', {waitUntil: 'networkidle0'}); await shot('05_volumes');
  const nv = await p.$$eval('a.vol', a => a.length); ok('volumes index', nv > 200, nv + ' volumes');
  const v = await p.$$eval('a.vol', a => a.map(x => x.getAttribute('href')));
  for (const h of [v[40], v[120], v[200]]) {
    await p.goto(BASE + '/' + h, {waitUntil: 'networkidle0'});
    const st = await p.evaluate(() => ({h1: document.querySelector('h1').innerText, rows: document.querySelectorAll('#tb tr').length, gaps: document.querySelectorAll('#tb tr.gap').length}));
    await p.click('#fa'); await wait(100);
    const vis = await p.$$eval('#tb tr', t => t.filter(x => x.style.display !== 'none').length);
    ok('volume ' + st.h1, st.rows > 2 && vis <= st.rows, `${st.rows} rows, ${st.gaps} gap rows, age-filter visible=${vis}`);
  }
  await shot('06_volume');
  // 5 places A-Z
  await p.goto(BASE + '/places.html', {waitUntil: 'networkidle0'});
  const n0 = await p.$eval('#cnt', e => e.innerText);
  await p.type('#q', 'madras'); await wait(150); const n1 = await p.$eval('#cnt', e => e.innerText);
  await p.select('#cl', ''); await wait(150); const n2 = await p.$eval('#cnt', e => e.innerText);
  await p.$eval('#q', e => { e.value=''; e.dispatchEvent(new Event('input')) }); await p.click('th[data-s="1901"]'); await wait(150);
  const top = await p.$eval('#tb tr:nth-child(2) td', e => e.innerText);
  ok('places A-Z filter/sort', n0 !== n1, `default ${n0}; "madras" ${n1}; all types ${n2}; top by 1901: ${top}`); await shot('07_places');
  // 6 search
  for (const q of ['kanpur', 'cawnpore 1901', 'table vii madras', 'trichinopoly', 'zzzzqq']) {
    await p.goto(BASE + '/search.html?q=' + encodeURIComponent(q), {waitUntil: 'networkidle0'});
    const r = await p.evaluate(() => ({cnt: document.getElementById('cnt').innerText, ph: document.querySelectorAll('.ph').length, first: document.querySelector('a.row')?.getAttribute('href') || '', empty: document.getElementById('res').innerText.slice(0, 80)}));
    ok('search "' + q + '"', q === 'zzzzqq' ? r.cnt.startsWith('0') && r.empty.includes('No table') : parseInt(r.cnt) > 0, `${r.cnt}, ${r.ph} place boxes`);
    if (q === 'kanpur') { await shot('08_search'); await Promise.all([p.waitForNavigation({waitUntil: 'networkidle0'}), p.click('a.row')]); ok('search -> table', !!(await p.$('table.mx')), p.url().slice(-50)); }
  }
  // 7 legacy pages
  for (const h of ['tree.html', 'best.html', 'coverage.html', 'ages.html', 'singleage.html', 'assembled.html', 'highlights.html', 'tags.html', 'area.html', 'data_guide.html']) {
    const r = await p.goto(BASE + '/' + h, {waitUntil: 'networkidle0'}); ok('legacy ' + h, r.status() === 200 && !!(await p.$('nav.nav')), await p.title());
  }
  // 8 phone width
  await p.setViewport({width: 390, height: 844, isMobile: true});
  for (const h of ['atlas.html?y=1901&l=child', 'place.html?k=gurgaon-district', 'search.html?q=kanpur']) {
    await p.goto(BASE + '/' + h, {waitUntil: 'networkidle0'});
    const w = await p.evaluate(() => document.documentElement.scrollWidth); ok('mobile ' + h, w <= 420, 'scrollWidth=' + w);
    await shot('09_mobile_' + h.split('.')[0]);
  }
  ok('no JS errors / failed requests anywhere', errs.length === 0, errs.slice(0, 8).join(' | '));
  const fails = results.filter(r => !r.pass);
  for (const r of results) if (!r.pass || /home|click|inventory|table page|volumes index|places|search|mobile|no JS/.test(r.name)) console.log((r.pass ? 'PASS ' : 'FAIL ') + r.name + ' — ' + r.detail);
  console.log(`\n${results.length - fails.length}/${results.length} passed`);
  await b.close();
})();
