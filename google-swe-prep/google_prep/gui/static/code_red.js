/* Code Red: the exam-condition sprint screens (Today checklist, Re-Solve Queue, Progress). */
"use strict";
const CodeRed = (() => {
  const notCodeRed = (ctx, what) => mount(ctx, h('div', {class:'page'},
    pageTitle(`${what} is part of Code Red.`, 'Switch to Code Red for the final interview sprint.'),
    btn('Choose preparation mode', openPreparationMode, {cls:'primary'})));

  function startBlock(block) {
    const a = block.action || {};
    if (a.type === 'recall') return go(`#/recall?mode=${encodeURIComponent(a.mode)}`);
    if (a.type === 'mock') { S.boardStart = {problem_id:a.problem_id, practice:false}; return go('#/board'); }
    if (a.type === 'star') return go('#/star');
    if (a.type === 'behavioral') return behavioralDrill();
    return null;
  }

  async function behavioralDrill() {
    let data;
    try {data = await api('star_practice');} catch (err) {toast(err.message, true); return;}
    let left = data.seconds;
    const clock = h('b', {class:'cr-clock', 'data-test':'behavioral-clock'}, '2:00');
    const paint = () => { clock.textContent = `${Math.floor(Math.max(0, left) / 60)}:${String(Math.max(0, left) % 60).padStart(2, '0')}`; };
    const timer = setInterval(() => { left -= 1; paint(); if (left <= 0) clearInterval(timer); }, 1000);
    let done;
    try {
      done = await openModal(`Googleyness · ${data.label}`, h('div', null,
        h('p', {class:'cr-prompt'}, `“${data.prompt}”`),
        h('p', {class:'muted'}, 'Answer out loud: Situation, Task, Action, Result. Use “I”, give a number, name what you would change.'),
        h('div', {class:'row'}, h('span', {class:'muted small'}, 'Time left'), clock),
        data.stories.length ? h('p', {class:'muted small'}, 'Your stories here: ', data.stories.map((s) => s.title).join(' · '))
          : h('p', {class:'muted small'}, 'No story saved for this topic yet. Answer anyway, then write it down.')),
        [{label:'Not now', value:false}, {label:'I answered out loud', value:true, primary:true, test:'behavioral-done'}]);
    } finally { clearInterval(timer); }
    if (!done) return;
    try {await api('code_red_behavioral_done'); toast('Behavioral drill done.'); await route();}
    catch (err) {toast(err.message, true);}
  }

  function blockRow(block, index, isNext) {
    const label = block.done ? 'Done' : block.action && block.action.type === 'behavioral' ? 'Start drill'
      : block.action && block.action.type === 'star' ? 'Write a story' : block.action && block.action.type === 'recall' ? 'Start warmup' : 'Start 45-min round';
    return h('section', {class:`cr-block ${block.done ? 'is-done' : ''} ${isNext ? 'is-next' : ''}`.trim(), 'data-test':`cr-block-${block.id}`},
      h('span', {class:'cr-num', 'aria-hidden':'true'}, block.done ? '✓' : String(index + 1)),
      h('div', {class:'cr-copy'},
        h('div', {class:'row'}, h('b', null, block.title), h('span', {class:'muted small'}, `${block.minutes} min`)),
        block.problem ? h('div', {class:'row cr-problem'}, h('span', null, block.problem.title),
          h('span', {class:`chip ${block.problem.difficulty.toLowerCase()}`}, block.problem.difficulty),
          h('span', {class:'chip'}, block.problem.category), h('span', {class:'muted small mono'}, block.problem.id)) : null,
        h('p', {class:'muted small'}, block.detail)),
      block.done ? h('span', {class:'ok-t small'}, 'Done')
        : btn(label, () => startBlock(block), {cls: isNext ? 'primary' : 'ghost', primary:isNext || null, key: isNext ? 'Enter' : null,
          test:`cr-start-${block.id}`}));
  }

  async function today(ctx) {
    const st = S.state;
    const data = await api('code_red_today');
    setCrumbs(['Code Red', 'Today'], ctx);
    if (st.session) {
      const s = st.session;
      const resume = h('section', {class:'today-next', 'data-test':'next-card'},
        h('div', {class:'next-copy'}, h('h2', null, 'Pick up where you left off'),
          h('h3', null, s.text.replace(/^Session restored: /, '')),
          h('div', {class:'actions'}, btn('Resume session', () => resumeSession(s), {cls:'primary big', primary:true, key:'Enter', test:'resume-continue'}),
            btn('Discard session', async () => {
              if (!(await confirmBox('Discard this unfinished session?', 'Saved code files and completed grades are kept.', 'Discard session', 'Keep session', true))) return;
              try {await api('session_discard'); await route();} catch (err) {toast(err.message, true);}
            }, {cls:'ghost danger', test:'discard-session'}))));
      mount(ctx, h('div', {class:'page cr-page'}, pageTitle('Daily tactical checklist', 'Finish the open round first.'), resume));
      return {};
    }
    const next = data.blocks.findIndex((b) => !b.done);
    const evidence = data.readiness_provisional ? 'Early estimate: not enough mocks yet.' : 'Practice estimate from your self-graded work.';
    mount(ctx, h('div', {class:'page cr-page'},
      pageTitle('Daily tactical checklist', `${data.completed}/${data.total} done · ${data.days_left >= 0 ? `T-minus ${data.days_left} ${data.days_left === 1 ? 'day' : 'days'}` : 'target date passed'}`),
      h('section', {class:'cr-status', 'data-test':'cr-status'},
        h('div', null, h('span', {class:'muted small'}, 'Readiness estimate'), h('b', null, `${Math.round(data.readiness)}%`),
          h('span', {class:'muted small'}, `${evidence} Not a prediction of an offer.`)),
        h('div', null, h('span', {class:'muted small'}, 'Re-solves due'), h('b', null, String(data.queue_due)),
          btn('Open queue', () => go('#/queue'), {cls:'ghost small', test:'cr-open-queue'})),
        h('div', null, h('span', {class:'muted small'}, 'Today'), h('b', null, `${data.completed}/${data.total}`), bar(data.completed / Math.max(1, data.total), 'red'))),
      data.notes.length ? h('section', {class:'cr-notes'}, data.notes.map((n) => h('p', null, n))) : null,
      h('div', {class:'cr-blocks'}, data.blocks.map((b, i) => blockRow(b, i, i === next))),
      next < 0 ? h('section', {class:'card'}, h('h2', null, 'Mission complete.'),
        h('p', {class:'muted'}, 'Stop here. Rest is part of the sprint. Extra time? Clear one problem from the Re-Solve Queue.')) : null));
    return {onKey(e, typing) {
      if (!typing && e.key === 'Enter' && !e.ctrlKey && !e.altKey && next >= 0) { startBlock(data.blocks[next]); return true; }
      return false;
    }};
  }

  function whenText(row) {
    if (row.due) return row.overdue_days ? `Overdue ${plural(row.overdue_days, 'day')}` : 'Due now';
    return `Back ${fmtDate(row.next_review.slice(0, 10))}`;
  }

  ROUTES.queue = async (ctx) => {
    if (!isCodeRed()) return notCodeRed(ctx, 'The Re-Solve Queue');
    const data = await api('code_red_queue');
    setCrumbs(['Code Red', 'Re-Solve Queue'], ctx);
    const rows = data.rows.map((r) => h('tr', {'data-test':`queue-${r.id}`},
      h('td', {class:'mono small'}, r.id), h('td', null, r.title),
      h('td', null, h('span', {class:`chip ${r.difficulty.toLowerCase()}`}, r.difficulty)),
      h('td', {class:'muted'}, r.category), h('td', null, r.reason),
      h('td', {class: r.due ? 'red-t' : 'muted small'}, whenText(r)),
      h('td', null, btn(r.due ? 'Re-solve now' : 'Re-solve early', () => { S.boardStart = {problem_id:r.id, practice:false}; go('#/board'); },
        {cls: r.due ? 'primary small' : 'ghost small', test:`queue-start-${r.id}`}))));
    mount(ctx, h('div', {class:'page cr-page'},
      pageTitle('Re-Solve Queue', `${data.due_count} due now · every problem that beat you, most urgent first.`),
      h('section', {class:'card'}, rows.length
        ? h('table', {class:'table'}, h('thead', null, h('tr', null, ['ID', 'Problem', 'Level', 'Topic', 'Why', 'When', ''].map((x) => h('th', null, x)))), h('tbody', null, rows))
        : h('p', {class:'muted', 'data-test':'queue-empty'}, 'Nothing has beaten you yet. Failed, hinted or slow rounds land here.'))));
    return {};
  };

  async function progress(ctx) {
    const data = await api('code_red_progress');
    setCrumbs(['Code Red', 'Progress'], ctx);
    const scored = data.trend.filter((d) => d.average != null);
    const trend = h('div', {class:'cr-trend', role:'img', 'aria-label':`Mock score average per day, last ${data.trend.length} days`},
      data.trend.map((d) => h('div', {class:'cr-trend-day'},
        h('div', {class:'cr-trend-bar'}, h('i', {vars:{'--h': `${d.average == null ? 0 : Math.max(4, d.average)}%`}})),
        h('b', null, d.average == null ? '–' : String(Math.round(d.average))),
        h('span', {class:'muted small'}, new Date(`${d.day}T00:00:00`).toLocaleDateString(undefined, {weekday:'short'})))));
    const maxMistake = Math.max(1, ...data.mistakes.map((m) => m.count));
    const mistakes = data.mistakes.some((m) => m.count) ? data.mistakes.filter((m) => m.count).map((m) => h('div', {class:'meter'},
      h('span', null, m.label), bar(m.count / maxMistake, 'red'), h('span', {class:'val'}, String(m.count))))
      : h('p', {class:'muted'}, 'No misses logged yet. Each failed or slow round asks what cost you the most.');
    mount(ctx, h('div', {class:'page cr-page'},
      pageTitle('Progress', `${data.days_left >= 0 ? `T-minus ${plural(data.days_left, 'day')}` : 'Target date passed'} · no streaks, no phase bars`),
      h('div', {class:'grid-2'},
        h('section', {class:'card', 'data-test':'cr-solved-solo'}, h('h2', null, 'Google pool solved solo'),
          h('div', {class:'cr-big'}, `${data.solved_solo} / ${data.pool_total}`),
          bar(data.solved_solo / Math.max(1, data.pool_total), 'red'),
          h('p', {class:'muted small'}, 'Solved with no hint and without seeing the answer. Medium and Hard, asked in the last 3 months.')),
        h('section', {class:'card'}, h('h2', null, '7-day mock score trend'), trend,
          h('p', {class:'muted small'}, scored.length ? 'Average 45-minute mock score per day, out of 100.' : 'Your first mock adds the first bar.'))),
      h('section', {class:'card'}, h('h2', null, 'What costs you points'), mistakes),
      h('p', {class:'muted small'}, `Readiness estimate ${Math.round(data.readiness)}%${data.readiness_provisional ? ' (early, few mocks)' : ''}. A summary of self-graded practice, not a prediction of an offer.`)));
    return {};
  }

  return {today, progress, behavioralDrill};
})();
