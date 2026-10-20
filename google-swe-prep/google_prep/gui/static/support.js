/* Practical support for starting, returning and continuing. All evidence is saved locally. */
"use strict";
const Support = (() => {
  const requestId = () => globalThis.crypto && crypto.randomUUID ? crypto.randomUUID() : Date.now() + '-' + Math.random().toString(36).slice(2);
  function actions(onSaved, onError) {
    const value = {busy:false, async run(method, payload) {
      if (value.busy) return false;
      value.busy = true;
      try {const data = await api(method, payload); onSaved(data); return true;}
      catch (error) {onError(error); return false;}
      finally {value.busy = false;}
    }};
    return value;
  }
  function seconds(block, received, now) {
    return Math.min(7200, block.elapsed_seconds + (block.status === 'active' ? Math.max(0, now - received) : 0));
  }
  async function parkSaved(save, leave) {
    if (!(await save())) return false;
    await leave();
    return true;
  }
  function panelTitle(title, detail) {
    return h('div', {class:'support-heading'}, h('h2', null, title), h('p', {class:'muted small'}, detail));
  }
  function formField(label, input) {
    const id = 'support-' + requestId(); input.id = id;
    return h('div', {class:'field'}, h('label', {for:id}, label), input);
  }
  const skillActions = {
    problem_solving:'Read one example. Explain its input, output and one edge case.',
    system_design:'Choose one small system. State its requirements and one design trade-off.',
    ai_design:'Choose one AI feature. Explain its inputs, evaluation and one failure case.',
    communication:'Explain one project decision in plain words, with evidence from your work.'
  };
  function startCard(ctx, initial, full = false, onChanged = ()=>{}) {
    let data = initial, received = Date.now()/1000, draftNote = initial.active ? initial.active.note : '', timer, disposed = false;
    const root = h('section', {class:'support-start', 'data-test':'support-start'});
    const error = h('p', {class:'support-error', role:'alert', hidden:true});
    const act = actions(next => {
      if (disposed || !ctx.alive()) return;
      data = next; received = Date.now()/1000; error.hidden = true; render(); onChanged(next);
    }, err => {if (!disposed && ctx.alive()) {error.textContent = err.message + ' Your text is still here. Retry or refresh.';error.hidden=false;}});
    const mutate = (action, note='') => act.run('support_transition', {block_id:data.active.id, action, note});
    function render() {
      clearInterval(timer);
      document.documentElement.dataset.readingSize = data.preferences.reading_size;
      const active = data.active;
      const heading = active ? 'One block. One next step.' : data.today.returning ? 'A fresh start after a gap.' : 'Make starting easier.';
      root.replaceChildren(panelTitle(heading, active ? 'You choose when to pause, finish or switch.' :
        data.today.minimum_done ? 'Your minimum is recorded. More practice is optional; your full study goal stays separate.' :
        'Confidence grows from actions you can point to. Begin with a small, specific one.'));
      if (active) {
        const display = h('span', {class:'support-clock', 'data-test':'support-clock'});
        const limit = h('div', {class:'support-boundary', role:'status', hidden:true});
        const note = h('textarea', {rows:'2',maxlength:'240',value:draftNote,
          placeholder:'Optional: what can you explain now? If parking, write the next step.',
          oninput:e=>{draftNote=e.target.value;},'data-test':'support-note'});
        const transition = async action => {
          const ok = await mutate(action, note.value);
          if(ok && (action === 'complete' || action === 'park')) draftNote='';
        };
        const controls = h('div', {class:'row support-actions'},
          btn(active.status === 'paused' ? 'Resume block' : 'Pause block', ()=>transition(active.status === 'paused' ? 'resume' : 'pause'), {cls:'ghost',test:'support-pause'}),
          btn('Finish block', ()=>transition('complete'), {cls:'primary',test:'support-finish'}),
          btn('Save & park', ()=>transition('park'), {cls:'ghost',test:'support-park'}));
        function update() {
          if (!ctx.alive() || disposed) return;
          const elapsed = seconds(active,received,Date.now()/1000);
          display.textContent = mmss(elapsed) + ' / ' + active.target_minutes + ':00';
          limit.hidden = elapsed < active.target_minutes*60;
          if (!limit.hidden && !limit.childNodes.length) limit.append(
            h('b',null,'Time to choose your next step.'),
            h('p',null,'Finish, save & park, or deliberately extend this block.'),
            btn('Extend 10 minutes', ()=>transition('extend'), {cls:'ghost small',test:'support-extend'}));
        }
        root.append(h('div',{class:'support-focus-line'},h('h3',null,active.task),display),
          h('p',{class:'muted small'},active.day !== data.date ? 'This saved block is from an earlier day. Park or finish it before starting today.' :
            active.status === 'paused' ? 'Paused. This time does not count toward your block.' : 'Focus time is self-acknowledged when you finish. A timer is not a coding grade.'),
          limit,formField('One short learning or next-step note · optional',note),controls,
          h('div',{class:'row'},btn('Open practice',()=>go(data.preferences.core_focus === 'problem_solving' ? '#/mock' : '#/role'),{cls:'ghost small'}),
            btn('I feel like giving up',()=>rescue(ctx,{host:root,onSaved:onChanged}),{cls:'ghost small',test:'support-rescue'})));
        update(); timer=setInterval(update,1000);
      } else {
        const task = h('textarea',{rows:'2',maxlength:'240',value:data.today.returning
          ? 'Revisit one familiar example and explain the input and output.'
          : skillActions[data.preferences.core_focus],
          'data-test':'support-task'});
        const begin = minutes => act.run('support_start',{date:data.date,minutes,task:task.value});
        root.append(formField('Your next small action',task),
          h('div',{class:'row support-actions'},
            btn('Start '+data.preferences.minimum_minutes+' minutes',()=>begin(data.preferences.minimum_minutes),{cls:'primary',test:'support-start-small'}),
            btn('Start '+data.preferences.focus_minutes+'-minute block',()=>begin(data.preferences.focus_minutes),{cls:'ghost',test:'support-start-focus'}),
            btn('I feel like giving up',()=>rescue(ctx,{host:root,onSaved:onChanged}),{cls:'ghost',test:'support-rescue'})),
          h('p',{class:'support-commitment'},h('b',null,'Your starting commitment: '),
            data.preferences.cue || 'Choose a cue in your support settings.'));
      }
      root.append(error);
      if (!full) root.append(btn('Confidence & consistency',()=>go('#/support'),{cls:'ghost small',test:'open-support'}));
    }
    render();
    return {node:root, update(next){data=next;received=Date.now()/1000;render();}, cleanup(){disposed=true;clearInterval(timer);}};
  }
  function attachToday(ctx, target) {
    let card = null, closed = false;
    const unavailable = h('section',{class:'support-unavailable'},h('p',null,'Study support could not load. Your practice is still available.'),
      btn('Retry support',()=>load(),{cls:'ghost small'}));
    async function load() {
      try {
        const data=await api('support_state',{}, {timeoutMs:4000});
        if(closed || !ctx.alive()) return;
        card && card.cleanup();card=startCard(ctx,data);target.replaceChildren(card.node);
      } catch(error) {if(!closed && ctx.alive())target.replaceChildren(unavailable);}
    }
    load();
    return {cleanup(){closed=true;card && card.cleanup();}};
  }
  async function rescue(ctx, options={}) {
    const host=options.host || $('#view');
    const prior=host.querySelector('.support-rescue-panel'); if(prior){prior.remove();return;}
    const panel=h('section',{class:'support-rescue-panel','data-test':'support-rescue-panel'});
    host.append(panel);
    const loading=h('p',null,'Loading a small next step…');panel.append(loading);
    let interval, selected=null, rescueId=null, pending=false;
    const alive=()=>ctx.alive() && panel.isConnected;
    try {
      const state=await api('support_state',{}, {timeoutMs:4000});
      if(!alive())return {cleanup(){}};
      const advice=h('div',{class:'support-advice',role:'status'});
      const error=h('p',{class:'support-error',role:'alert'});
      const helpful=btn('This helped me continue',async()=>{
        if(!selected || pending)return;pending=true;
        try {const next=await api('support_rescue',{date:state.date,blocker:selected,outcome:'helpful',request_id:rescueId});
          if(alive()) {helpful.disabled=true;helpful.textContent='Recovery recorded · self-report';if(options.onSaved)options.onSaved(next);}}
        catch(err){if(alive())error.textContent=err.message;}finally{pending=false;}
      },{cls:'ghost small',test:'rescue-helpful'});helpful.hidden=true;
      const smallTimer=h('span',{class:'small muted','aria-live':'polite'});
      const retry=btn('Try two minutes',()=>{
        clearInterval(interval);let until=Date.now()+120000;smallTimer.textContent='02:00 · one small step';
        interval=setInterval(()=>{if(!alive()){clearInterval(interval);return;}const left=Math.max(0,Math.ceil((until-Date.now())/1000));smallTimer.textContent=left?mmss(left)+' · one small step':'Now choose: continue, save & park, or stop.';if(!left)clearInterval(interval);},1000);
      },{cls:'primary small',test:'rescue-two-minutes'});retry.hidden=true;
      const choices=h('div',{class:'support-blockers'},state.blockers.map(item=>btn(item.label,async()=>{
        if(pending)return;pending=true;error.textContent='';
        const nextId=requestId();
        try{await api('support_rescue',{date:state.date,blocker:item.id,outcome:'opened',request_id:nextId});
          if(!alive())return;selected=item.id;rescueId=nextId;
          advice.replaceChildren(h('h3',null,'Your next useful step'),h('p',null,item.step));
          helpful.hidden=false;helpful.disabled=false;helpful.textContent='This helped me continue';retry.hidden=false;
        }catch(err){if(alive())error.textContent=err.message+' You can still try one small example.';}
        finally{pending=false;}
      },{cls:'ghost small',test:'blocker-'+item.id})));
      const close=()=>{clearInterval(interval);panel.remove();};
      panel.replaceChildren(panelTitle('Before you give up, make the task smaller.','Choose what is blocking you. Two useful minutes are a valid next step.'),
        choices,advice,h('div',{class:'row'},retry,smallTimer,helpful),
        options.park ? btn('Save work & return to Today',async()=>{
          if(pending)return;pending=true;error.textContent='';
          try{if(await options.park()){close();}}
          catch(err){if(alive())error.textContent=err.message+' Keep this window open and retry.';}
          finally{pending=false;}
        },{cls:'ghost',test:'rescue-save-park'}) : null,
        h('p',{class:'muted small'},options.mock ? 'Your interview clock and hint rules stay in effect. This rescue gives no solution.' :
          'You may stop deliberately. Parking a task is a choice, and does not count as solving it.'),
        error,btn('Close rescue',close,{cls:'ghost small'}));
    }catch(err){if(alive())panel.replaceChildren(h('p',{class:'support-error'},'Rescue could not load. Try one tiny example, or save your work and take a break.'),btn('Close',()=>panel.remove(),{cls:'ghost small'}));}
    return {cleanup(){clearInterval(interval);panel.remove();}};
  }
  async function page(ctx) {
    const data=await api('support_state');
    if(!ctx.alive())return {};
    setCrumbs(['Interview Prep','Confidence & consistency'],ctx);
    const block=startCard(ctx,data,true,updateEvidence);
    const prefs=data.preferences;
    const role=h('input',{type:'text',maxlength:'240',value:prefs.general_role,'data-test':'support-general-role'});
    const cue=h('input',{type:'text',maxlength:'240',value:prefs.cue,'data-test':'support-cue'});
    const minimum=h('select',{'data-test':'support-minimum'},[2,5,10].map(n=>h('option',{value:n,selected:n===prefs.minimum_minutes},n+' minutes')));
    const focus=h('select',{'data-test':'support-duration'},[15,25,45].map(n=>h('option',{value:n,selected:n===prefs.focus_minutes},n+' minutes')));
    const reading=h('select',null,['standard','comfortable'].map(value=>h('option',{value,selected:value===prefs.reading_size},value==='standard'?'Standard':'Comfortable')));
    const prompts=h('input',{type:'checkbox',checked:prefs.rescue_prompts});
    const skill=h('select',{'data-test':'support-core-focus'},[
      ['problem_solving','Problem solving & coding'],['system_design','LLD / HLD & trade-offs'],
      ['ai_design','ML / GenAI design'],['communication','Clear explanations & interview communication']
    ].map(([value,label])=>h('option',{value,selected:value===prefs.core_focus},label)));
    const saved=h('p',{role:'status',class:'small'});
    const form=h('form',{class:'support-settings',onsubmit:e=>{e.preventDefault();savePreferences();}},
      panelTitle('Discipline: give yourself a clear starting cue.','Attach a small action to something that already happens in your day.'),
      formField('When will you start?',cue),
      formField('General role target · company optional',role),
      formField('Reusable skill to practise next',skill),
      h('p',{class:'muted small'},'This target belongs to General mode. Google mode keeps its own target date and preparation plan. Your recorded skills stay with you.'),
      h('div',{class:'support-settings-grid'},formField('Small daily minimum',minimum),formField('Normal focus block',focus),formField('Reading size',reading)),
      h('label',{class:'support-checkbox'},prompts,h('span',null,'Show the rescue prompt during practice')),
      btn('Save my routine',savePreferences,{cls:'primary',test:'support-save-preferences'}),saved);
    let saving=false;
    async function savePreferences(){
      if(saving)return;saving=true;saved.textContent='Saving…';
        try{const next=await api('support_preferences',{changes:{cue:cue.value,minimum_minutes:Number(minimum.value),focus_minutes:Number(focus.value),reading_size:reading.value,rescue_prompts:prompts.checked,core_focus:skill.value,general_role:role.value}});
        if(ctx.alive()){saved.textContent='Routine saved. Your full problem goal stays separate.';document.documentElement.dataset.readingSize=reading.value;block.update(next);updateEvidence(next);}}
      catch(err){if(ctx.alive())saved.textContent='Not saved: '+err.message+' Your settings are still here.';}
      finally{saving=false;}
    }
    function confidenceView(data) { return h('section',{class:'support-evidence'},
      panelTitle('Confidence: look at what you have actually done.','Last seven days. Focus completion and recovery are self-reported; solve grades are separate.'),
      h('dl',{class:'support-evidence-list'},
        h('div',null,h('dt',null,'Completed focus blocks'),h('dd',null,String(data.week.completed_blocks))),
        h('div',null,h('dt',null,'Problems solved without help'),h('dd',null,String(data.confidence.independent_solves))),
        h('div',null,h('dt',null,'Rescue steps you said helped'),h('dd',null,String(data.week.helpful_rescues)))),
      data.confidence.notes.length ? h('div',null,h('h3',null,'Things you said you can explain'),data.confidence.notes.map(n=>
        h('p',{class:'support-learning-note'},n.note,h('span',{class:'muted small'},' · '+n.day+' · self-report')))) :
        h('p',{class:'muted small'},'No learning notes saved yet. After a block, optionally name one thing you can explain. You can begin before feeling confident.')); }
    function consistencyView(data) { return h('section',{class:'support-evidence'},
      panelTitle('Consistency: practise returning, even after a gap.',data.week.active_days+' active days in the last seven. Missing records do not prove you did no study.'),
      h('div',{class:'support-week'},data.week.days.map(day=>h('div',{class:day.active?'active':''},
        h('span',null,new Date(day.date+'T12:00:00').toLocaleDateString(undefined,{weekday:'short'})),
        h('b',null,day.active?'Practised':'Not recorded')))),
      h('p',{class:'small'},data.week.return_days+' return days recorded after a gap. Missed work is not added to today’s quota.'),
      h('p',{class:'muted small'},'Completed focus blocks and outside-practice logs are acknowledgements, not automatic proof of focused work.'),
      btn('Plan a small start today',()=>go('#/today'),{cls:'ghost'})); }
    const confidence=h('div'), consistency=h('div');
    function updateEvidence(next) {
      confidence.replaceChildren(confidenceView(next));consistency.replaceChildren(consistencyView(next));
    }
    updateEvidence(data);
    const giveup=h('section',{class:'support-evidence'},
      panelTitle('When you want to give up: change the next step.','Name the obstacle, try one smaller action, then choose whether to continue or park.'),
      btn('Find a rescue step',()=>rescue(ctx,{host:giveup,onSaved:updateEvidence}),{cls:'primary',test:'support-open-rescue'}));
    const outsideId=requestId();
    const outsideText=h('input',{type:'text',maxlength:'240',placeholder:'For example: traced Two Sum on paper','data-test':'support-outside-task'});
    const outsideMinutes=h('input',{type:'number',min:'1',max:'120',value:'5','data-test':'support-outside-minutes'});
    const outsideStatus=h('p',{class:'small',role:'status'});let outsideBusy=false;
    const elsewhere=h('details',{class:'support-evidence'},h('summary',null,'Practised somewhere else? Record it.'),
      h('p',{class:'muted small'},'This can count toward your small minimum. It does not create a graded solve.'),
      formField('What did you practise?',outsideText),formField('Minutes · your estimate',outsideMinutes),
      btn('Record outside practice',async()=>{
        if(outsideBusy)return;outsideBusy=true;
        try{const next=await api('support_outside',{date:data.date,minutes:Number(outsideMinutes.value),task:outsideText.value,request_id:outsideId});
          if(ctx.alive()){outsideStatus.textContent='Saved as outside practice. Your evidence is updated.';block.update(next);updateEvidence(next);$('[data-test="support-outside-save"]',elsewhere).disabled=true;}}
        catch(err){if(ctx.alive())outsideStatus.textContent='Not saved: '+err.message;}finally{outsideBusy=false;}
      },{cls:'ghost',test:'support-outside-save'}),outsideStatus);
    const parked=data.parked.length ? h('section',{class:'support-evidence'},h('h2',null,'Parked next steps'),
      data.parked.map(item=>h('div',{class:'support-parked'},h('b',null,item.task),h('p',{class:'small muted'},item.note || 'No next-step note saved.'),
        btn('Use this next step',()=>{go('#/today');toast('Copy this next step into your small start: '+(item.note || item.task));},{cls:'ghost small'})))) : null;
    mount(ctx,h('div',{class:'page support-page'},pageTitle('Build skills you can rely on.',(prefs.general_role || 'Your next role')+' · General mode. Start small, return after gaps, and build evidence through practice.'),block.node,confidence,form,consistency,giveup,parked,elsewhere));
    return {cleanup(){block.cleanup();}};
  }
  async function today(ctx) {
    const data=await api('support_state');
    if(!ctx.alive())return {};
    setCrumbs(['Interview Prep','Today'],ctx);
    const summary=h('section',{class:'support-evidence'});
    function updateSummary(next) {
      summary.replaceChildren(panelTitle('A little evidence you can build on.',
        next.week.active_days+' active days · '+next.week.completed_blocks+' completed focus blocks · '+next.confidence.independent_solves+' independently solved problems in seven days.'),
        h('p',{class:'small'},next.today.minimum_done ? 'Your small minimum is recorded. Continue toward your chosen skill when you have time.' :
          'Your small minimum is '+next.preferences.minimum_minutes+' minutes. Completing it does not mark a problem solved.'),
        h('p',{class:'muted small'},next.today.returning ? 'Returning after a gap? One familiar example is enough to start. Yesterday’s backlog is not added to today.' :
          'Start before you feel ready. If you get stuck, change the next step instead of judging the whole day.'),
        btn('See my confidence & consistency',()=>go('#/support'),{cls:'ghost small'}));
    }
    const block=startCard(ctx,data,false,updateSummary);updateSummary(data);
    const practice=h('section',{class:'support-evidence'},panelTitle('Build a skill that travels with you.','Problem solving, coding, design and clear explanations apply across interviews.'),
      h('div',{class:'row support-actions'},
        btn('Coding practice',()=>go('#/mock'),{cls:'ghost'}),
        btn('Design & communication',()=>go('#/role'),{cls:'ghost'}),
        btn('Python foundations',()=>go('#/fasttrack'),{cls:'ghost'})),
      h('p',{class:'muted small'},'Coding uses the existing shared problem bank, originally curated for Google. Your General routine and role target are separate.'));
    mount(ctx,h('div',{class:'page support-page'},pageTitle('One useful step today.',
      (data.preferences.general_role || 'Your next role')+' · General mode'),block.node,summary,practice));
    return {cleanup(){block.cleanup();}};
  }
  ROUTES.support=async ctx=>{
    if(S.state && S.state.settings.preparation_mode !== 'general') {
      mount(ctx,h('div',{class:'page'},pageTitle('Study support is in General mode.','Your Google preparation flow stays available here.'),
        btn('Choose preparation mode',openPreparationMode,{cls:'primary'})));
      return {};
    }
    return page(ctx);
  };
  return {actions,seconds,parkSaved,attachToday,today,page,rescue};
})();
