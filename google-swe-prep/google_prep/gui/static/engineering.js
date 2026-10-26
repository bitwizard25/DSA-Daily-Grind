/* Engineering learning work. No learner code is executed in this module. */
'use strict';
const Engineering = (() => {
  const helpLabels = {none:'No help', hint:'A hint', ai:'AI assistance', answer:'Reviewed a solution'};
  const helpOrder = Object.keys(helpLabels);
  const fields = [['read','1. Understand','Where does data enter, what changes, and what comes out?'],
    ['debug','2. Find the bug','Which input breaks the contract, and why?'],
    ['change','3. Explain your change','What did you change, and why does it fix the cause?'],
    ['verify','4. Verify','Record expected and actual results from a hand trace or your own local editor.'],
    ['next','Next step · optional','If you pause, leave one precise step to make returning easier.']];
  function snapshot(work) {return JSON.stringify({code:work.code,notes:work.notes,help:work.help});}
  function persistence(initial, read, status) {
    let attempt=initial, saved=snapshot(initial), pending=null;
    const ranks={none:0,hint:1,ai:2,answer:3};
    const workNow=()=>{const work=read();return {...work,help:ranks[work.help]<ranks[attempt.help]?attempt.help:work.help};};
    const editor={get attempt(){return attempt;},dirty:()=>attempt.status==='draft' && snapshot(workNow())!==saved,
      accept(row){attempt=row;saved=snapshot({code:row.code,notes:row.notes,help:row.help});},
      async save(complete=false){
        if(pending){const ok=await pending;return ok ? editor.save(complete) : false;}
        if(attempt.status==='completed' || (!editor.dirty()&&!complete)) return true;
        const work=JSON.parse(snapshot(workNow()));
        status('Saving…');
        pending=(async()=>{
          try {
            const row=await api('engineering_save',{attempt_id:attempt.id,revision:attempt.revision,...work,complete});
            attempt=row;
            // Server may upgrade assistance; preserve edits made while the request ran.
            saved=snapshot({...work,help:row.help});
            status(complete?'Practice recorded · self-reported':'Saved on this device',row);
            return true;
          }catch(err){status(`Not saved: ${err.message}`);return false;}
        })();
        const result=await pending;pending=null;return result;
      }};
    return editor;
  }
  function retry(item){
    return btn(`Retry ${item.title}`,()=>{
      S.boardStart={problem_id:item.id,practice:true,fresh:true};return go('#/board');
    },{cls:'ghost small'});
  }
  function preview(ctx){
    const slot=h('section',{class:'card engineering-preview','data-test':'engineering-preview'},
      h('div',{class:'section-heading'},h('h2',null,'Turn help into independent skill'),
        btn('Engineering practice',()=>go('#/engineering'),{cls:'ghost small'})),
      h('p',{class:'muted small'},'Loading your next review…'));
    api('engineering_state',{}, {timeoutMs:4000}).then(data=>{
      if(!ctx.alive())return;
      slot.replaceChildren(h('div',{class:'section-heading'},h('h2',null,'Turn help into independent skill'),
        btn('Engineering practice',()=>go('#/engineering'),{cls:'ghost small'})),
        data.retries.length?h('div',null,h('p',null,`${data.retries[0].title} · ${data.retries[0].reason}. Try a fresh attempt without notes.`),retry(data.retries[0])):
          h('p',{class:'muted'},data.upcoming_retries?'A fresh retry will be available tomorrow. Continue today’s main task.':'Optional: read a small program, find its bug, and explain your fix.'),
        data.recovered?h('p',{class:'ok-t small'},`${data.recovered} problems later solved without help after an earlier difficulty · based on your saved grades`):null);
    }).catch(()=>{if(ctx.alive())slot.replaceChildren(h('h2',null,'Engineering practice'),
      h('p',{class:'muted'},'Review progress could not load. Your main session is still available.'),
      btn('Open practice',()=>go('#/engineering'),{cls:'ghost small'}));});
    return slot;
  }
  async function page(ctx){
    const language=ctx.query.get('lang') || S.state?.settings?.coding_language || 'python';
    const exerciseId=ctx.parts[1];
    setCrumbs(['Google SWE II OS','Interview prep','Engineering practice'],ctx);
    if(!exerciseId){
      const data=await api('engineering_state');
      const open=(id,lang=language)=>go(`#/engineering/${id}?lang=${lang}`);
      mount(ctx,h('div',{class:'page engineering-page'},
        pageTitle('Read. Debug. Improve.','Small engineering exercises. Build understanding one change at a time.',
          btn('Interview prep',()=>go('#/role'),{cls:'ghost small'})),
        h('p',{class:'muted small'},'Original practice material, not an official Google interview format. Completion is self-reported and separate from DSA readiness.'),
        data.drafts.length?h('section',{class:'card'},h('h2',null,'Pick up your saved work'),
          h('div',{class:'row'},data.drafts.map(x=>btn(`${data.exercises.find(e=>e.id===x.exercise_id)?.title || x.exercise_id} · ${x.language==='cpp'?'C++':'Python'}`,
            ()=>open(x.exercise_id,x.language),{cls:'ghost'})))):null,
        h('div',{class:'engineering-catalog'},data.exercises.map((item,i)=>h('section',{class:'card'},
          h('span',{class:'kicker'},`${item.level} · ${item.minutes} min`),h('h2',null,item.title),
          h('p',{class:'muted'},'Understand → find a bug → change → verify'),
          h('div',{class:'row'},btn('Python',()=>open(item.id,'python'),{cls:i===0?'primary':'ghost'}),
            btn('C++',()=>open(item.id,'cpp'),{cls:'ghost'}))))),
        h('section',{class:'card'},h('h2',null,'Try it independently'),
          h('p',{class:'muted'},'Failed or assisted DSA attempts return the next day. A later solve recorded without help clears the item.'),
          data.retries.length?data.retries.slice(0,8).map(item=>h('div',{class:'engineering-retry'},h('div',null,h('b',null,item.title),
            h('p',{class:'muted small'},`${item.reason} · available since ${item.due}`)),retry(item))):h('p',null,'No independent retries due.'),
          data.upcoming_retries?h('p',{class:'muted small'},`${data.upcoming_retries} more available tomorrow.`):null,
          h('p',{class:'ok-t'},`${data.recovered} problems later solved without help after an earlier difficulty.`)),
        h('section',{class:'card'},h('h2',null,'Recent practice'),
          data.recent.length?data.recent.map(x=>h('p',null,`${data.exercises.find(e=>e.id===x.exercise_id)?.title || x.exercise_id} · ${x.language==='cpp'?'C++':'Python'} · ${helpLabels[x.help]} · self-reported`)):
            h('p',{class:'muted'},'Your completed exercises will appear here.'))));
      return {};
    }
    const data=await api('engineering_open',{exercise_id:exerciseId,language});
    if(!ctx.alive())return {};
    const item=data.exercise, row=data.attempt, done=row.status==='completed';
    const feedback=h('p',{class:'save-feedback',role:'status','aria-live':'polite'},done?'Completed · self-reported':'Saved draft · autosaves every 5 seconds');
    const help=h('select',{id:'engineering-help',disabled:done},Object.entries(helpLabels).map(([value,label])=>
      h('option',{value,selected:value===row.help,disabled:helpOrder.indexOf(value)<helpOrder.indexOf(row.help)},label)));
    const inputs=Object.fromEntries(fields.map(([key])=>[key,h('textarea',{id:`engineering-${key}`,rows:key==='next'?'1':'2',maxlength:'3000',value:row.notes[key]||'',disabled:done})]));
    const code=h('textarea',{class:'engineering-code',id:'engineering-code','aria-label':`${language==='cpp'?'C++':'Python'} working code`,
      rows:'22',maxlength:'32000',spellcheck:'false',value:row.code,disabled:done});
    const reference=h('div',{class:'engineering-reference'});
    const read=()=>({code:code.value,notes:Object.fromEntries(fields.map(([key])=>[key,inputs[key].value])),help:help.value});
    const initial={...row,notes:Object.fromEntries(fields.map(([key])=>[key,row.notes[key]||'']))};
    const editor=persistence(initial,read,(text,savedRow)=>{
      feedback.textContent=text;
      if(savedRow){
        if(helpOrder.indexOf(help.value)<helpOrder.indexOf(savedRow.help))help.value=savedRow.help;
        Array.from(help.options).forEach(option=>{option.disabled=helpOrder.indexOf(option.value)<helpOrder.indexOf(savedRow.help);});
      }
    });
    const controls=[code,help,...Object.values(inputs)];
    function lock(value){controls.forEach(input=>{input.disabled=value;});}
    let actionBusy=false;
    const download=()=>downloadText(`${item.id}-${language}-practice.txt`,JSON.stringify(read(),null,2));
    async function reveal(){
      if(actionBusy)return;
      actionBusy=true;lock(true);
      try{
        if(!(await editor.save()))return;
        const result=await api('engineering_reference',{attempt_id:row.id,revision:editor.attempt.revision});
        editor.accept({...result.attempt,notes:Object.fromEntries(fields.map(([key])=>[key,result.attempt.notes[key]||'']))});help.value='answer';
        Array.from(help.options).forEach(option=>{option.disabled=option.value!=='answer';});
        reference.replaceChildren(h('h3',null,'Reference · solution help recorded'),h('p',null,result.reference.explanation),codeBlock(result.reference.code,language));
        feedback.textContent='Reference opened. This attempt is recorded as reviewed a solution.';
      }catch(err){feedback.textContent=`Could not open reference: ${err.message}`;}
      finally{lock(false);actionBusy=false;}
    }
    async function finish(){
      if(actionBusy)return;
      actionBusy=true;
      lock(true);
      if(await editor.save(true)){if(ctx.alive())go(`#/engineering/${item.id}?lang=${language}`);}
      else lock(false);
      actionBusy=false;
    }
    const reset=h('details',{class:'engineering-reset'},h('summary',null,'Stuck? Take one small step'),
      h('ol',null,h('li',null,'Trace the provided example on paper. What should each variable contain?'),
        h('li',null,'Compare the first wrong value with what the code actually does.'),
        h('li',null,'Write one hypothesis and one test. Work on that for two minutes.')),
      h('p',null,'Need to stop? Write one next step below and save. Returning is part of practice.'));
    mount(ctx,h('div',{class:'page engineering-page'},pageTitle(item.title,`${language==='cpp'?'C++':'Python'} · about ${item.minutes} minutes`,
      btn('All exercises',()=>go('#/engineering'),{cls:'ghost small'})),
      h('div',{class:'row'},btn('Python',()=>go(`#/engineering/${item.id}?lang=python`),{cls:language==='python'?'primary small':'ghost small'}),
        btn('C++',()=>go(`#/engineering/${item.id}?lang=cpp`),{cls:language==='cpp'?'primary small':'ghost small'})),
      h('p',{class:'muted small'},'Learning workspace · code is not executed here. Verify by hand or in your own editor. This is not an official interview simulation.'),
      h('div',{class:'engineering-layout'},h('section',{class:'card engineering-notes'},h('h2',null,'Your task'),h('p',null,item.brief),
        h('p',{class:'q'},item.example),reset,
        fields.map(([key,label,prompt])=>h('div',{class:'field'},h('label',{for:`engineering-${key}`},label),h('p',{class:'muted small'},prompt),inputs[key])),
        h('details',null,h('summary',null,'Test checklist and optional follow-up'),
          h('ul',null,item.test_checklist.map(x=>h('li',null,x))),h('p',null,item.extension))),
        h('section',{class:'card engineering-work'},h('h2',null,'Working code'),code,
          h('div',{class:'field'},h('label',{for:'engineering-help'},'Help used in this attempt'),help),
          h('p',{class:'muted small'},'Help can increase during an attempt. A fresh attempt starts a separate record.'),feedback,
          h('div',{class:'row'},done?btn('Start a fresh attempt',async()=>{
            try{await api('engineering_open',{exercise_id:item.id,language,restart:true});go(`#/engineering/${item.id}?lang=${language}`);}
            catch(err){feedback.textContent=err.message;}
          },{cls:'primary'}):[
            btn('Save draft',()=>editor.save(),{cls:'ghost',test:'engineering-save'}),
            btn('Finish · self-report',finish,{cls:'primary',test:'engineering-finish'})],
            btn('Download work',download,{cls:'ghost small'})),
          !done?btn('Show reference · records solution help',reveal,{cls:'ghost small',test:'engineering-reference'}):null,reference))));
    const onEdit=()=>{feedback.textContent='Unsaved changes';};
    for(const input of [code,help,...Object.values(inputs)])input.addEventListener('input',onEdit);
    const timer=done?null:setInterval(()=>{if(editor.dirty())editor.save();},5000);
    return {isDirty:editor.dirty,cleanup:()=>clearInterval(timer),beforeLeave:async()=>{
      if(await editor.save())return true;
      return Boolean(await openModal('Your latest work is not saved',h('div',null,h('p',null,'Keep editing or download a copy before leaving.'),
        btn('Download work',download,{cls:'ghost'})),[{label:'Keep editing',value:false,primary:true},{label:'Leave anyway',value:true,danger:true}]));
    }};
  }
  return {persistence,preview,page};
})();
ROUTES.engineering=Engineering.page;
