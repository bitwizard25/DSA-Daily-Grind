const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../google_prep/gui/static/support.js'),'utf8');
function env(api){const ctx={api,ROUTES:{},Date,Math,setInterval,clearInterval,setTimeout,clearTimeout,console};vm.createContext(ctx);vm.runInContext(source,ctx);return vm.runInContext('Support',ctx)}
test('failed support save retains the form and never claims success',async()=>{
  let saved=0,error='';
  const support=env(async()=>{throw Error('Offline')});
  const actions=support.actions(()=>saved++,e=>error=e.message);
  assert.equal(await actions.run('support_transition',{block_id:'one',action:'complete',note:'My learning'}),false);
  assert.equal(saved,0);assert.equal(error,'Offline');assert.equal(actions.busy,false);
});
test('parallel clicks do not create two blocks and update only after persistence',async()=>{
  let release,calls=0,saved=0;
  const support=env(()=>{calls++;return new Promise(resolve=>release=resolve)});
  const actions=support.actions(()=>saved++,()=>{});
  const first=actions.run('support_start',{date:'2026-10-20',minutes:5,task:'One example'});
  assert.equal(await actions.run('support_start',{}),false);
  assert.equal(saved,0);assert.equal(calls,1);
  release({active:{id:'one'}});assert.equal(await first,true);assert.equal(saved,1);
});
test('focus clock includes elapsed running time but excludes paused and backwards time',()=>{
  const support=env(async()=>({}));
  assert.equal(support.seconds({elapsed_seconds:120,status:'paused'},100,400),120);
  assert.equal(support.seconds({elapsed_seconds:120,status:'active'},100,400),420);
  assert.equal(support.seconds({elapsed_seconds:120,status:'active'},400,100),120);
});

test('parking never navigates away when code could not be saved',async()=>{
  const support=env(async()=>({}));let left=false;
  assert.equal(await support.parkSaved(async()=>false,async()=>{left=true}),false);
  assert.equal(left,false);
  assert.equal(await support.parkSaved(async()=>true,async()=>{left=true}),true);
  assert.equal(left,true);
});

test('support drafts persist before leaving and remain recoverable after a failed save',async()=>{
  const calls=[];let online=false;
  const support=env(async(method,payload)=>{if(!online)throw Error('Offline');calls.push([method,payload]);return {ok:true}});
  const drafts=support.drafts({});
  drafts.set('task',{text:'Keep this next step'});
  assert.equal(await drafts.flush(),false);
  assert.equal(drafts.get('task').text,'Keep this next step');
  online=true;
  assert.equal(await drafts.flush(),true);
  assert.equal(calls.length,1);assert.equal(calls[0][0],'support_draft');
  assert.equal(calls[0][1].value.text,'Keep this next step');
  assert.equal(await drafts.clear('task'),true);
  assert.equal(drafts.get('task'),undefined);
  assert.equal(calls[1][1].expected.text,'Keep this next step');
  const count=calls.length;
  assert.equal(await drafts.clear('routine'),true);
  assert.equal(calls.length,count);
});

test('General Today keeps practice links usable when support is unavailable',async()=>{
  function tree(tag,attrs,...children){return {tag,attrs:attrs||{},children:children.flat(Infinity).filter(x=>x!=null),
    get textContent(){return this.children.map(x=>typeof x==='object'?x.textContent:String(x)).join(' ');}}}
  let root;
  const ctx={api:async()=>{throw Error('Support offline')},ROUTES:{},Date,Math,setInterval,clearInterval,setTimeout,clearTimeout,
    h:tree,btn:(label)=>tree('button',null,label),pageTitle:(...text)=>tree('header',null,...text),setCrumbs:()=>{},
    mount:(_,node)=>{root=node},S:{state:{session:null}}};
  vm.createContext(ctx);vm.runInContext(source,ctx);
  const handle=await vm.runInContext('Support.today({alive:()=>true})',ctx);
  assert.match(root.textContent,/Coding practice/);
  assert.match(root.textContent,/Retry/);
  handle.cleanup?.();
});
