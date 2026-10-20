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
