const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
function load(api,extra={}){
  const ctx={api,ROUTES:{},console,...extra}; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../google_prep/gui/static/engineering.js'),'utf8'),ctx);
  return vm.runInContext('Engineering',ctx);
}
test('failed saves retain dirty work and do not advance revision',async()=>{
  let message='';const work={code:'unsaved code',notes:{read:'my explanation'},help:'none'};
  const editor=load(async()=>{throw Error('offline')}).persistence({id:'one',revision:0,code:'',notes:{},help:'none',status:'draft'},()=>work,x=>message=x);
  assert.equal(await editor.save(),false);assert.equal(editor.dirty(),true);
  assert.equal(editor.attempt.revision,0);assert.match(message,/offline/);
  assert.equal(work.code,'unsaved code');
});

function tree(tag,attrs,...children){
  const kids=children.flat(Infinity).filter(x=>x!=null);
  return {tag,attrs:attrs||{},value:attrs?.value||(tag==='select'?(kids.find(x=>x.attrs?.selected)||kids[0])?.attrs?.value:'')||'',children:kids,
    addEventListener(){},replaceChildren(...kids){this.children=kids.flat(Infinity);},
    get options(){return this.children;},
    get textContent(){return this.children.map(x=>typeof x==='object'?x.textContent:String(x)).join(' ');},
    set textContent(value){this.children=[value];}};
}
function find(root,id){if(root?.attrs?.id===id)return root;for(const child of root?.children||[]){const found=find(child,id);if(found)return found;}}
test('exercise navigation preserves unsaved work after a failed save',async()=>{
  let root,warning='';
  const initial={id:'one',revision:0,status:'draft',code:'starter',help:'none',notes:{}};
  const editor=load(async method=>{
    if(method==='engineering_open')return {exercise:{id:'events',title:'Events',minutes:20,brief:'Keep IDs once',example:'a,a -> a',test_checklist:[],extension:'Bound memory'},attempt:initial};
    throw Error('offline');
  },{h:tree,btn:(label,onclick,opts={})=>tree('button',{onclick,...opts},label),
    S:{state:{settings:{coding_language:'python'}}},pageTitle:()=>tree('header'),setCrumbs(){},
    mount:(ctx,node)=>{root=node;},setInterval:()=>1,clearInterval(){},
    openModal:async(title)=>{warning=title;return false;}});
  const handle=await editor.page({parts:['engineering','events'],query:new URLSearchParams('lang=python'),alive:()=>true});
  assert.equal(handle.isDirty(),false);
  find(root,'engineering-read').value='Keep my explanation';
  assert.equal(await handle.beforeLeave(),false);
  assert.equal(find(root,'engineering-read').value,'Keep my explanation');
  assert.match(warning,/not saved/);assert.equal(handle.isDirty(),true);
  handle.cleanup();
});
test('edits made during a save stay dirty until the newer version is saved',async()=>{
  let finish;let work={code:'first',notes:{},help:'none'};
  const editor=load((name,payload)=>new Promise(resolve=>{finish=()=>resolve({...payload,id:'one',revision:payload.revision+1,status:'draft'});})).persistence(
    {id:'one',revision:0,code:'',notes:{},help:'none',status:'draft'},()=>work,()=>{});
  const pending=editor.save();work={...work,code:'second'};finish();
  assert.equal(await pending,true);assert.equal(editor.dirty(),true);assert.equal(editor.attempt.revision,1);
  const next=editor.save();finish();assert.equal(await next,true);assert.equal(editor.dirty(),false);
});
test('two finish clicks produce one completed attempt',async()=>{
  let finish,calls=0;const work={code:'fixed',notes:{read:'flow'},help:'none'};
  const editor=load((name,payload)=>{calls++;return new Promise(resolve=>{finish=()=>resolve({...payload,id:'one',revision:1,status:'completed'});});}).persistence(
    {id:'one',revision:0,code:'',notes:{},help:'none',status:'draft'},()=>work,()=>{});
  const first=editor.save(true),second=editor.save(true);finish();
  assert.equal(await first,true);assert.equal(await second,true);assert.equal(calls,1);
});

test('resumed assisted draft cannot display a lower help level',async()=>{
  let root;
  const editor=load(async()=>({exercise:{id:'events',title:'Events',minutes:20,brief:'Read',example:'Trace',test_checklist:[],extension:''},
    attempt:{id:'one',revision:3,status:'draft',code:'starter',help:'answer',notes:{}}}),
    {h:tree,btn:(label,onclick,opts={})=>tree('button',{onclick,...opts},label),S:{state:{}},
      pageTitle:()=>tree('header'),setCrumbs(){},mount:(ctx,node)=>{root=node;},setInterval:()=>1,clearInterval(){}});
  const handle=await editor.page({parts:['engineering','events'],query:new URLSearchParams('lang=python'),alive:()=>true});
  const help=find(root,'engineering-help');
  assert.equal(help.value,'answer');
  assert.equal(help.options.find(x=>x.attrs.value==='none').attrs.disabled,true);
  assert.equal(help.options.find(x=>x.attrs.value==='ai').attrs.disabled,true);
  handle.cleanup();
});
