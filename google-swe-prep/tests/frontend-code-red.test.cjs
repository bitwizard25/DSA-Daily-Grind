/* Code Red screens: the checklist starts the next open block, and Code Red pages stay inside Code Red. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../google_prep/gui/static/code_red.js'),'utf8');

function tree(tag,attrs,...children){
  return {tag,attrs:attrs||{},children:children.flat(Infinity).filter(x=>x!=null&&x!==false),
    get textContent(){return this.children.map(x=>typeof x==='object'?x.textContent:String(x)).join('');}};
}
function find(node,pred){
  if(!node||typeof node!=='object')return null;
  if(pred(node))return node;
  for(const c of node.children||[]){const r=find(c,pred);if(r)return r;}
  return null;
}
const DAY={phase:'sprint',days_left:12,readiness:41.2,readiness_provisional:true,queue_due:2,completed:1,total:5,notes:[],
  blocks:[
    {id:'warmup',title:'5-minute mistake warmup',minutes:5,done:true,detail:'cards',action:{type:'recall',mode:'code_red'}},
    {id:'strike',title:'Re-Solve Strike',minutes:45,done:false,kind:'resolve',detail:'due',
     problem:{id:'G-200',title:'Number of Islands',difficulty:'Medium',category:'Graphs'},action:{type:'mock',problem_id:'G-200'}},
    {id:'mock1',title:'Live mock 1',minutes:45,done:false,kind:'mock',detail:'new',
     problem:{id:'G-56',title:'Merge Intervals',difficulty:'Medium',category:'Intervals'},action:{type:'mock',problem_id:'G-56'}},
    {id:'mock2',title:'Live mock 2',minutes:45,done:false,kind:'mock',detail:'new',
     problem:{id:'G-42',title:'Trapping Rain Water',difficulty:'Hard',category:'Monotonic Queues'},action:{type:'mock',problem_id:'G-42'}},
    {id:'behavioral',title:'Write one STAR story',minutes:15,done:false,detail:'0/8',action:{type:'star'}}]};

function env(mode='code_red',data=DAY){
  const nav=[];
  const ctx={h:tree,btn:(label,onclick,o={})=>tree('button',{'data-test':o.test,onclick,class:o.cls},label),bar:()=>tree('bar'),
    pageTitle:(...x)=>tree('title',{},...x),setCrumbs:()=>{},plural:(n,w)=>`${n} ${w}${n===1?'':'s'}`,fmtDate:x=>x,
    mount:(_,node)=>{ctx.root=node;},go:h=>{nav.push(h);},toast:()=>{},openModal:async()=>false,confirmBox:async()=>false,
    openPreparationMode:()=>{},resumeSession:()=>{},route:async()=>{},ROUTES:{},
    S:{state:{settings:{preparation_mode:mode},session:null}},isCodeRed:()=>mode==='code_red',
    api:async name=>name==='code_red_today'?data:name==='code_red_queue'?{rows:[],due_count:0}:{},
    Date,Math,setInterval,clearInterval,console};
  vm.createContext(ctx);vm.runInContext(source,ctx);
  return {ctx,nav,CodeRed:vm.runInContext('CodeRed',ctx)};
}
const context={alive:()=>true};

test('checklist highlights the first open block and Enter starts it as a 45-minute mock',async()=>{
  const {ctx,nav,CodeRed}=env();
  const handle=await CodeRed.today(context);
  const strike=find(ctx.root,n=>n.attrs['data-test']==='cr-block-strike');
  assert.match(strike.attrs.class,/is-next/);
  assert.match(strike.textContent,/Number of Islands/);
  assert.ok(find(ctx.root,n=>n.attrs['data-test']==='cr-block-warmup').textContent.includes('Done'));
  assert.equal(handle.onKey({key:'Enter'},false),true);
  assert.equal(JSON.stringify(ctx.S.boardStart),JSON.stringify({problem_id:'G-200',practice:false}));
  assert.deepEqual(nav,['#/board']);
});

test('readiness is shown as an estimate, never a hiring prediction',async()=>{
  const {ctx,CodeRed}=env();
  await CodeRed.today(context);
  const status=find(ctx.root,n=>n.attrs['data-test']==='cr-status');
  assert.match(status.textContent,/Readiness estimate/);
  assert.match(status.textContent,/Not a prediction of an offer/);
});

test('the story block opens STAR stories, and the warmup uses the Code Red drills',async()=>{
  const {ctx,nav,CodeRed}=env();
  await CodeRed.today(context);
  find(ctx.root,n=>n.attrs['data-test']==='cr-start-behavioral').attrs.onclick();
  assert.deepEqual(nav,['#/star']);
  const done={...DAY,blocks:DAY.blocks.map((b,i)=>i===0?{...b,done:false}:b)};
  const second=env('code_red',done);
  await second.CodeRed.today(context);
  find(second.ctx.root,n=>n.attrs['data-test']==='cr-start-warmup').attrs.onclick();
  assert.deepEqual(second.nav,['#/recall?mode=code_red']);
});

test('the Re-Solve Queue stays inside Code Red',async()=>{
  const outside=env('google');
  await outside.ctx.ROUTES.queue(context);
  assert.match(outside.ctx.root.textContent,/part of Code Red/);
  const inside=env('code_red');
  await inside.ctx.ROUTES.queue(context);
  assert.ok(find(inside.ctx.root,n=>n.attrs['data-test']==='queue-empty'));
});
