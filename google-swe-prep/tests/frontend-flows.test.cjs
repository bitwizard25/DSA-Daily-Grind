/* Regression checks of shipped route functions; no browser or network required. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const app = fs.readFileSync(path.join(__dirname, '../google_prep/gui/static/app.js'), 'utf8');
const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/design_prep.json'), 'utf8'));

function tree(tag, attrs, ...children) {
  const node = {tag, attrs:attrs||{}, children:children.flat(Infinity).filter(x=>x!=null),
    checked:!!attrs?.checked, value:attrs?.value||'', disabled:!!attrs?.disabled,
    replaceChildren(...kids) {this.children=kids.flat(Infinity).filter(x=>x!=null);},
    setAttribute(name,value) {this.attrs[name]=value;},
    get textContent() {return this.children.map(x=>typeof x==='object' ? x.textContent : String(x)).join('');},
    set textContent(value) {this.children=[value];},
  };
  return node;
}
function find(node, predicate) {
  if (!node || typeof node !== 'object') return null;
  if (predicate(node)) return node;
  for (const child of node.children||[]) {const result=find(child, predicate);if(result) return result;}
  return null;
}
function source(start, end) {const offset=app.indexOf(start);assert.ok(offset>=0);return app.slice(offset, app.indexOf(end, offset));}
function env(extra={}) {
  const state={h:tree, svg:tree, btn:(label, onclick, opts={})=>tree('button', {'data-test':opts.test, onclick}, label),
    rich:x=>x, icon:x=>tree('icon', {}, x), bar:()=>tree('bar'), setCrumbs:()=>{}, pageTitle:(...x)=>tree('title', {}, ...x),
    mount:(_, node)=>{state.root=node;}, go:()=>{}, toast:()=>{}, ROUTES:{},
    Date, JSON, URLSearchParams, ...extra};
  vm.createContext(state);return state;
}
const routeSource=source('ROUTES.today = async (ctx) => {', '\n};')+'\n};';
const context={query:new URLSearchParams(), alive:()=>true};

async function today(session, confirmed=true) {
  const calls=[];
  const state=env({api:async name=>{calls.push(name);return {kind:'deep', title:'Contains Duplicate', bullets:['Solve Contains Duplicate'], status:[], action:'mock'};},
    refreshState:async()=>({session, day:3, now:'2026-10-14T10:00:00', phase:{short:'Foundation'}}),
    optionalTraining:async()=>null, confirmBox:async()=>confirmed,
    route:async()=>{calls.push('route');}, dailyChecklist:()=>null, habitPreview:()=>null, progressMilestones:()=>null});
  vm.runInContext(routeSource, state);
  await state.ROUTES.today(context);
  return {state,calls};
}
test('Today recall recovery describes the recall session and restores Discard', async()=>{
  const {state,calls}=await today({kind:'recall', text:'Session restored: Pattern recall, 5 drills left'});
  const hero=find(state.root,n=>n.attrs['data-test']==='next-card');
  assert.match(hero.textContent, /Pattern recall, 5 drills left/);
  assert.match(hero.textContent, /remaining drills/);
  assert.doesNotMatch(hero.textContent, /Contains Duplicate/);
  await find(hero,n=>n.attrs['data-test']==='discard-session').attrs.onclick();
  assert.deepEqual(calls,['today','session_discard','route']);
});
test('cancelled Discard keeps the unfinished session', async()=>{
  const {state,calls}=await today({kind:'mock', text:'Session restored: Two Sum, 35m remaining'}, false);
  const hero=find(state.root,n=>n.attrs['data-test']==='next-card');
  assert.match(hero.textContent, /saved code, trace and timer/);
  assert.doesNotMatch(hero.textContent, /Contains Duplicate/);
  await find(hero,n=>n.attrs['data-test']==='discard-session').attrs.onclick();
  assert.deepEqual(calls,['today']);
});
test('optional coaching failure resolves to null with a bounded request', async()=>{
  let options;
  const state=env({api:async(_,__,opts)=>{options=opts;throw Error('offline');}});
  vm.runInContext(source('function optionalTraining()', '\nfunction downloadText'), state);
  assert.equal(await state.optionalTraining(),null);
  assert.equal(options.timeoutMs,4000);
});
test('Micro-Mode checklist asks only for the short session, not the full coding goal',()=>{
  const state=env();
  vm.runInContext(source('function dailyChecklist(', '\nfunction habitPreview'),state);
  const root=state.dailyChecklist({progress:{daily:{micro_completed:true, completed:0, total:2,
    items:[{title:'Practise 3 problems', optional:false}, {title:'Quick reflection', optional:true}]}}},true);
  assert.match(root.textContent,/10-minute practice/);
  assert.match(root.textContent,/1\/1/);
  assert.doesNotMatch(root.textContent,/Practise 3 problems/);
});
test('failed whiteboard save offers export, safe default and explicit escape', async()=>{
  let downloaded=false;
  const state=env({confirmBox:()=>assert.fail('failed save must show the recovery dialog'),
    openModal:async(title,body,buttons)=>{
      assert.match(body.textContent,/not confirmed/);
      assert.equal(buttons[0].value,false);assert.equal(buttons[0].primary,true);
      assert.equal(buttons[1].label,'Leave anyway');
      find(body,n=>n.attrs['data-test']==='download-pending').attrs.onclick();
      return true;
    }});
  vm.runInContext(source('async function confirmWhiteboardLeave(', '\nfunction progressMilestones'),state);
  assert.equal(await state.confirmWhiteboardLeave(async()=>false,()=>{downloaded=true;},()=>false),true);
  assert.ok(downloaded);
});
test('learning track opens a checklist and saves checked topics without a code pad', async()=>{
  const calls=[];
  const state=env({api:async(name,payload)=>{calls.push({name,payload});}, confirmBox:async()=>false});
  vm.runInContext(source('function designLearning(', '\nROUTES.design'), state);
  const track={...catalog.tracks[0], learned:[], learning_total:10};
  const page=state.designLearning(context,{...catalog,tracks:[track]},track,catalog.exercises.filter(x=>x.track==='lld'));
  assert.equal(find(state.root,n=>n.tag==='textarea'||n.tag==='pre'),null);
  assert.match(state.root.textContent,/Classes/);assert.match(state.root.textContent,/Errors/);
  const box=find(state.root,n=>n.attrs['data-test']==='learn-lld-contracts');
  box.checked=true;box.attrs.onchange();
  assert.ok(page.isDirty());assert.equal(await page.beforeLeave(),false);
  await find(state.root,n=>n.attrs['data-test']==='learning-save').attrs.onclick();
  assert.equal(calls[0].name,'design_learning_save');
  assert.equal(JSON.stringify(calls[0].payload.completed),'["lld-contracts"]');
  assert.equal(page.isDirty(),false);assert.equal(await page.beforeLeave(),true);
  assert.match(state.root.textContent,/1\/10 learned/);
});

test('update settings preserve preference after a failed save and stop polling on close',async()=>{
  let cleared=false, requests=[];
  const state=env({setInterval:()=>7,clearInterval:id=>{assert.equal(id,7);cleared=true;},
    api:async(name,payload)=>{requests.push(name);
      if(name==='update_preferences') throw Error('Could not save update preference.');
      return {supported:true,enabled:true,state:'idle',version:'1.6.0',build_number:100,message:'Up to date.'};}});
  vm.runInContext(source('function updaterSettings(', '\nasync function openSettings'),state);
  const ui=state.updaterSettings();await ui.refresh();
  const checkbox=find(ui.node,n=>n.attrs['data-test']==='auto-updates');
  checkbox.checked=false;await checkbox.attrs.onchange();
  assert.equal(checkbox.checked,true);
  assert.match(ui.node.textContent,/Could not save/);
  ui.cleanup();assert.ok(cleared);
});

test('update check stays disabled while downloading and disposed settings make no requests',async()=>{
  let requests=0;
  const state=env({setInterval:()=>7,clearInterval:()=>{},
    api:async()=>{requests++;return {supported:true,enabled:true,state:'downloading',version:'1.6.0',build_number:100,message:'Downloading'};}});
  vm.runInContext(source('function updaterSettings(', '\nasync function openSettings'),state);
  const ui=state.updaterSettings();await ui.refresh();
  const button=find(ui.node,n=>n.attrs['data-test']==='update-check');
  await button.attrs.onclick();assert.equal(button.disabled,true);
  ui.cleanup();const before=requests;await ui.refresh();assert.equal(requests,before);
});

test('sidebar update indicator uses real download progress and handles unavailable states',()=>{
  const state=env();
  vm.runInContext(source('function updateIndicatorState(', '\nfunction startUpdateIndicator'),state);
  const show=status=>state.updateIndicatorState(status);
  assert.equal(show({state:'available'}).available,true);
  const download=show({state:'downloading',downloaded:25,total:100});
  assert.equal(download.progress,.25);assert.equal(download.detail,'25% downloaded');
  assert.equal(show({state:'downloading',downloaded:150,total:100}).progress,1);
  assert.equal(show({state:'downloading',downloaded:10,total:0}).detail,'Downloading…');
  assert.equal(show({state:'ready',enabled:true}).detail,'Installs after closing');
  assert.equal(show({state:'ready',enabled:false}).detail,'Automatic updates off');
  assert.equal(show({state:'unsupported'}).detail,'Manual download');
  assert.equal(show({state:'offline'}).available,false);
  assert.equal(show({state:'failed'}).detail,'Needs attention');
});

test('installed updates remain clickable and do not send automatic downloads to GitHub',async()=>{
  const state=env({setInterval:()=>1,clearInterval:()=>{},api:async()=>({supported:true,enabled:true,
    state:'ready',version:'1.6.0',build_number:100,release_url:'https://github.com/release',message:'Ready'})});
  vm.runInContext(source('function updaterSettings(', '\nasync function openSettings'),state);
  const ui=state.updaterSettings({showHeading:false});await ui.refresh();
  const checkbox=find(ui.node,n=>n.attrs['data-test']==='auto-updates');
  assert.equal(checkbox.disabled,false);assert.equal(checkbox.checked,true);
  assert.equal(find(ui.node,n=>n.tag==='button'&&n.textContent==='Open download page').hidden,true);
  assert.equal(find(ui.node,n=>n.tag==='h3'),null);
  ui.cleanup();
});

test('unsupported copies explain the disabled checkbox even after detecting a release',async()=>{
  const state=env({setInterval:()=>1,clearInterval:()=>{},api:async()=>({supported:false,enabled:true,
    state:'available',version:'1.6.0',build_number:0,release_url:'https://github.com/release',message:'Available'})});
  vm.runInContext(source('function updaterSettings(', '\nasync function openSettings'),state);
  const ui=state.updaterSettings();await ui.refresh();
  assert.equal(find(ui.node,n=>n.attrs['data-test']==='auto-updates').disabled,true);
  const reason=find(ui.node,n=>n.tag==='p'&&n.textContent.includes('unavailable in this copy'));
  assert.equal(reason.hidden,false);
  assert.equal(find(ui.node,n=>n.tag==='button'&&n.textContent==='Open download page').hidden,false);
  ui.cleanup();
});

test('sidebar polling never overlaps and stops after page close',async()=>{
  let tick, resolve, calls=0, removed=false, cleared=false;
  const button=tree('button');
  const state=env({$:()=>button, COLORS:{blue:'#4285F4'},ring:()=>tree('svg'),
    document:{visibilityState:'visible',addEventListener:()=>{},removeEventListener:()=>{removed=true;}},
    setInterval:fn=>{tick=fn;return 1;},clearInterval:()=>{cleared=true;},
    api:()=>{calls++;return new Promise(r=>{resolve=r;});}});
  vm.runInContext(source('function updateIndicatorState(', '\nasync function openUpdates'),state);
  const stop=state.startUpdateIndicator();
  await tick();assert.equal(calls,1);
  stop();resolve({state:'available'});await Promise.resolve();
  await tick();assert.equal(calls,1);assert.ok(removed && cleared);
  assert.equal(button.textContent,'');
});
