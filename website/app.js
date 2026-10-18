'use strict';
const $=s=>document.querySelector(s);
const slides=[...document.querySelectorAll('.slide')];
const playIndex=slides.findIndex(slide=>slide.classList.contains('play-slide'));
const demoKeys={w:'k',a:'h',s:'j',d:'l',q:'y',e:'u',z:'b',c:'n',p:',',r:'z',f:'f',ArrowUp:'k',ArrowDown:'j',ArrowLeft:'h',ArrowRight:'l'};
const dialog=$('#details-dialog');
let current=0,publicObservation=null,backend=false,game=null,busy=false;
let browserGame=null;
window.addEventListener('pagehide',event=>{if(!event.persisted)browserGame?.close();});
const sources=[
  ['MiniHack task documentation','https://minihack.readthedocs.io/en/stable/envs/skills/wod.html'],
  ['Level definitions used in this demo','https://github.com/facebookresearch/minihack/blob/09eaa7ee45c567850668d3743d7fe1853917d98f/minihack/envs/skills_wod.py'],
  ['MiniHack task rewards','https://github.com/facebookresearch/minihack/blob/09eaa7ee45c567850668d3743d7fe1853917d98f/minihack/base.py'],
  ['Game-time penalty in NLE 1.1.0','https://github.com/NetHack-LE/nle/blob/v1.1.0/nle/env/tasks.py']
];
const goals={Easy:'Success: kill the minotaur.',Medium:'Success: reach the staircase beyond the minotaur.',Hard:'Success: reach the staircase beyond the minotaur.',Pro:'Success: reach the staircase.'};

function modal(title,node){$('#dialog-title').textContent=title;$('#dialog-content').replaceChildren(node);dialog.showModal();}
function paragraph(text){const p=document.createElement('p');p.textContent=text;return p;}
function closeDialog(){dialog.close();}
$('.dialog-close').addEventListener('click',closeDialog);
dialog.addEventListener('click',e=>{if(e.target===dialog&&e.clientX>=0){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeDialog();}});

slides.forEach((slide,i)=>{const button=document.createElement('button');button.setAttribute('aria-label',`Slide ${i+1}: ${slide.dataset.title}`);button.addEventListener('click',()=>showSlide(i));$('#slide-dots').append(button);});
function showSlide(index,updateHash=true){
  current=Math.max(0,Math.min(slides.length-1,index));
  slides.forEach((slide,i)=>{slide.hidden=i!==current;slide.classList.toggle('active',i===current);if(i!==current)slide.querySelectorAll('video').forEach(v=>v.pause());});
  [...$('#slide-dots').children].forEach((b,i)=>{b.classList.toggle('current',i===current);b.setAttribute('aria-current',i===current?'step':'false');});
  $('#slide-count').textContent=`${String(current+1).padStart(2,'0')} / ${String(slides.length).padStart(2,'0')}`;
  $('#prev').disabled=current===0;$('#next').disabled=current===slides.length-1;
  if(updateHash)history.replaceState(null,'',`#${current+1}`);
  window.scrollTo({top:0,behavior:'instant'});
  if(current===0&&!matchMedia('(prefers-reduced-motion: reduce)').matches)slides[0].querySelectorAll('video').forEach(v=>{if(v.dataset.userPaused!=='true')v.play().catch(()=>{});});
  if(current===playIndex&&backend&&!game&&!busy)startGame(0);
}
$('#prev').addEventListener('click',()=>showSlide(current-1));$('#next').addEventListener('click',()=>showSlide(current+1));
window.addEventListener('hashchange',()=>showSlide((Number(location.hash.slice(1))||1)-1,false));
$('#fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{}});
document.addEventListener('fullscreenchange',()=>{$('#fullscreen').firstChild.textContent=document.fullscreenElement?'Exit full screen':'Full screen';});
document.addEventListener('keydown',e=>{
  if(dialog.open)return;
  const target=e.target;
  if(['SELECT','INPUT','TEXTAREA'].includes(target.tagName))return;
  if(current===playIndex&&$('#game-board').contains(document.activeElement)){
    if(e.key==='Escape'){$('#game-board').blur();e.preventDefault();return;}
    if(e.ctrlKey||e.metaKey||e.altKey)return;
    if(e.code==='Space'||e.key===' '){e.preventDefault();if(!e.repeat)restartGame();return;}
    const key=demoKeys[e.key]||demoKeys[e.key.toLowerCase()];
    if(key){e.preventDefault();sendAction(key);}
    return;
  }
  if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();showSlide(current+1);}
  if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();showSlide(current-1);}
  if(e.key==='Home'){e.preventDefault();showSlide(0);}
  if(e.key==='End'){e.preventDefault();showSlide(slides.length-1);}
});

document.querySelectorAll('.video-toggle').forEach(button=>{
  const video=document.getElementById(button.dataset.video);
  function sync(){button.textContent=video.paused?'Play':'Pause';button.setAttribute('aria-label',(video.paused?'Play ':'Pause ')+video.id.replace('video-','')+' recording');button.setAttribute('aria-pressed',String(!video.paused));}
  function toggle(){if(video.paused){video.dataset.userPaused='false';video.play().catch(()=>{});}else{video.dataset.userPaused='true';video.pause();}}
  button.addEventListener('click',toggle);video.addEventListener('click',toggle);
  video.addEventListener('play',sync);video.addEventListener('pause',sync);sync();
});

fetch('assets/public-observation.json').then(r=>r.json()).then(data=>{publicObservation=data;}).catch(()=>{});
function showObservation(state){
  const div=document.createElement('div');
  div.append(paragraph('Example observation fields. Hidden engine state and unexplored terrain are not included.'));
  const pre=document.createElement('pre');
  pre.textContent=JSON.stringify(state?{message:state.message,inventory:state.inventory,position:state.position,hp:state.hp,hp_max:state.hp_max,game_time:state.game_time,shapes:state.shapes}: {status:'Observation not yet available.'},null,2);
  div.append(pre);modal('Observation and inventory',div);
}
$('#show-observation').addEventListener('click',()=>showObservation(game||publicObservation));
$('#notes-button').addEventListener('click',()=>{
  const div=document.createElement('div');div.append(paragraph(slides[current].querySelector('.speaker-note').textContent));
  const heading=document.createElement('h3');heading.textContent='Sources';div.append(heading);
  const list=document.createElement('ul');for(const [title,url] of sources){const li=document.createElement('li'),a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener';a.textContent=title;li.append(a);list.append(li);}div.append(list);
  div.append(paragraph('Demo versions: MiniHack 1.0.0+09eaa7e and NLE 1.1.0, Restricted variant. Videos are recordings of real local episodes. Presentation code and materials were prepared with AI assistance; see the repository for source credits and recording details.'));
  const credits=document.createElement('a');credits.href='https://github.com/wojtke/minihack-wand-agent/blob/main/README.md#sources-and-licenses';credits.target='_blank';credits.rel='noopener';credits.textContent='Project source and artwork credits';div.append(credits);
  modal(`Slide ${current+1} · notes`,div);
});
$('#offline-replay').addEventListener('click',()=>showSlide(0));
$('#local-instructions').addEventListener('click',()=>{
  const div=document.createElement('div');
  div.append(paragraph('Play in a current browser with WebAssembly support. No installation or game server is needed.'));
  div.append(paragraph('The repository also includes a native Python environment for training and watching the agent in a terminal. Setup instructions cover macOS and Linux; on Windows use WSL2.'));
  const link=document.createElement('a');link.href='https://github.com/wojtke/minihack-wand-agent#setup-for-training';link.target='_blank';link.rel='noopener';link.textContent='Setup instructions on GitHub';div.append(link);
  modal('Browser and Python versions',div);
});
dialog.addEventListener('close',()=>dialog.querySelectorAll('video').forEach(v=>v.pause()));

function setBusy(value){busy=value;document.body.classList.toggle('game-busy',value);$('#new-game').disabled=value||!backend;$('#restart-game').disabled=value||!backend||!game;$('#game-level').disabled=value||!backend;}
async function api(path,data){return browserGame.request(path,data);}
async function startGame(seed){
  if(!backend||busy)return;
  setBusy(true);$('#game-state').textContent='Loading game…';
  try{game=await api('reset',{level:$('#game-level').value,seed});renderGame();$('#game-board').focus({preventScroll:true});}
  catch(e){game=null;$('#game-message').textContent=e.message;$('#game-state').textContent='Could not start the game';}
  finally{setBusy(false);}
}
async function sendAction(key){
  if(!backend||busy||!game||game.done)return;
  setBusy(true);try{game=await api('step',{key});renderGame();}catch(e){$('#game-message').textContent=e.message;}finally{setBusy(false);}
}
function renderGame(){
  $('#game-placeholder').hidden=true;$('#game-image').hidden=false;$('#game-image').src=game.image;
  const labels={RUNNING:'playing',TASK_SUCCESSFUL:'success',DEATH:'death',ABORTED:'action limit'};
  $('#game-state').textContent=`${game.level} · ${labels[game.status]||game.status} · HP ${game.hp}/${game.hp_max}`;
  $('#game-counters').textContent=`${game.steps}/${game.horizon} actions · return ${game.total.toFixed(2)}`;
  $('#game-message').textContent=game.steps===0?'Choose a direction or command to begin.':game.message||'…';
  const selecting=game.message?.includes('What do you want to zap?');
  const item=game.inventory.find(item=>item.key==='f');
  $('[data-key="f"] .command-label').textContent=selecting?(item?.name.includes('wand')?'Select wand':'Select item f'):'Fire';
  const list=$('#inventory-list');list.replaceChildren();
  for(const item of game.inventory){
    const li=document.createElement('li'),id=document.createElement('span'),name=document.createElement('span');
    id.className='item-id';id.textContent=item.key;name.textContent=item.name;
    li.append(id,name);li.classList.toggle('selected-item',Boolean(selecting&&item.key==='f'));
    list.append(li);
  }
  if(!game.inventory.length){const li=document.createElement('li');li.textContent='Empty inventory.';list.append(li);}
  $('#inventory-prompt').hidden=!selecting;
  $('#inventory-prompt').textContent=item?.name.includes('wand')?'Press F to select the wand (item f).':'Press F to select item f.';
  const overlay=$('#game-outcome');overlay.hidden=!game.done;
  if(game.done){overlay.replaceChildren(document.createTextNode(game.status==='TASK_SUCCESSFUL'?'Task successful':game.status==='DEATH'?'Game over':'Action limit'));const small=document.createElement('small');small.textContent=`${game.steps} actions · last reward ${game.reward.toFixed(2)}`;overlay.append(small);}
  updateHints();
}
function updateHints(){
  const box=$('#hint-text');box.hidden=!$('#hints').checked;
  $('#inventory-prompt').hidden=!game?.message?.includes('What do you want to zap?')||$('#hints').checked;
  if(!game){box.textContent='Start a game. Hints explain controls and messages; they do not perform actions.';return;}
  if(game.done){box.textContent=goals[game.level]+' Start a new game or choose another level.';return;}
  const message=game.message||'';
  if(message.includes('What do you want to zap?'))box.textContent='Press F to select the highlighted wand (item f).';
  else if(message.includes('In what direction?'))box.textContent='Choose one of the eight directions. This key aims the wand instead of moving the player.';
  else if(message.includes('What do you want to'))box.textContent='Menu prompts reuse the same inputs. F sends the native item letter f.';
  else box.textContent='Pick up with P. Use a wand: R → F → direction. '+goals[game.level];
}
$('#hints').addEventListener('change',updateHints);
function restartGame(){if(game&&!busy)startGame(game.seed);}
$('#restart-game').addEventListener('click',restartGame);
$('#new-game').addEventListener('click',()=>startGame(game?.level===$('#game-level').value?game.seed+1:0));
$('#game-level').addEventListener('change',()=>startGame(0));
document.querySelectorAll('[data-key]').forEach(b=>b.addEventListener('click',()=>{$('#game-board').focus({preventScroll:true});sendAction(b.dataset.key);}));
$('#game-board').addEventListener('focus',()=>{$('#focus-help').textContent='Keyboard controls active · Esc returns to slide navigation.';});
$('#game-board').addEventListener('blur',()=>{$('#focus-help').textContent='Click the board to use the keyboard. Esc returns to the slides.';});
async function connect(){
  try{
    if(!window.WebAssembly||!window.Worker||!window.OffscreenCanvas)throw new Error('Browser support missing');
    const {BrowserGame}=await import('./engine/client.mjs');browserGame=new BrowserGame();backend=true;
  }catch{backend=false;}
  $('#offline-note').hidden=backend;slides[playIndex].classList.toggle('offline',!backend);setBusy(false);
  if(backend){$('#game-state').textContent='MiniHack ready';if(current===playIndex&&!game)startGame(0);}
}
showSlide((Number(location.hash.slice(1))||1)-1,false);connect();
