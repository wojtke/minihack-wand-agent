'use strict';
importScripts('wand.js');
let game;
const checked=async url=>{const response=await fetch(url);if(!response.ok)throw new Error('Could not load game files. Please reload the page.');return response;};
const resources=Promise.all([
  import('./runtime.mjs'),
  checked('levels.json').then(r=>r.json()),
  checked('tiles.json').then(r=>r.json()),
  checked('tiles.png').then(r=>r.blob()).then(createImageBitmap)
]);
// Same glyph classes as nethack.glyph_is_monster / glyph_is_object in NLE 1.1.
const isMonster=g=>(g>=0&&g<=761)||(g>=763&&g<=1143)||(g>=1525&&g<=1905);
const isObject=g=>(g>=1144&&g<=1524)||(g>=1906&&g<=2358)||(g>=5595&&g<=5975);
let memory;
const newMemory=()=>({objects:new Set(),exits:new Set(),visits:new Int32Array(21*79),items:new Set()});
// Mirrors PublicMemory.encode in experiments/wand_agent/features.py (one update per observation).
function remember(state) {
  const [x,y]=state.position,here=y*79+x,code=i=>state.chars[Math.floor(i/79)].charCodeAt(i%79);
  for(let i=0;i<21*79;i++){if(isObject(state.glyphs[i]))memory.objects.add(i);if(code(i)===62)memory.exits.add(i);}
  const items=new Set(state.inventory.map(item=>item.key));
  if([...items].some(key=>!memory.items.has(key)))memory.objects.delete(here);
  memory.items=items;
  for(const i of [...memory.objects])if(!(isObject(state.glyphs[i])||i===here||code(i)===32))memory.objects.delete(i);
  memory.visits[here]+=1;
}
function nearest(points,count,x,y) {
  return [...points].map(i=>[i%79,Math.floor(i/79)])
    .sort((a,b)=>((a[0]-x)**2+(a[1]-y)**2)-((b[0]-x)**2+(b[1]-y)**2)||(a[1]-b[1])||(a[0]-b[0])).slice(0,count);
}
async function png(canvas) {
  const bytes=new Uint8Array(await (await canvas.convertToBlob({type:'image/png'})).arrayBuffer());
  return 'data:image/png;base64,'+btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));
}
async function draw(state, mapping, atlas) {
  let minX=79,minY=21,maxX=0,maxY=0;
  for(let y=0;y<21;y++)for(let x=0;x<79;x++) {
    const code=state.chars[y].charCodeAt(x);
    if(code!==32&&code!==0){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x+1);maxY=Math.max(maxY,y+1);}
  }
  if(minX>=maxX){minX=minY=0;maxX=79;maxY=21;}
  const tileAt=(context,x,y)=>{const tile=mapping[state.glyphs[y*79+x]];
    context.drawImage(atlas,(tile%32)*16,Math.floor(tile/32)*16,16,16,(x-minX)*16,(y-minY)*16,16,16);};
  const canvas=new OffscreenCanvas((maxX-minX)*16,(maxY-minY)*16);
  const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;
  for(let y=minY;y<maxY;y++)for(let x=minX;x<maxX;x++)tileAt(context,x,y);
  // Crop only the observable/remembered view, exactly like the native demo.
  state.image=await png(canvas);
  // Agent view: only the policy's inputs. A 9x9 window around the player (tinted by the
  // agent's own visit counts) plus the positions of the nearest remembered items (3),
  // staircases (2) and visible monsters (2). Item and monster types are not inputs.
  remember(state);
  // Centred on the player: the 9x9 input window in the middle of a 25x13 canvas. Positions
  // beyond the canvas are drawn at its edge, in their direction (the inputs cap at 20).
  const RX=12,RY=6,agent=new OffscreenCanvas((2*RX+1)*16,(2*RY+1)*16),view=agent.getContext('2d');
  view.imageSmoothingEnabled=false;view.fillStyle='#000';view.fillRect(0,0,agent.width,agent.height);
  const [px,py]=state.position;
  for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++){
    const x=px+dx,y=py+dy;if(x<0||x>78||y<0||y>20)continue;
    const tile=mapping[state.glyphs[y*79+x]];
    view.drawImage(atlas,(tile%32)*16,Math.floor(tile/32)*16,16,16,(RX+dx)*16,(RY+dy)*16,16,16);
    const visits=Math.min(memory.visits[y*79+x],8);
    if(visits){view.fillStyle=`rgba(40,120,255,${0.35+0.35*visits/8})`;view.fillRect((RX+dx)*16,(RY+dy)*16,16,16);}
  }
  view.strokeStyle='#fff';view.lineWidth=1;view.strokeRect((RX-4)*16+.5,(RY-4)*16+.5,9*16-1,9*16-1);
  const mark=(points,color,fill)=>{
    for(const [x,y] of points){
      let dx=x-px,dy=y-py;const far=Math.max(Math.abs(dx),Math.abs(dy));
      if(far<=4)continue;
      const k=Math.min(1,dx?RX/Math.abs(dx):1,dy?RY/Math.abs(dy):1);dx=Math.round(dx*k);dy=Math.round(dy*k);
      view.fillStyle=fill;view.fillRect((RX+dx)*16+2,(RY+dy)*16+2,12,12);
      view.strokeStyle=color;view.lineWidth=2;view.strokeRect((RX+dx)*16+2,(RY+dy)*16+2,12,12);
    }
  };
  const monsters=[];
  for(let i=0;i<21*79;i++)if(isMonster(state.glyphs[i])&&i!==py*79+px)monsters.push(i);
  mark(nearest(memory.objects,3,px,py),'#f2c200','rgba(242,194,0,.35)');
  mark(nearest(memory.exits,2,px,py),'#39d353','rgba(57,211,83,.35)');
  mark(nearest(monsters,2,px,py),'#ff4d4d','rgba(255,77,77,.35)');
  state.agent_image=await png(agent);
  delete state.glyphs;
  return state;
}
self.onmessage=async event=>{
  const {id,operation,level,seed,key}=event.data;
  try {
    const [{WandGame},levels,mapping,atlas]=await resources;
    if(operation==='reset') {
      if(!Object.hasOwn(levels,level))throw new Error('Unknown level.');
      const [module,nhdat]=await Promise.all([
        createWandEngine({locateFile:file=>new URL(file,self.location.href).href,print:()=>{},printErr:()=>{}}),
        checked(level.toLowerCase()+'.nhdat').then(r=>r.arrayBuffer())
      ]);
      game=new WandGame(module,level,seed,levels[level],new Uint8Array(nhdat));
      memory=newMemory();
    } else if(operation==='step') {
      if(!game)throw new Error('Start a game first.');
      game.step(key);
    } else {throw new Error('Unknown operation.');}
    self.postMessage({id,state:await draw(game.state(),mapping,atlas)});
  } catch(error) {
    self.postMessage({id,error:error.message||'The game could not start. Please reload the page.'});
  }
};
