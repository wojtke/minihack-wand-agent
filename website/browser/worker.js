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
async function draw(state, mapping, atlas) {
  let minX=79,minY=21,maxX=0,maxY=0;
  for(let y=0;y<21;y++)for(let x=0;x<79;x++) {
    const code=state.chars[y].charCodeAt(x);
    if(code!==32&&code!==0){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x+1);maxY=Math.max(maxY,y+1);}
  }
  if(minX>=maxX){minX=minY=0;maxX=79;maxY=21;}
  const canvas=new OffscreenCanvas((maxX-minX)*16,(maxY-minY)*16);
  const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;
  for(let y=minY;y<maxY;y++)for(let x=minX;x<maxX;x++) {
    const tile=mapping[state.glyphs[y*79+x]];
    context.drawImage(atlas,(tile%32)*16,Math.floor(tile/32)*16,16,16,(x-minX)*16,(y-minY)*16,16,16);
  }
  const bytes=new Uint8Array(await (await canvas.convertToBlob({type:'image/png'})).arrayBuffer());
  // Crop only the observable/remembered view, exactly like the native demo.
  state.image='data:image/png;base64,'+btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));
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
    } else if(operation==='step') {
      if(!game)throw new Error('Start a game first.');
      game.step(key);
    } else {throw new Error('Unknown operation.');}
    self.postMessage({id,state:await draw(game.state(),mapping,atlas)});
  } catch(error) {
    self.postMessage({id,error:error.message||'The game could not start. Please reload the page.'});
  }
};
