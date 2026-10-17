// NLE-derived port, 2026-10-17. NetHack General Public License; see licenses/NetHack.txt.
// MiniHack 09eaa7e / NLE 1.1.0 task wrapper around the original wasm engine.
// Ported from minihack/base.py and nle/env/{base,tasks}.py; see browser/README.md.
export const KEYS='kljhunby,zf';
export function readRaw(m) {
  const array=(field,heap,length)=>{const pointer=m._wand_pointer(field)/heap.BYTES_PER_ELEMENT;return Array.from(heap.subarray(pointer,pointer+length));};
  return {glyphs:array(0,m.HEAP16,21*79),chars:array(1,m.HEAPU8,21*79),colors:array(2,m.HEAPU8,21*79),
    blstats:array(3,m.HEAP32,27),message:m.UTF8ToString(m._wand_pointer(4)),
    inv_strs:Array.from({length:55},(_,i)=>m.UTF8ToString(m._wand_pointer(5)+i*80)),
    inv_letters:array(6,m.HEAPU8,55),internal:array(7,m.HEAP32,9),program:array(8,m.HEAP32,6),
    done:Boolean(m._wand_flag(0))};
}
export class WandGame {
  constructor(module,level,seed,config,nhdat) {
    if(!config||!Number.isInteger(seed)||seed<0||seed>=2**31)throw new Error('Invalid level or seed.');
    this.module=module;this.level=level;this.seed=seed;this.horizon=config.horizon;
    this.steps=0;this.total=0;this.done=false;
    module.FS.mkdir('/game');module.FS.mkdir('/game/save');
    for(const name of ['perm','record','logfile','xlogfile'])module.FS.writeFile('/game/'+name,'');
    module.FS.writeFile('/game/nhdat',nhdat);
    module.ccall('wand_start','number',['number','string'],[seed,config.options]);
    this.raw=readRaw(module);
    let count=0;
    while(!this.raw.program[3]&&count++<1000)this.nativeStep(' ');
    if(!this.raw.program[3]||this.raw.done)throw new Error('The game could not finish initialization.');
    this.visible=this.raw;
  }
  nativeStep(key){this.module._wand_step(key.charCodeAt(0));this.raw=readRaw(this.module);}
  step(key) {
    if(this.done)throw new Error('This game has ended. Start a new game.');
    if(typeof key!=='string'||key.length!==1||!KEYS.includes(key))throw new Error('Invalid game input.');
    const previousTime=this.raw.blstats[20];
    this.nativeStep(key);
    // NLE automatically dismisses --More-- and text-entry prompts. Single-key
    // inventory/direction prompts remain visible, exactly as in MiniHackSkill.
    let automatic=0;
    while(!this.raw.done) {
      if(++automatic>1000)throw new Error('The game is stuck in an automatic prompt.');
      if(this.raw.internal[3]){this.nativeStep(' ');continue;}
      if(this.raw.internal[2]){this.nativeStep('\x1b');continue;}
      break;
    }
    this.steps++;
    const aborted=this.steps>=this.horizon;
    const kill=!aborted&&this.level==='Easy'&&this.raw.message.includes('You kill the minotaur');
    const success=!aborted&&(kill||Boolean(this.raw.internal[4]));
    this.status=this.raw.done?'DEATH':aborted?'ABORTED':success?'TASK_SUCCESSFUL':'RUNNING';
    // Easy uses MessageEvent's collected reward; other tasks use success status.
    const taskReward=this.level==='Easy'?Number(kill):Number(this.status==='TASK_SUCCESSFUL');
    this.reward=taskReward+(this.raw.blstats[20]===previousTime?-.01:0);
    this.total+=this.reward;this.done=this.status!=='RUNNING';
    // Preserve the observable pre-quit screen. The native demo does the same.
    if(!this.done||this.raw.chars.includes(64))this.visible=this.raw;
    this.lastKey=key;
    return this.state();
  }
  state() {
    const r=this.visible;
    return {level:this.level,seed:this.seed,steps:this.steps,total:this.total,reward:this.reward||0,
      status:this.status||'RUNNING',done:this.done,horizon:this.horizon,last_key:this.lastKey,
      glyphs:r.glyphs,chars:Array.from({length:21},(_,y)=>String.fromCharCode(...r.chars.slice(y*79,(y+1)*79))),
      position:r.blstats.slice(0,2),hp:r.blstats[10],hp_max:r.blstats[11],game_time:r.blstats[20],message:r.message,
      inventory:r.inv_letters.flatMap((letter,i)=>letter?[{key:String.fromCharCode(letter),name:r.inv_strs[i]}]:[]),
      shapes:{glyphs:[21,79],chars:[21,79],colors:[21,79],blstats:[27],message:[256],inv_strs:[55,80]}};
  }
}
