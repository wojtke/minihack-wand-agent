// Each reset creates a fresh Worker: this resets all native globals and frees
// the previous WebAssembly heap. No sessions, API server or shared memory.
export class BrowserGame {
  constructor(){this.worker=null;this.pending=null;this.nextId=0;}
  close(){
    if(this.worker)this.worker.terminate();
    this.worker=null;
    if(this.pending){clearTimeout(this.pending.timer);this.pending.reject(new Error('Game restarted.'));this.pending=null;}
  }
  request(operation,data){
    if(operation==='reset') {
      this.close();
      this.worker=new Worker(new URL('worker.js',import.meta.url));
      this.worker.onmessage=event=>{
        const result=event.data,pending=this.pending;
        if(!pending||result.id!==pending.id)return;
        this.pending=null;clearTimeout(pending.timer);
        if(result.error)pending.reject(new Error(result.error));else pending.resolve(result.state);
      };
      this.worker.onerror=()=>{
        const pending=this.pending;this.pending=null;
        if(pending){clearTimeout(pending.timer);pending.reject(new Error('The browser game failed to load. Please restart or reload the page.'));}
      };
    }
    if(!this.worker)return Promise.reject(new Error('Start a game first.'));
    if(this.pending)return Promise.reject(new Error('Wait for the current action to finish.'));
    return new Promise((resolve,reject)=>{
      const id=++this.nextId;
      const timer=setTimeout(()=>{this.pending=null;this.close();reject(new Error('Loading the game timed out. Check your connection and start a new game.'));},30000);
      this.pending={id,resolve,reject,timer};
      this.worker.postMessage({id,operation,...data});
    });
  }
}
