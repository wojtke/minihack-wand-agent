"""Serve the presentation and real, unmodified MiniHack games locally."""
import argparse
import base64
import io
import json
from pathlib import Path
import secrets
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image
from .native import make_env, text
from .terminal import FinalObservation

LEVELS=('Easy','Medium','Hard','Pro')
KEYS='kljhunby,zf'
GAMES={}
MAX_GAMES=8
SESSION_IDLE_SECONDS=900
# NLE construction temporarily changes the process working directory. Serialize
# all engine access, while allowing videos and health checks to load concurrently.
ENGINE_LOCK=threading.RLock()


def frame(raw):
    stream=io.BytesIO()
    # Zoom only to publicly visible/remembered cells. Unknown terrain stays
    # unknown; raw 21 x 79 arrays remain unchanged in the observation.
    ys,xs=np.where((raw['chars']!=32)&(raw['chars']!=0))
    picture=Image.fromarray(raw['pixel'])
    if len(xs):picture=picture.crop((16*int(xs.min()),16*int(ys.min()),16*int(xs.max()+1),16*int(ys.max()+1)))
    picture.save(stream,format='PNG',optimize=False)
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def observation(raw):
    return dict(image=frame(raw),chars=[''.join(map(chr,row)) for row in raw['chars']],
                position=list(map(int,raw['blstats'][:2])),hp=int(raw['blstats'][10]),
                hp_max=int(raw['blstats'][11]),game_time=int(raw['blstats'][20]),
                message=text(raw['message']),inventory=[dict(key=chr(int(k)),name=text(s))
                for k,s in zip(raw['inv_letters'],raw['inv_strs']) if k],
                shapes=dict(glyphs=list(raw['glyphs'].shape),chars=list(raw['chars'].shape),
                            colors=list(raw['colors'].shape),blstats=list(raw['blstats'].shape),
                            message=list(raw['message'].shape),inv_strs=list(raw['inv_strs'].shape)))


class Game:
    def __init__(self,level,seed):
        self.env=FinalObservation(make_env(level,'Restricted',pixels=True))
        try:
            assert ''.join(chr(int(x)) for x in self.env.unwrapped.actions)==KEYS
            self.level,self.seed,self.steps,self.total=level,seed,0,0.
            self.done=False;self.last_used=time.monotonic()
            raw,_=self.env.reset(seed=seed)
            self.last_observation=observation(raw)
            self.state=self.serialize(0.,'RUNNING')
        except BaseException:
            self.env.close()
            raise

    def serialize(self,reward,status):
        return dict(level=self.level,seed=self.seed,steps=self.steps,total=self.total,
                    reward=reward,status=status,done=self.done,horizon=self.env.unwrapped._max_episode_steps,
                    **self.last_observation)

    def step(self,key):
        if self.done:raise ValueError('This game has ended. Please start a new game.')
        self.last_used=time.monotonic()
        raw,reward,term,trunc,info=self.env.step(KEYS.index(key))
        self.steps+=1;self.total+=float(reward);self.done=bool(term or trunc)
        if self.done and self.env.saved is not None:raw=self.env.saved
        # NLE clears its screen on some terminal events. Keep the last actual
        # public screen and label the native outcome separately.
        if not self.done or np.any(raw['chars']==ord('@')):
            self.last_observation=observation(raw)
        self.state=self.serialize(float(reward),info['end_status'].name)
        self.state['last_key']=key
        return self.state

    def close(self):self.env.close()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'docs'),**kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_json(self,data,status=200):
        encoded=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(encoded)));self.send_header('Cache-Control','no-store')
        try:self.end_headers();self.wfile.write(encoded)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_GET(self):
        if urlsplit(self.path).path=='/api/health':
            self.send_json(dict(ready=True,engine='MiniHack / NLE',levels=LEVELS,keys=KEYS));return
        super().do_GET()

    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
            self.send_json({'error':'Origin mismatch'},403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8192:raise ValueError('Invalid request.')
            if self.headers.get_content_type()!='application/json':raise ValueError('JSON required.')
            data=json.loads(self.rfile.read(length));path=urlsplit(self.path).path
            if not isinstance(data,dict):raise ValueError('A JSON object is required.')
            with ENGINE_LOCK:result,status=self.game_request(path,data)
            self.send_json(result,status)
        except (ValueError,TypeError) as e:self.send_json({'error':str(e)},400)
        except Exception:
            import traceback
            traceback.print_exc();self.send_json({'error':'Game engine error. Please start a new game.'},500)

    def game_request(self,path,data):
        sid=data.get('session')
        if sid is not None and not isinstance(sid,str):
            raise ValueError('Invalid session.')
        for expired in list(GAMES):
            if time.monotonic()-GAMES[expired].last_used>SESSION_IDLE_SECONDS:
                GAMES.pop(expired).close()
        if path=='/api/close':
            game=GAMES.pop(sid,None)
            if game is not None:game.close()
            return {'closed':True},200
        if path=='/api/reset':
            level=data.get('level','Easy');seed=data.get('seed',0)
            if level not in LEVELS or type(seed) is not int or not 0<=seed<2**31:
                raise ValueError('Invalid level or seed.')
            prior=data.get('session')
            if prior in GAMES:GAMES.pop(prior).close()
            # A local demo must recover even if a tab crashed before releasing
            # its session. Keep memory bounded by retiring the least-recently
            # used game; that tab can restart if it is revisited.
            if len(GAMES)>=MAX_GAMES:
                oldest=min(GAMES,key=lambda token:GAMES[token].last_used)
                GAMES.pop(oldest).close()
            game=Game(level,seed);sid=secrets.token_urlsafe(18);GAMES[sid]=game
            return dict(session=sid,**game.state),200
        if path=='/api/step':
            sid=data.get('session');key=data.get('key')
            if not isinstance(key,str) or len(key)!=1 or key not in KEYS:raise ValueError('This action is not one of the 11 available keys.')
            if sid not in GAMES:return {'error':'Game expired. Please start a new game.'},404
            return dict(session=sid,**GAMES[sid].step(key)),200
        return {'error':'Unknown operation.'},404


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8768)
    p.add_argument('--open',action='store_true');a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.daemon_threads=False
    print(f'Wand of Death: http://127.0.0.1:{a.port}',flush=True)
    if a.open:
        import webbrowser
        threading.Timer(.5,lambda:webbrowser.open(f'http://127.0.0.1:{a.port}')).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        with ENGINE_LOCK:
            for game in GAMES.values():game.close()


if __name__=='__main__':main()
