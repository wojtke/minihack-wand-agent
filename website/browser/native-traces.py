"""Native oracle for the browser port. Run from the repository root."""
import json
from pathlib import Path
import torch
from wand_agent.native import LEVELS, KEYS
import base64
import io
import numpy as np
from PIL import Image
from wand_agent.native import make_env, text
from wand_agent.terminal import FinalObservation
from wand_agent.network import load
from wand_agent.env import AgentEnv

ROOT = Path(__file__).resolve().parents[2]

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
    result = dict(image=frame(raw),chars=[''.join(map(chr,row)) for row in raw['chars']],
                position=list(map(int,raw['blstats'][:2])),hp=int(raw['blstats'][10]),
                hp_max=int(raw['blstats'][11]),game_time=int(raw['blstats'][20]),
                message=text(raw['message']),inventory=[dict(key=chr(int(k)),name=text(s))
                for k,s in zip(raw['inv_letters'],raw['inv_strs']) if k],
                shapes=dict(glyphs=list(raw['glyphs'].shape),chars=list(raw['chars'].shape),
                            colors=list(raw['colors'].shape),blstats=list(raw['blstats'].shape),
                            message=list(raw['message'].shape),inv_strs=list(raw['inv_strs'].shape)))

    result['glyphs'] = raw['glyphs'].ravel().tolist()
    return result


class Game:
    def __init__(self,level,seed):
        self.env=FinalObservation(make_env(level,'Restricted',pixels=True))
        try:
            assert ''.join(chr(int(x)) for x in self.env.unwrapped.actions)==KEYS
            self.level,self.seed,self.steps,self.total=level,seed,0,0.
            self.done=False
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


def main():
    torch.set_num_threads(1)
    model = load(ROOT / 'experiments/checkpoints/shared-best.npz')
    cases = []
    for level in LEVELS:
        for seed in [0, 1, 42, 1234, 210000000]:
            agent, game = AgentEnv(level), Game(level, seed)
            try:
                obs = agent.reset(seed)
                states, keys = [game.state], []
                while not game.done:
                    action = model.action(obs)
                    keys.append(KEYS[action])
                    obs, _, _, _ = agent.step(action)
                    states.append(game.step(keys[-1]).copy())
                cases.append(dict(level=level, seed=seed, keys=''.join(keys), states=states, kind='policy'))
            finally:
                agent.close()
                game.close()
        for key, kind in [('f', 'timeout'), ('l', 'move')]:
            game = Game(level, 0)
            try:
                keys, states = [], [game.state]
                while not game.done:
                    keys.append(key)
                    states.append(game.step(key).copy())
                cases.append(dict(level=level, seed=0, keys=''.join(keys), states=states, kind=kind))
            finally:
                game.close()
    # Initial, menu, movement and terminal images support browser pixel checks.
    for case in cases:
        for i, state in enumerate(case['states']):
            if i not in {0, 1, 2, len(case['states']) - 1}:
                state.pop('image', None)
    out = ROOT / '.build/native-traces.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cases, separators=(',', ':')))
    print('Native traces:', len(cases), 'games,', sum(len(c['keys']) for c in cases), 'actions')


if __name__ == '__main__':
    main()
