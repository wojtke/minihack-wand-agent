"""Record short clips of a uniformly random agent for the presentation.

Each level uses the first seed of a range fixed before recording; no seed is reselected.
The policy draws one of the environment's 11 Restricted actions uniformly at every step.
Nothing is trained and no reward is changed.

Run from experiments/ with its environment activated:
    python tools/record_random.py
"""
# // START GENAI — Claude, 28.10.2026
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wand_agent.native import HORIZONS, LEVELS, make_env, text
from wand_agent.terminal import FinalObservation

OUT = Path(__file__).resolve().parents[2] / "website" / "assets"
SEED_BASE = 950_000_000          # fixed before the first recording
SHOWN = dict(Easy=50, Medium=150, Hard=400, Pro=150)   # Pro: first 150 of 1000 actions
FPS = dict(Easy=4, Medium=10, Hard=5, Pro=10)
TILE = 16


def visible_box(chars):
    ys, xs = np.where((chars != 32) & (chars != 0))
    if not len(xs):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def record(index, level):
    seed = SEED_BASE + index * 10_000
    rng = np.random.default_rng(seed)
    env = FinalObservation(make_env(level, "Restricted", pixels=True))
    obs, _ = env.reset(seed=seed)
    keys = [chr(int(k)) for k in env.unwrapped.actions]
    assert env.action_space.n == 11, env.action_space
    frames, boxes, trace = [], [], []
    status, total, steps = "EXCERPT", 0.0, 0
    while True:
        if visible_box(obs["chars"]) is not None:
            frames.append(obs["pixel"].copy())
            boxes.append(visible_box(obs["chars"]))
        if steps >= SHOWN[level]:
            break
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total += float(reward)
        trace.append(dict(step=steps, key=keys[action], reward=float(reward), message=text(obs["message"])))
        if terminated or truncated:
            status = info["end_status"].name
            if visible_box(obs["chars"]) is not None:
                frames.append(obs["pixel"].copy())
                boxes.append(visible_box(obs["chars"]))
            break
    env.close()

    x0 = max(0, min(b[0] for b in boxes) - 1) * TILE
    y0 = max(0, min(b[1] for b in boxes) - 1) * TILE
    x1 = (max(b[2] for b in boxes) + 1) * TILE
    y1 = (max(b[3] for b in boxes) + 1) * TILE
    name = f"random-{level.lower()}"
    with tempfile.TemporaryDirectory(prefix="uw-random-") as tmp:
        tmp = Path(tmp)
        for i, frame in enumerate(frames):
            image = Image.fromarray(frame).crop((x0, y0, min(x1, frame.shape[1]), min(y1, frame.shape[0])))
            image.save(tmp / f"{i:04d}.png")
            if i == 0:
                image.save(OUT / f"{name}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS[level]), "-i", str(tmp / "%04d.png"),
                        "-vf", "scale=iw*2:ih*2:flags=neighbor,pad=ceil(iw/2)*2:ceil(ih/2)*2,tpad=stop_mode=clone:stop_duration=1.5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{name}.mp4")],
                       check=True)
    messages = [t["message"] for t in trace if t["message"].strip()]
    return dict(level=level, seed=seed, horizon=HORIZONS[level], actions_shown=steps, outcome=status,
                native_return=round(total, 4), fps=FPS[level], file=f"{name}.mp4",
                last_messages=messages[-3:])


def main():
    manifest = dict(purpose="Presentation illustration of a uniformly random Restricted-action agent; not an evaluation",
                    seed_rule=f"seed = {SEED_BASE} + 10000 * level index, fixed before recording", recordings=[])
    for index, level in enumerate(LEVELS):
        result = record(index, level)
        manifest["recordings"].append(result)
        print(result, flush=True)
    (OUT / "random-recordings.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
# // END GENAI
