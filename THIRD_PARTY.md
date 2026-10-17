# Sources and credits

This project runs the original tasks; it does not bundle the game engine.

- **MiniHack**: Facebook, Inc. and its affiliates, Apache License 2.0. Pinned source: [09eaa7ee](https://github.com/facebookresearch/minihack/tree/09eaa7ee45c567850668d3743d7fe1853917d98f). The task definitions, observations and tile renderer come from MiniHack. License copy: [licenses/MiniHack-Apache-2.0.txt](licenses/MiniHack-Apache-2.0.txt).
- **NetHack Learning Environment and NetHack**: [NLE v1.1.0 source](https://github.com/facebookresearch/nle/tree/v1.1.0), distributed under the NetHack General Public License. License copy: [licenses/NetHack.txt](licenses/NetHack.txt). Original NetHack copyright notices remain in that source.
- **Game artwork** in `docs/assets/` and the PDF is captured from native MiniHack game rendering. Icons are crops of that rendering. Recordings are accelerated and cropped around the player; overview maps reveal terrain in separate illustration-only instances. Original tile data and renderer: [MiniHack tiles source](https://github.com/facebookresearch/minihack/tree/09eaa7ee45c567850668d3743d7fe1853917d98f/minihack/tiles). These game graphics retain their upstream licenses; the project's MIT license does not relicense them.

Other dependencies (PyTorch, NumPy, Gymnasium, Pillow and build tools) are installed separately and retain their respective licenses and notices.

Presentation text/code and the agent implementation were prepared with AI assistance. The [MiniHack task documentation](https://minihack.readthedocs.io/en/stable/envs/skills/wod.html), pinned [task definitions](https://github.com/facebookresearch/minihack/blob/09eaa7ee45c567850668d3743d7fe1853917d98f/minihack/envs/skills_wod.py) and [NLE time-penalty implementation](https://github.com/facebookresearch/nle/blob/v1.1.0/nle/env/tasks.py) are the primary sources for the presentation's environment claims.
