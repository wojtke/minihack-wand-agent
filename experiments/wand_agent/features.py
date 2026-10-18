"""Observation-only features shared by every difficulty; no map or level input."""
import numpy as np
from nle import nethack
from .native import KEYS, text

MONSTER = nethack.glyph_is_monster(np.arange(nethack.MAX_GLYPH + 1))
OBJECT = nethack.glyph_is_object(np.arange(nethack.MAX_GLYPH + 1))
DIM = 463


class PublicMemory:
    """Generic visible entity/terrain memory; no action choice or path search."""
    def __init__(self):
        self.visits = np.zeros((21, 79), dtype=np.int32)
        self.objects = set()
        self.exits = set()
        self.previous_items = set()
        self.previous_action = None
        self.previous_mode = None
        self.command = None

    @staticmethod
    def mode(raw):
        message=text(raw['message'])
        return (1 if "What do you want to zap?" in message else
                2 if "In what direction?" in message else
                3 if "What do you want to" in message else
                4 if "[yn" in message else 0)

    def encode(self, raw, elapsed=0):
        x, y = map(int, raw["blstats"][:2])
        chars, glyphs = raw["chars"], raw["glyphs"]
        monsters, objects = MONSTER[glyphs].copy(), OBJECT[glyphs].copy()
        monsters[y, x] = False
        # A visible object is remembered when the player stands over its glyph.
        self.objects.update((int(xx), int(yy)) for yy, xx in zip(*np.where(objects)))
        self.exits.update((int(xx),int(yy)) for yy,xx in zip(*np.where(chars==ord('>'))))
        items = set(map(int, raw["inv_letters"][raw["inv_letters"] != 0]))
        if items - self.previous_items:
            self.objects.discard((x, y))
        self.previous_items = items
        self.objects = {(xx, yy) for xx, yy in self.objects
                        if objects[yy, xx] or (xx == x and yy == y) or chars[yy, xx] == 32}
        self.visits[y, x] += 1
        solid = np.isin(chars, [32, 124, 45])
        channels = np.stack([solid, chars != 32, monsters, objects,
                             np.minimum(self.visits, 8)/8.]).astype(np.float32)
        padded = np.zeros((5, 29, 87), dtype=np.float32)
        padded[0] = 1.  # Outside the screen is indistinguishable from public unknown stone.
        padded[:, 4:-4, 4:-4] = channels
        local = padded[:, y:y+9, x:x+9].reshape(-1)
        def relative(points, count):
            ordered = sorted(points, key=lambda p: ((p[0]-x)**2+(p[1]-y)**2, p[1]-y, p[0]-x))[:count]
            values = [[1., np.clip(xx-x, -20, 20)/20., np.clip(yy-y, -20, 20)/20.]
                      for xx, yy in ordered]
            values += [[0., 0., 0.]] * (count-len(values))
            return np.asarray(values, np.float32).reshape(-1)
        enemy = relative([(int(xx), int(yy)) for yy, xx in zip(*np.where(monsters))], 2)
        item_positions = relative(self.objects, 3)
        inventory = np.array([ord(key) in items for key in KEYS], dtype=np.float32)
        mode = self.mode(raw)
        modes = np.eye(5, dtype=np.float32)[mode]
        previous = np.zeros(11, dtype=np.float32)
        if self.previous_action is not None:
            previous[self.previous_action] = 1.
        hp, maximum = map(float, raw["blstats"][[10, 11]])
        other = np.array([hp/max(1., maximum), min(elapsed/1000., 1.),
                          min(len(items)/10., 1.), self.command == "z", self.command == "f"], np.float32)
        old_mode=np.zeros(5,np.float32)
        if self.previous_mode is not None:old_mode[self.previous_mode]=1.
        encoded = np.concatenate([local, enemy, item_positions, inventory, modes, previous, other,
                                  relative(self.exits,2),old_mode])
        assert encoded.shape == (DIM,) and np.isfinite(encoded).all()
        return encoded
