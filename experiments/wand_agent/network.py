"""Small MLP; portable checkpoints contain arrays only (no pickled classes)."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .features import DIM
from .native import KEYS


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(KEYS)))

    def forward(self, observation):
        return self.layers(observation)

    @torch.no_grad()
    def action(self, observation):
        return int(self(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    temporary.replace(path)


def load(path):
    model = QNetwork()
    with np.load(path, allow_pickle=False) as data:
        state = {key: torch.from_numpy(data[key].copy()) for key in data.files}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint has nonfinite weights')
    model.load_state_dict(state, strict=True)
    return model.eval()
