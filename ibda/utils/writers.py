import json
from pathlib import Path

import numpy as np


def save_as_json(data, savedir, fname, indent = None):
    Path(savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(savedir, fname), "w") as f:
        json.dump(data, f, indent=indent)


def save_as_np(data, savedir, fname):
    Path(savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(savedir, fname), "wb") as f:
        np.save(f, data)
