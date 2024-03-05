from pathlib import Path

import numpy as np

import json

def save_as_json(data, savedir, fname):
	Path(savedir).mkdir(parents=True, exist_ok=True)
	with open(Path(savedir, fname), "w") as f:
		json.dump(data, f)

def save_as_np(data, savedir, fname):
	Path(savedir).mkdir(parents=True, exist_ok=True)
	with open(Path(savedir, fname), "wb") as f:
		np.save(f, data)
