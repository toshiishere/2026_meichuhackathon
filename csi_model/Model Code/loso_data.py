"""LOSO (leave-one-subject-out) data loading for ESP-Fi HAR, matching dataset.py's preprocessing."""
import glob
import os
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

ACTIVITY_MAP = {
    1: "run", 2: "fall", 3: "walk", 4: "turn",
    5: "jump", 6: "squat", 7: "arm_wave",
}
ACTIVITIES = ["arm_wave", "fall", "jump", "run", "squat", "turn", "walk"]
ENV_DIRS = {
    1: "EnvironmentNo.1(corridor)",
    2: "EnvironmentNo.2（Office）",
    3: "EnvironmentNo.3（boardrooms）",
    4: "EnvironmentNo.4（laboratory）",
}

FNAME_RE = re.compile(r"(\d+)-(\d+)-(\d+)-(\d+)\.mat$")


def load_environment(root_dir, env_id):
    """Load all samples for one environment. Returns dict: participant -> list of (x, label)."""
    env_dir = os.path.join(root_dir, ENV_DIRS[env_id], "mat")
    by_participant = {}
    for path in glob.glob(os.path.join(env_dir, "*.mat")):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        scenario, participant, activity, trial = map(int, m.groups())
        mat = sio.loadmat(path)
        x = mat["CSIamp"]
        if x.shape == (950, 52):
            pass
        elif x.shape == (52, 950):
            x = x.T
        else:
            raise ValueError(f"Unexpected shape {x.shape} in {path}")
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = x.reshape(1, 950, 52).astype(np.float32)
        label = ACTIVITIES.index(ACTIVITY_MAP[activity])
        by_participant.setdefault(participant, []).append((x, label))
    return by_participant


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def loso_fold(by_participant, held_out):
    train_items, test_items = [], []
    for p, items in by_participant.items():
        if p == held_out:
            test_items.extend(items)
        else:
            train_items.extend(items)
    return ListDataset(train_items), ListDataset(test_items)
