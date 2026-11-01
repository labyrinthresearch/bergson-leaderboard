"""Synthetic pretokenized datasets of 511-token documents, the length of the leaderboard's WikiText chunks, for sweeping dataset size and number of queries."""
import os

import numpy as np
from datasets import Dataset

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = np.random.default_rng(0)


def make(n: int, path: str):
    ids = rng.integers(0, 50257, size=(n, 511), dtype=np.int32)
    Dataset.from_dict({"input_ids": ids.tolist()}).save_to_disk(path)
    print(path, n, flush=True)


for q in (1, 10, 50, 200, 1000):
    make(q, f"{root}/queries_{q}")
for n in (1152, 4608, 18432, 73728):
    make(n, f"{root}/train_{n}")
