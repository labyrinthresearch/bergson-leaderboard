"""The largest synthetic dataset sizes, built from Arrow arrays directly."""
import os

import numpy as np
import pyarrow as pa
from datasets import Dataset

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = np.random.default_rng(1)
for n in (294912, 1179648):
    values = pa.array(rng.integers(0, 50257, size=n * 511, dtype=np.int32))
    offsets = pa.array(np.arange(0, (n + 1) * 511, 511, dtype=np.int32))
    table = pa.table({"input_ids": pa.ListArray.from_arrays(offsets, values)})
    Dataset(table).save_to_disk(f"{root}/train_{n}")
    print(n, flush=True)
