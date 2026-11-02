"""Estimate every leaderboard method's run at a ladder of dataset sizes and numbers of queries.

Each method's pipeline is the leaderboard recipe with its training set and
queries swapped for synthetic 511-token documents (see make_synth.py). Writes
est/<method>/<dataset size>_<queries>.json and prints each command it runs.

Usage: sweep.py [method ...]
"""
import copy
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

LB = Path(__file__).resolve().parent
EXAMPLES = LB / "tree/examples/gpt2_wikitext_leaderboard"
PYTHON = os.environ.get("PYTHON", os.path.expanduser("~/code/bergson-cost/.venv/bin/python"))
CHECKPOINT = "runs/compare_wikitext/interval/exported/checkpoint-72"
SIZES = [1152, 4608, 18432, 73728, 294912, 1179648]
QUERIES = [1, 10, 50, 200, 1000]
REFERENCE = (4608, 50)
TIMEOUT = None
"""Seconds one estimate may take: the slowest points, such as MAGIC with many
queries, run for hours, so none is cut short."""
# Off the two sweeps, to separate per-pair costs from per-document and
# per-query ones, and to test the fitted cost model
CORNERS = [(1152, 1000), (73728, 1), (73728, 1000), (294912, 200)]


def load(name: str) -> dict:
    return yaml.safe_load(open(EXAMPLES / name))


# The node type to price with, and how many; every recipe gets all their GPUs, so
# methods compare on the same hardware
RATES = os.environ.get("RATES", "a40_48core")
NODES = int(os.environ.get("NODES", 1))
GPUS = int(os.environ.get("GPUS", 8))
# Where the estimates go, under LB
EST = os.environ.get("EST", "est")
# The two dataset sizes another node type is estimated at, with 50 queries, to
# price the cost model's terms on it
ANCHORS = [4608, 73728]


def swap(node, size: int, queries: int):
    """Point training splits at the synthetic set of ``size`` documents and
    query splits at ``queries`` synthetic queries."""
    if isinstance(node, dict):
        if "dataset" in node and "split" in node:
            split = str(node.pop("split"))
            node["dataset"] = str(
                LB / "data" / (f"train_{size}" if split.startswith("train") else f"queries_{queries}")
            )
        for k, v in node.items():
            if k == "nproc_per_node":
                node[k] = GPUS
            elif k == "nnode":
                node[k] = NODES
            elif k == "model" and v == CHECKPOINT:
                node[k] = "gpt2"
            elif k == "optimizer_state" and v == f"{CHECKPOINT}/optimizer.pt":
                # make_optimizer.py's stand-in, with the checkpoint's shapes
                node[k] = str(LB / "data" / "optimizer.pt")
            else:
                swap(v, size, queries)
    elif isinstance(node, list):
        for v in node:
            swap(v, size, queries)
    return node


def steps(name: str, *keep: int) -> list[dict]:
    config = load(name)
    return [config["steps"][i] for i in keep] if keep else config["steps"]


def interval_training(size: int, saves: bool) -> dict:
    """The leaderboard model's training run, saving six evenly spaced checkpoints
    when ``saves``."""
    (step,) = steps("2_interval.yaml")
    train = step["train"]
    total = size * train["num_epochs"] // train["batch_size"]
    if saves:
        train["save_interval"] = max(total // 6, 1)
    else:
        for key in ("save_mode", "save_interval", "save_optimizer_state"):
            train.pop(key, None)
    return step


def source(size: int) -> list[dict]:
    train = interval_training(size, saves=True)
    every = train["train"]["save_interval"]
    # The plain step: the Adam-preconditioned one also reads each checkpoint's
    # optimizer state, which a replayed training's stand-in checkpoints lack
    (step,) = steps("source.yaml", 0)
    step = copy.deepcopy(step)
    cfg = step["approxunrolling"]
    run = train["train"]["run_path"]
    cfg["approx_unrolling_cfg"]["checkpoints"] = [
        f"{run}/checkpoints/step_{every * k}.ckpt" for k in range(1, 7)
    ]
    return [train, step]


def trak(size: int) -> list[dict]:
    (step,) = steps("trak.yaml")
    step = copy.deepcopy(step)
    members = step["trak"]["trak_cfg"]["checkpoints"]
    step["trak"]["trak_cfg"]["checkpoints"] = ["gpt2"] * len(members)
    return [interval_training(size, saves=False), step]


def magic(size: int) -> list[dict]:
    (step,) = steps("1_magic.yaml")
    step["magic"]["skip_validation"] = True
    return [step]


def unrolled_gauss_newton(size: int) -> list[dict]:
    """EleutherAI/bergson#546's recipe, from bergson main with the PR merged."""
    path = LB / "tree546/examples/gpt2_wikitext_leaderboard/unrolled_gauss_newton.yaml"
    return yaml.safe_load(open(path))["steps"]


# Methods replayed with bergson other than tree/'s
# kfac_p64 runs main from #543 on, which inverts the Kronecker factors in the projected space
TREES = {"unrolled_gauss_newton": "tree546", "kfac_p64": "treemain"}

METHODS = {
    "ekfac": lambda n: steps("ekfac.yaml"),
    "kfac": lambda n: steps("kfac.yaml"),
    "kfac_p64": lambda n: steps("kfac_p64.yaml"),
    "ekfac_cg": lambda n: steps("ekfac_astra_cg.yaml"),
    "ekfac_astra": lambda n: steps("ekfac_astra.yaml", 0),
    "shampoo": lambda n: steps("shampoo.yaml"),
    "shampoo_astra": lambda n: steps("shampoo_astra.yaml"),
    "source_adam": source,
    "lissa": lambda n: steps("lissa.yaml"),
    "trackstar_p64": lambda n: steps("trackstar.yaml", 2),
    "trackstar_adam_p64": lambda n: steps("trackstar_adam.yaml", 2),
    "trak": trak,
    "gradient_cosine": lambda n: steps("gradient_cosine.yaml", 0, 1),
    "gradient_cosine_projected": lambda n: steps("gradient_cosine.yaml", 2, 3),
    "gradient_dot": lambda n: steps("gradient_dot.yaml"),
    "magic": magic,
    "unrolled_gauss_newton": unrolled_gauss_newton,
}


def run(method: str, size: int, queries: int) -> str:
    out = LB / EST / method
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{size}_{queries}"
    if Path(f"{stem}.json").exists():
        return f"{method} {size} {queries} done before"
    config = {"run_path": f"runs/sweep/{method}", "steps": METHODS[method](size)}
    yaml.safe_dump(swap(copy.deepcopy(config), size, queries), open(f"{stem}.yaml", "w"), sort_keys=False)
    work = LB / "work" / EST / f"{method}_{size}_{queries}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-m", "bergson.estimate", "--rates", RATES, "--nodes", str(NODES), "--json", f"{stem}.json", f"{stem}.yaml"]
    tree = str(LB / TREES.get(method, "tree"))
    print(" ".join(["PYTHONPATH=" + tree, "NCCL_P2P_LEVEL=SYS", *cmd]), flush=True)
    # A port of its own: estimates side by side start process groups in their own process
    port = 20000 + zlib.crc32(str(stem).encode()) % 20000
    env = {**os.environ, "PYTHONPATH": tree, "NCCL_P2P_LEVEL": "SYS", "MASTER_PORT": str(port)}
    with open(f"{stem}.log", "w") as log:
        try:
            code = subprocess.run(
                cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT
            ).returncode
        except subprocess.TimeoutExpired:
            return f"{method} {size} {queries} timed out after {TIMEOUT} s"
    return f"{method} {size} {queries} exit {code}"


if __name__ == "__main__":
    methods = sys.argv[1:] or list(METHODS)
    if os.environ.get("POINTS") == "anchors":
        jobs = [(m, n, REFERENCE[1]) for m in methods for n in ANCHORS]
    else:
        jobs = [(m, n, REFERENCE[1]) for m in methods for n in SIZES]
        jobs += [(m, REFERENCE[0], q) for m in methods for q in QUERIES if q != REFERENCE[1]]
        jobs += [(m, n, q) for m in methods for n, q in CORNERS]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as pool:
        for line in pool.map(lambda job: run(*job), jobs):
            print(line, flush=True)
