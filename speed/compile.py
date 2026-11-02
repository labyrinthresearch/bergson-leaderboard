"""Collect the sweep's estimates into one file of per-stage seconds per method, dataset size and number of queries.

Usage: compile.py <out.json> [<nodes_out.json>]

The second file gets the estimates of each node count of 4x GH200 nodes under est_gh200/.
"""
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parent

# Estimator stages and parts, by the stage of the method they belong to
PART_STAGES = {
    ("hessian", "batches0"): "Covariance fit",
    ("hessian", "batches1"): "Eigenvalue correction",
    ("run_magic", "train"): "Training replay",
    ("run_magic", "traced"): "Backward through training",
    ("run_magic", "recompute"): "Backward through training",
    ("run_magic", "state_copies"): "Trainer-state copies",
    ("run_magic", "checkpoints"): "Checkpoint I/O",
    ("run_magic", "checkpoint_saves"): "Checkpoint saves",
    ("run_magic", "query"): "Query gradients",
    ("run_magic", "after_query"): "Query gradients",
}
STAGES = {
    "build": "Query gradients",
    "hessian": "Eigendecomposition",
    "apply_hessian": "Query preconditioning",
    "astra": "ASTRA refinement",
    "lissa": "LiSSA iterations",
    "score": "Training gradients and scoring",
    "train": "Model training",
    "run_magic": "Startup and overhead",
    "parent": "Startup and overhead",
    "inverse_gram": "Gram inversion",
    "train_label_probs": "Training-set probabilities",
}
# Stages that belong to the shared training run, which every method's model needs
SHARED = {"source_adam": {"train"}}
# Stages repeated for an ensemble: TRAK trains eight members on half the set each
REPEATS = {"trak": {"train": 4}}


def stages(method: str, estimate: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for stage in estimate["stages"]:
        name = stage["name"]
        if name in SHARED.get(method, set()):
            continue
        times = REPEATS.get(method, {}).get(name, 1)
        for part, seconds in stage["seconds"].items():
            if part == "startup":
                label = "Startup and overhead"
            elif name == "run_magic" and part.startswith("query"):
                # One phase per size of query stream
                label = "Query gradients"
            else:
                label = PART_STAGES.get((name, part)) or STAGES.get(name, name.replace("_", " ").capitalize())
            out[label] = out.get(label, 0.0) + seconds * times
    # The command's own process: launching, relaunching and parent-side work
    spent = sum(s["total_seconds"] for s in estimate["stages"])
    out["Startup and overhead"] = out.get("Startup and overhead", 0.0) + max(estimate["total_seconds"] - spent, 0.0)
    return out


def collect(root: Path) -> dict:
    """Per-stage seconds of every estimate under root, by method."""
    data = {}
    for path in sorted(root.glob("*/*.json")):
        method = path.parent.name
        size, queries = map(int, path.stem.split("_"))
        estimate = json.load(open(path))
        peak = max((s["peak_bytes"] for s in estimate["stages"] if s["name"] not in SHARED.get(method, set())), default=0)
        data.setdefault(method, []).append(
            dict(dataset_size=size, queries=queries, stages=stages(method, estimate), peak_gib=peak / 2**30)
        )
    return data


if __name__ == "__main__":
    data = collect(LB / "est")
    json.dump(data, open(sys.argv[1], "w"), indent=1)
    print({m: len(v) for m, v in data.items()})
    if len(sys.argv) > 2:
        # Each node count's estimates on another node type, under est_gh200/n<count>
        nodes = {d.name[1:]: collect(d) for d in sorted((LB / "est_gh200").glob("n[0-9]*")) if d.name[1:].isdigit()}
        json.dump(nodes, open(sys.argv[2], "w"), indent=1)
        print({n: sum(len(v) for v in d.values()) for n, d in nodes.items()})
