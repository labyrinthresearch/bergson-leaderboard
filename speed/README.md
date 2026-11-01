# Speed estimates

`estimates.json` holds [bergson-estimate](https://github.com/labyrinthresearch/bergson-estimate)'s predictions for each leaderboard method's recipe on one 8x A40 node (48 CPU cores, `--rates a40_48core`, `NCCL_P2P_LEVEL=SYS`), per stage, at a ladder of dataset sizes and numbers of queries. `speed_plots.py` at the repository root fits each method's cost model to them and draws the figures.

How they were made:

1. `tree/` is bergson `main` at `320cedb9` with bergson-estimate's package copied in as `bergson/estimate` (bergson-estimate `e7e152f`; LiSSA's and EKFAC + ASTRA's estimates were made with earlier commits back to `9148f33f`, whose later changes only cut the replay's memory and time or price MAGIC), and `tree/examples/gpt2_wikitext_leaderboard` holds the leaderboard recipes.
2. `make_synth.py` and `make_synth_big.py` write synthetic datasets of 511-token documents, the length of the leaderboard's WikiText chunks: 1, 10, 50, 200 and 1,000 queries, and 1,152 to 1,179,648 training documents in steps of 4.
3. `sweep.py` writes each method's recipe with its training set and queries swapped for synthetic ones and runs `python -m bergson.estimate` on it, at every dataset size with 50 queries, at every number of queries with 4,608 documents, and at four points off those lines, which separate per-pair costs from per-document and per-query ones:

   ```bash
   CUDA_VISIBLE_DEVICES= PYTHON=/path/to/python WORKERS=5 python sweep.py
   ```

   With no GPU visible, the estimator replays on stand-ins for the node's GPUs. `sweep.py` sets every recipe's `nproc_per_node` to 8, so each method gets the whole node.
4. `compile.py estimates.json` collects the estimates per stage.

Per method:

- MAGIC runs `1_magic.yaml` without its validation bank (`skip_validation: true`).
- KFAC (projection 64) runs bergson `main` at `15caf720` in `treemain/`, which includes [bergson #543](https://github.com/EleutherAI/bergson/pull/543): it inverts the Kronecker factors in the projected space.
- Unrolled Gauss-Newton runs `unrolled_gauss_newton.yaml` from [bergson #546](https://github.com/EleutherAI/bergson/pull/546), merged onto `main` in `tree546/`.
- SOURCE runs `2_interval.yaml`, saving six evenly spaced checkpoints, then the plain step of `source.yaml`; the training run is left out of SOURCE's time. The Adam-preconditioned step also reads each checkpoint's optimizer state, which the stand-in checkpoints of a replayed training don't hold.
- TrackStar has no 1-query point: bergson can't spread one query over 8 GPUs.
- TRAK counts `2_interval.yaml`'s training (without interval saves) four times, for eight members trained on half the set each, then `trak.yaml` over eight checkpoints of GPT-2's shape.
- TrackStar (Adam) reads `make_optimizer.py`'s stand-in for the checkpoint's `optimizer.pt`, with its shapes and parameter names.
- EKFAC + ASTRA is the first step of `ekfac_astra.yaml`, TrackStar the projection-64 step of `trackstar.yaml` and `trackstar_adam.yaml`, and the gradient baselines the steps of `gradient_cosine.yaml` and `gradient_dot.yaml`.
