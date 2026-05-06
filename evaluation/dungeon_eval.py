"""
Dungeon-generation algorithm comparison harness.

For each (algorithm, grid_size, num_rooms, seed) it generates a dungeon and
records both performance metrics (time, nodes explored) and quality metrics
(solvability, connectivity, path length, branching, room-type distribution).

Outputs:
  results/dungeon_runs.csv         — one row per run
  results/dungeon_summary.csv      — algorithm × config means
  results/dungeon_*.png            — comparison plots
"""

import csv
import io
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List

# Make project root importable when this file is run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison_generators import (
    BFSDungeonGenerator,
    DFSDungeonGenerator,
    GreedyDungeonGenerator,
)
from csp_generator import DungeonCSP
from evaluation.quality_metrics import metric_bundle

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Run config

CONFIGS = [
    # (grid_size, num_rooms)
    (5, 6),
    (5, 10),
    (8, 12),
    (8, 20),
    (10, 25),
    (12, 35),
]

ALGORITHMS = {
    "CSP":    lambda w, h, n, s: DungeonCSP(w, h, n, seed=s),
    "BFS":    lambda w, h, n, s: BFSDungeonGenerator(w, h, n, seed=s),
    "DFS":    lambda w, h, n, s: DFSDungeonGenerator(w, h, n, seed=s),
    "Greedy": lambda w, h, n, s: GreedyDungeonGenerator(w, h, n, seed=s),
}

NUM_SEEDS = 30


# Single-run wrapper

def run_one(algo: str, grid: int, n_rooms: int, seed: int) -> dict:
    gen = ALGORITHMS[algo](grid, grid, n_rooms, seed)
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buf):
            dungeon = gen.generate()
    except Exception as e:  # pragma: no cover, surfaces in CSV as failure row
        return {
            "algo": algo, "grid": grid, "n_rooms": n_rooms, "seed": seed,
            "time_s": time.perf_counter() - t0,
            "nodes_explored": getattr(gen, "nodes_explored", -1),
            "backtracks": getattr(gen, "backtrack_count", 0),
            "succeeded": 0, "error": str(e)[:80],
            **metric_bundle(None),
        }
    elapsed = time.perf_counter() - t0
    succeeded = dungeon is not None
    row = {
        "algo": algo, "grid": grid, "n_rooms": n_rooms, "seed": seed,
        "time_s": elapsed,
        "nodes_explored": getattr(gen, "nodes_explored", -1),
        "backtracks": getattr(gen, "backtrack_count", 0),
        "succeeded": int(succeeded),
        "error": "",
        **metric_bundle(dungeon),
    }
    return row


# Main sweep

def run_sweep() -> List[dict]:
    rows: List[dict] = []
    total = len(ALGORITHMS) * len(CONFIGS) * NUM_SEEDS
    done = 0
    for grid, n_rooms in CONFIGS:
        for algo in ALGORITHMS:
            for seed in range(NUM_SEEDS):
                rows.append(run_one(algo, grid, n_rooms, seed))
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{total} runs", flush=True)
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[dict]) -> List[dict]:
    """Aggregate by (algo, grid, n_rooms) → means + success rate."""
    from statistics import mean, stdev

    buckets: dict = {}
    for r in rows:
        key = (r["algo"], r["grid"], r["n_rooms"])
        buckets.setdefault(key, []).append(r)

    out = []
    for (algo, grid, nr), runs in sorted(buckets.items()):
        ok = [r for r in runs if r["succeeded"] == 1]
        def avg(field, source=runs):
            vals = [r[field] for r in source if r[field] != -1]
            return mean(vals) if vals else float("nan")
        def sd(field, source=runs):
            vals = [r[field] for r in source if r[field] != -1]
            return stdev(vals) if len(vals) > 1 else 0.0
        out.append({
            "algo": algo, "grid": grid, "n_rooms": nr,
            "n_runs": len(runs),
            "success_rate": len(ok) / len(runs),
            "solvable_rate": sum(r["solvable"] for r in runs) / len(runs),
            "mean_time_s": avg("time_s"),
            "sd_time_s": sd("time_s"),
            "mean_nodes": avg("nodes_explored"),
            "mean_backtracks": avg("backtracks"),
            "mean_rooms_built": avg("rooms_built"),
            "mean_branching": avg("branching", ok) if ok else 0.0,
            "mean_dead_end_ratio": avg("dead_end_ratio", ok) if ok else 0.0,
            "mean_start_boss_dist": avg("start_boss_dist", ok) if ok else 0.0,
        })
    return out


# Plots

def make_plots(rows: List[dict], summary: List[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = list(ALGORITHMS.keys())
    cfg_labels = [f"{g}x{g}/{n}r" for g, n in CONFIGS]

    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = list(range(len(CONFIGS)))
    for i, algo in enumerate(algos):
        ys = [next((s["success_rate"] for s in summary
                    if s["algo"] == algo and (s["grid"], s["n_rooms"]) == cfg), 0)
              for cfg in CONFIGS]
        ax.bar([xi + i * width for xi in x], ys, width=width, label=algo)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(cfg_labels, rotation=20)
    ax.set_ylabel("Success rate (valid + connected)")
    ax.set_title("Generator success rate by configuration")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dungeon_success_rate.png", dpi=130)
    plt.close(fig)

    # Solvability rate 
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, algo in enumerate(algos):
        ys = [next((s["solvable_rate"] for s in summary
                    if s["algo"] == algo and (s["grid"], s["n_rooms"]) == cfg), 0)
              for cfg in CONFIGS]
        ax.bar([xi + i * width for xi in x], ys, width=width, label=algo)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(cfg_labels, rotation=20)
    ax.set_ylabel("Solvable rate (start→key→boss reachable)")
    ax.set_title("Generator solvability rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dungeon_solvable_rate.png", dpi=130)
    plt.close(fig)

    # --- Time scaling --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in algos:
        ys = [next((s["mean_time_s"] for s in summary
                    if s["algo"] == algo and (s["grid"], s["n_rooms"]) == cfg), 0)
              for cfg in CONFIGS]
        ax.plot(cfg_labels, ys, marker="o", label=algo)
    ax.set_ylabel("Mean generation time (s)")
    ax.set_title("Generation time vs. configuration")
    ax.set_yscale("log")
    ax.legend()
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dungeon_time_scaling.png", dpi=130)
    plt.close(fig)

    # Quality: branching factor + dead end ratio
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for algo in algos:
        bs = [next((s["mean_branching"] for s in summary
                    if s["algo"] == algo and (s["grid"], s["n_rooms"]) == cfg), 0)
              for cfg in CONFIGS]
        des = [next((s["mean_dead_end_ratio"] for s in summary
                     if s["algo"] == algo and (s["grid"], s["n_rooms"]) == cfg), 0)
               for cfg in CONFIGS]
        axes[0].plot(cfg_labels, bs, marker="o", label=algo)
        axes[1].plot(cfg_labels, des, marker="o", label=algo)
    axes[0].set_title("Mean branching factor")
    axes[0].set_ylabel("avg connections per room")
    axes[1].set_title("Dead-end ratio")
    axes[1].set_ylabel("share of rooms with 1 connection")
    for a in axes:
        a.legend()
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dungeon_quality.png", dpi=130)
    plt.close(fig)

# Entry point

def main() -> None:
    print(f"Running dungeon eval: {len(ALGORITHMS)} algos × "
          f"{len(CONFIGS)} configs × {NUM_SEEDS} seeds "
          f"= {len(ALGORITHMS) * len(CONFIGS) * NUM_SEEDS} runs")
    rows = run_sweep()
    summary = summarize(rows)
    write_csv(rows, RESULTS_DIR / "dungeon_runs.csv")
    write_csv(summary, RESULTS_DIR / "dungeon_summary.csv")
    make_plots(rows, summary)
    print(f"\nWrote {RESULTS_DIR}/dungeon_runs.csv "
          f"({len(rows)} rows) + summary + plots")


if __name__ == "__main__":
    main()
