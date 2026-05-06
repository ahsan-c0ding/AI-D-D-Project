"""
NPC decision-tree evaluation.

Three things we measure:

1. Behavioral coverage
   Sweep a grid of game states (player HP, NPC HP, gold, has-items, attacked-flag)
   and record which leaf actions get taken for each NPC type. Detect dead leaves
   and report action distribution.

2. Plausibility vs. random baseline
   Score each (state, action) pair against a hand-written rubric (e.g. low-HP
   enemies should flee when player healthy). Compare the decision tree's
   plausibility score to a random-policy baseline.

3. Decision latency
   Time per call.

Outputs:
  results/npc_coverage.csv
  results/npc_plausibility.csv
  results/npc_action_distribution.png
  results/npc_plausibility.png
"""

import csv
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import NPC, NPCType, Player
from npc_decision_tree import NPCAction, NPCBehaviorManager

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# State sweep

PLAYER_HP_GRID = [10, 30, 50, 70, 100]
NPC_HP_GRID = [5, 20, 40, 80]
GOLD_GRID = [0, 30, 100, 500]
INVENTORY_GRID = [0, 1, 3]
ATTACKED_GRID = [False, True]
NPC_FIRST_MET = [True, False]


def make_player(hp: int, gold: int, n_items: int) -> Player:
    return Player(
        name="Hero", hp=hp, max_hp=100,
        attack=15, defense=8, position=(0, 0),
        gold=gold,
        inventory=[None] * n_items,  # decision tree only checks len()
    )


def make_npc(npc_type: NPCType, hp: int, name: str = None) -> NPC:
    default_names = {
        NPCType.ENEMY: "Goblin",
        NPCType.MERCHANT: "Merchant",
        NPCType.FRIENDLY: "Elder",
        NPCType.NEUTRAL: "Thief Stranger",  # 'thief' triggers steal branch
        NPCType.BOSS: "Dragon",
    }
    return NPC(
        name=name or default_names[npc_type],
        npc_type=npc_type,
        hp=hp, attack=10, defense=5,
        dialogue=["..."],
    )


def sweep_states():
    for npc_type in NPCType:
        for php, nhp, gold, inv, attacked, met in product(
            PLAYER_HP_GRID, NPC_HP_GRID, GOLD_GRID, INVENTORY_GRID,
            ATTACKED_GRID, NPC_FIRST_MET,
        ):
            yield {
                "npc_type": npc_type,
                "player_hp": php,
                "npc_hp": nhp,
                "gold": gold,
                "inventory": inv,
                "player_attacked": attacked,
                "npc_met": met,
            }


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------

def evaluate_coverage() -> List[dict]:
    mgr = NPCBehaviorManager()
    rows: List[dict] = []
    latencies: List[float] = []
    for st in sweep_states():
        player = make_player(st["player_hp"], st["gold"], st["inventory"])
        npc = make_npc(st["npc_type"], st["npc_hp"])
        gs = {"player_attacked": st["player_attacked"],
              "npc_met": st["npc_met"], "turn_count": 1}
        t0 = time.perf_counter_ns()
        action = mgr.get_npc_action(npc, player, gs)
        latencies.append(time.perf_counter_ns() - t0)
        rows.append({**st,
                     "npc_type": st["npc_type"].value,
                     "action": action.value})
    print(f"  swept {len(rows)} states, "
          f"mean decision latency = {sum(latencies)/len(latencies):.1f} ns")
    return rows


# 2. Plausibility vs random baseline

def plausibility_score(npc_type: NPCType, state: dict, action: str) -> float:
    """
    Hand-written rubric — each (npc_type, state) condition scores an action
    1.0 if it's clearly appropriate, 0.0 if clearly wrong, 0.5 if neutral.
    Designed so a sensible policy beats a random one substantially.
    """
    p_hp, n_hp, gold = state["player_hp"], state["npc_hp"], state["gold"]
    inv = state["inventory"]
    attacked = state["player_attacked"]

    if npc_type == NPCType.ENEMY:
        if n_hp <= 10 and p_hp > 50:
            return 1.0 if action in ("flee", "surrender") else 0.0
        if p_hp < 25:
            return 1.0 if action == "attack" else 0.2
        if n_hp < 25:
            return 1.0 if action in ("defend", "flee") else 0.4
        return 1.0 if action == "attack" else 0.3

    if npc_type == NPCType.BOSS:
        if n_hp < 30:
            return 1.0 if action == "attack" else 0.3   # desperate
        if p_hp > 70:
            return 1.0 if action == "defend" else 0.4
        return 1.0 if action == "attack" else 0.3

    if npc_type == NPCType.MERCHANT:
        if gold > 50 or inv > 0:
            return 1.0 if action == "trade" else 0.2
        return 1.0 if action == "talk" else 0.4

    if npc_type == NPCType.FRIENDLY:
        if p_hp < 50:
            return 1.0 if action == "help" else 0.2
        return 1.0 if action in ("talk", "idle") else 0.3

    if npc_type == NPCType.NEUTRAL:
        if attacked:
            return 1.0 if action == "attack" else 0.0
        return 1.0 if action in ("talk", "steal", "idle") else 0.3

    return 0.5


def evaluate_plausibility(coverage_rows: List[dict]) -> List[dict]:
    actions_universe = [a.value for a in NPCAction]
    summary: Dict[str, dict] = {}

    for row in coverage_rows:
        npc_type = NPCType(row["npc_type"])
        tree_score = plausibility_score(npc_type, row, row["action"])
        random_action = random.choice(actions_universe)
        random_score = plausibility_score(npc_type, row, random_action)
        bucket = summary.setdefault(npc_type.value,
                                    {"npc_type": npc_type.value,
                                     "n": 0,
                                     "tree_score": 0.0,
                                     "random_score": 0.0})
        bucket["n"] += 1
        bucket["tree_score"] += tree_score
        bucket["random_score"] += random_score

    out = []
    for v in summary.values():
        out.append({
            "npc_type": v["npc_type"],
            "n_states": v["n"],
            "mean_tree_plausibility": v["tree_score"] / v["n"],
            "mean_random_plausibility": v["random_score"] / v["n"],
            "advantage": (v["tree_score"] - v["random_score"]) / v["n"],
        })
    return out


# CSV + plots

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plots(coverage: List[dict], plausibility: List[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # action distribution per NPC type
    by_type: Dict[str, Dict[str, int]] = {}
    for row in coverage:
        by_type.setdefault(row["npc_type"], {}).setdefault(row["action"], 0)
        by_type[row["npc_type"]][row["action"]] += 1

    npc_types = sorted(by_type.keys())
    all_actions = sorted({a for d in by_type.values() for a in d})
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = [0] * len(npc_types)
    for action in all_actions:
        vals = [by_type[t].get(action, 0) for t in npc_types]
        ax.bar(npc_types, vals, bottom=bottom, label=action)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("# of swept states producing this action")
    ax.set_title("Decision-tree action distribution by NPC type")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "npc_action_distribution.png", dpi=130)
    plt.close(fig)

    # plausibility comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    types = [r["npc_type"] for r in plausibility]
    tree = [r["mean_tree_plausibility"] for r in plausibility]
    rand = [r["mean_random_plausibility"] for r in plausibility]
    x = list(range(len(types)))
    ax.bar([i - 0.2 for i in x], tree, width=0.4, label="Decision tree")
    ax.bar([i + 0.2 for i in x], rand, width=0.4, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Mean plausibility (0–1, higher = better)")
    ax.set_title("Decision tree vs. random policy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "npc_plausibility.png", dpi=130)
    plt.close(fig)


# Entry point

def main() -> None:
    random.seed(0)
    print("NPC decision-tree evaluation")
    coverage = evaluate_coverage()
    plausibility = evaluate_plausibility(coverage)
    write_csv(coverage, RESULTS_DIR / "npc_coverage.csv")
    write_csv(plausibility, RESULTS_DIR / "npc_plausibility.csv")
    make_plots(coverage, plausibility)
    print("\nPlausibility summary:")
    for r in plausibility:
        print(f"  {r['npc_type']:<10}  tree={r['mean_tree_plausibility']:.2f}"
              f"  random={r['mean_random_plausibility']:.2f}"
              f"  advantage={r['advantage']:+.2f}")


if __name__ == "__main__":
    main()
