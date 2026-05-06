"""
Bayesian combat-system evaluation.

1. Calibration
   For a sweep of (attacker_attack, defender_defense), simulate N attacks and
   compare the empirical hit rate to the model's predicted probability.
   A well-calibrated model should give empirical ≈ predicted.

2. Difficulty / win-rate matrix
   Run full combats for a grid of player vs. enemy stat configs, report
   player win rate. Used to check that the difficulty curve is smooth.

3. Combat duration & variance vs. random baseline
   Compare the Bayesian combat duration distribution against a 50/50 coin-flip
   baseline. Skill-based combat should produce *lower* variance and have
   duration correlated with the stat gap.

Outputs:
  results/combat_calibration.csv
  results/combat_winrate_matrix.csv
  results/combat_calibration.png
  results/combat_winrate_matrix.png
  results/combat_duration_dist.png
"""

import csv
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bayesian_combat import (
    BayesianCombatSystem, CombatOutcome, DiceRoller,
)
from models import NPC, NPCType, Player

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


N_TRIALS_CALIBRATION = 5000
ATTACK_GRID = [5, 10, 15, 20, 25, 30]
DEFENSE_GRID = [0, 5, 10, 15, 20, 25]


# 1. Calibration sweep

def evaluate_calibration() -> List[dict]:
    sys_ = BayesianCombatSystem()
    rows = []
    for atk in ATTACK_GRID:
        for d in DEFENSE_GRID:
            attacker = Player(name="A", hp=100, max_hp=100,
                              attack=atk, defense=10, position=(0, 0))
            defender = NPC(name="D", npc_type=NPCType.ENEMY,
                           hp=10**6, attack=10, defense=d,
                           dialogue=[])
            predicted = sys_.calculate_hit_probability(atk, d)
            hits = 0
            for _ in range(N_TRIALS_CALIBRATION):
                outcome, _ = sys_.resolve_attack(attacker, defender)
                if outcome in (CombatOutcome.HIT, CombatOutcome.CRITICAL_HIT):
                    hits += 1
            empirical = hits / N_TRIALS_CALIBRATION
            rows.append({
                "attack": atk, "defense": d,
                "predicted": predicted,
                "empirical": empirical,
                "abs_error": abs(predicted - empirical),
                "n_trials": N_TRIALS_CALIBRATION,
            })
    return rows


# 2. Win-rate matrix (full combats)

def simulate_combat(player_atk: int, player_def: int,
                    enemy_atk: int, enemy_def: int,
                    player_hp: int = 100, enemy_hp: int = 60,
                    max_rounds: int = 100) -> tuple:
    """Returns (player_won, rounds_taken)."""
    sys_ = BayesianCombatSystem()
    p = Player("P", hp=player_hp, max_hp=player_hp,
               attack=player_atk, defense=player_def, position=(0, 0))
    e = NPC("E", NPCType.ENEMY, hp=enemy_hp,
            attack=enemy_atk, defense=enemy_def, dialogue=[])
    rounds = 0
    while p.hp > 0 and e.hp > 0 and rounds < max_rounds:
        sys_.simulate_combat_round(p, e)
        rounds += 1
    return (p.hp > 0 and e.hp <= 0, rounds)


def evaluate_winrate_matrix(n_trials: int = 200) -> List[dict]:
    rows = []
    enemy_atk_grid = [5, 10, 15, 20]
    enemy_def_grid = [0, 5, 10, 15]
    for ea in enemy_atk_grid:
        for ed in enemy_def_grid:
            wins = 0
            durations = []
            for _ in range(n_trials):
                won, r = simulate_combat(15, 10, ea, ed)
                wins += int(won)
                durations.append(r)
            rows.append({
                "enemy_attack": ea,
                "enemy_defense": ed,
                "player_win_rate": wins / n_trials,
                "mean_rounds": mean(durations),
                "sd_rounds": pstdev(durations),
                "n_trials": n_trials,
            })
    return rows


# 3. Bayesian vs random-coin baseline

def simulate_random_combat(player_hp: int = 100,
                           enemy_hp: int = 60,
                           max_rounds: int = 100) -> tuple:
    """50/50 hit chance per side; damage is plain d6+constant."""
    p_hp, e_hp = player_hp, enemy_hp
    rounds = 0
    while p_hp > 0 and e_hp > 0 and rounds < max_rounds:
        if random.random() < 0.5:
            e_hp -= DiceRoller.d6() + 5
        if e_hp > 0 and random.random() < 0.5:
            p_hp -= DiceRoller.d6() + 5
        rounds += 1
    return (p_hp > 0 and e_hp <= 0, rounds)


def evaluate_duration_distribution(n_trials: int = 1000) -> dict:
    bayes_durations = []
    rand_durations = []
    for _ in range(n_trials):
        _, r1 = simulate_combat(15, 10, 10, 5)
        _, r2 = simulate_random_combat()
        bayes_durations.append(r1)
        rand_durations.append(r2)
    return {
        "bayes": bayes_durations,
        "random": rand_durations,
        "summary": {
            "bayes_mean": mean(bayes_durations),
            "bayes_sd": pstdev(bayes_durations),
            "random_mean": mean(rand_durations),
            "random_sd": pstdev(rand_durations),
        },
    }


# CSV + plots

def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plots(calib, winrate, dur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -- calibration scatter (predicted vs empirical) --
    fig, ax = plt.subplots(figsize=(7, 7))
    preds = [r["predicted"] for r in calib]
    emps = [r["empirical"] for r in calib]
    ax.scatter(preds, emps, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("Predicted hit probability")
    ax.set_ylabel(f"Empirical hit rate ({N_TRIALS_CALIBRATION} trials)")
    ax.set_title("Combat hit-probability calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "combat_calibration.png", dpi=130)
    plt.close(fig)

    # -- win-rate heatmap --
    enemy_atk = sorted({r["enemy_attack"] for r in winrate})
    enemy_def = sorted({r["enemy_defense"] for r in winrate})
    grid = [[0.0] * len(enemy_def) for _ in enemy_atk]
    for r in winrate:
        i = enemy_atk.index(r["enemy_attack"])
        j = enemy_def.index(r["enemy_defense"])
        grid[i][j] = r["player_win_rate"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(enemy_def)))
    ax.set_xticklabels(enemy_def)
    ax.set_yticks(range(len(enemy_atk)))
    ax.set_yticklabels(enemy_atk)
    ax.set_xlabel("Enemy defense")
    ax.set_ylabel("Enemy attack")
    ax.set_title("Player win rate (player atk=15, def=10, hp=100 vs enemy hp=60)")
    for i in range(len(enemy_atk)):
        for j in range(len(enemy_def)):
            ax.text(j, i, f"{grid[i][j]:.2f}",
                    ha="center", va="center",
                    color="black" if 0.3 < grid[i][j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, label="win rate")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "combat_winrate_matrix.png", dpi=130)
    plt.close(fig)

    # -- duration histogram --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dur["bayes"], bins=range(0, max(dur["bayes"] + dur["random"]) + 2),
            alpha=0.6, label="Bayesian combat")
    ax.hist(dur["random"], bins=range(0, max(dur["bayes"] + dur["random"]) + 2),
            alpha=0.6, label="50/50 baseline")
    ax.set_xlabel("Rounds to resolve combat")
    ax.set_ylabel("Frequency")
    ax.set_title("Combat duration distribution: Bayesian vs random baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "combat_duration_dist.png", dpi=130)
    plt.close(fig)


def main() -> None:
    random.seed(0)
    print("Combat eval: calibration sweep ...")
    calib = evaluate_calibration()
    write_csv(calib, RESULTS_DIR / "combat_calibration.csv")
    mean_err = mean(r["abs_error"] for r in calib)
    max_err = max(r["abs_error"] for r in calib)
    print(f"  mean |predicted - empirical| = {mean_err:.3f}, "
          f"max = {max_err:.3f}")

    print("Combat eval: win-rate matrix ...")
    winrate = evaluate_winrate_matrix()
    write_csv(winrate, RESULTS_DIR / "combat_winrate_matrix.csv")

    print("Combat eval: duration distribution ...")
    dur = evaluate_duration_distribution()
    print(f"  Bayesian:  mean={dur['summary']['bayes_mean']:.2f} "
          f"sd={dur['summary']['bayes_sd']:.2f}")
    print(f"  Random:    mean={dur['summary']['random_mean']:.2f} "
          f"sd={dur['summary']['random_sd']:.2f}")

    make_plots(calib, winrate, dur)
    print("Done.")


if __name__ == "__main__":
    main()
