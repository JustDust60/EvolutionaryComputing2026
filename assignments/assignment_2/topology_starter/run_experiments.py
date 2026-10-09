"""Provided runner. Algorithm functions intentionally remain student TODOs.

Run experiments with the existing ARIEL environment. --plan prints the matrix
without simulation; --plot makes report figures once metrics.csv files exist.
Person 1 owns integration; Person 3 runs --plot and interprets the output.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from common import Settings
from fixed import run_fixed, run_random_search
from growing import run_growing

CONDITIONS = ("fixed_small", "fixed_large", "growing", "random_small")


def plot_results(root: Path) -> None:
    """Aggregate complete independent runs only; never invent absent results."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    grouped = {}
    reference_configs = {}
    summary = []
    for condition in CONDITIONS:
        runs = []
        for path in sorted((root / condition).glob("seed_*/metrics.csv")):
            import json
            config = json.loads(path.with_name("config.json").read_text())
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"Empty metrics file: {path}")
            if any(r["condition"] != condition for r in rows):
                raise ValueError(f"Mislabeled condition: {path}")
            if any(int(r["seed"]) != int(config["seed"]) for r in rows):
                raise ValueError(f"Mislabeled seed: {path}")
            matrix = np.array([[float(r[k]) for k in
                              ("generation", "evaluations", "best_distance",
                               "mean_hidden", "mean_connections")]
                              for r in rows])
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite metrics: {path}")
            if int(matrix[-1, 1]) != int(config["evaluation_budget"]):
                raise ValueError(f"Incomplete run (budget not reached): {path}")
            if np.any(np.diff(matrix[:, 1]) <= 0) or np.any(np.diff(matrix[:, 2]) > 1e-10):
                raise ValueError(f"Invalid evaluations/best-so-far trajectory: {path}")
            normalized = {k: v for k, v in config.items() if k not in ("seed", "condition")}
            if condition in reference_configs and reference_configs[condition] != normalized:
                raise ValueError(f"Mixed settings within {condition}: {path}")
            reference_configs[condition] = normalized
            if runs and not np.array_equal(matrix[:, :2], runs[0][1][:, :2]):
                raise ValueError(f"Different logging grids within {condition}: {path}")
            runs.append((int(config["seed"]), matrix))
        if len(runs) < 5:
            raise ValueError(f"{condition}: need at least 5 complete runs, found {len(runs)}")
        if len({seed for seed, _ in runs}) != len(runs):
            raise ValueError(f"Duplicate independent seeds in {condition}")
        grouped[condition] = runs
        finals = np.array([matrix[-1, 2] for _, matrix in runs])
        summary.append({"condition": condition, "runs": len(runs),
                        "final_distance_mean": finals.mean(),
                        "final_distance_sd": finals.std(ddof=1),
                        "final_distance_median": np.median(finals)})

    reference = grouped["fixed_small"]
    for condition, runs in grouped.items():
        if {s for s, _ in runs} != {s for s, _ in reference}:
            raise ValueError("Use the same independent seed list in every condition")
        if reference_configs[condition] != reference_configs["fixed_small"]:
            raise ValueError("Common experiment settings must match across conditions")
        if not np.array_equal(runs[0][1][:, :2], reference[0][1][:, :2]):
            raise ValueError("Use identical generation/evaluation logging grids")

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for column, xlabel, filename in ((0, "Generation (random search: batch)", "fitness_generations"),
                                     (1, "Fitness evaluations", "fitness_evaluations")):
        fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="constrained")
        for condition, runs in grouped.items():
            x = runs[0][1][:, column]
            y = np.stack([m[:, 2] for _, m in runs])
            mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
            line, = ax.plot(x, mean, label=condition.replace("_", " "))
            ax.fill_between(x, mean-sd, mean+sd, alpha=.15, color=line.get_color())
        ax.set(xlabel=xlabel, ylabel="Best-so-far target distance (m; lower is better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(figures / f"{filename}.pdf")
        fig.savefig(figures / f"{filename}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3), layout="constrained")
    for condition, runs in grouped.items():
        if condition == "random_small":
            continue
        for ax, index, ylabel in zip(axes, (3, 4), ("Mean hidden neurons", "Mean connections")):
            y = np.stack([m[:, index] for _, m in runs])
            x = runs[0][1][:, 0]
            mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
            line, = ax.plot(x, mean, label=condition.replace("_", " "))
            ax.fill_between(x, mean-sd, mean+sd, alpha=.15, color=line.get_color())
            ax.set(xlabel="Generation", ylabel=ylabel)
    axes[0].legend(fontsize=8)
    fig.savefig(figures / "topology_growth.pdf")
    fig.savefig(figures / "topology_growth.png", dpi=180)
    plt.close(fig)
    with (figures / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved verified summaries and figures in {figures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=1020)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot_results(args.output)
        return
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds and conditions must be unique")
    if min(args.population, args.offspring) < 1 or args.budget < args.population:
        parser.error("Positive population/offspring and budget >= population required")
    if (args.budget-args.population) % args.offspring:
        parser.error("Budget minus initial population must be divisible by offspring size")
    if args.duration <= 0:
        parser.error("Duration must be positive")
    settings = Settings(population_size=args.population, offspring_size=args.offspring,
                        evaluation_budget=args.budget, simulation_seconds=args.duration)
    for condition in args.conditions:
        for seed in args.seeds:
            folder = args.output / condition / f"seed_{seed}"
            print(f"{condition:12s} seed={seed} budget={args.budget} -> {folder}")
            if args.plan:
                continue
            if folder.exists():
                parser.error(f"Run folder already exists; choose a new output root: {folder}")
            run_settings = replace(settings, seed=seed)
            if condition == "growing":
                run_growing(run_settings, folder)
            elif condition == "random_small":
                run_random_search(run_settings, folder)
            else:
                run_fixed(condition, run_settings, folder)


if __name__ == "__main__":
    main()
