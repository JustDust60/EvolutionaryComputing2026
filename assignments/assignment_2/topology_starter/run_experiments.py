"""Provided runner. Algorithm functions intentionally remain student TODOs.

Run experiments with the existing ARIEL environment. --plan prints the matrix
without simulation; --plot makes report figures once metrics.csv files exist.
Person 1 owns integration; Person 3 runs --plot and interprets the output.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from common import Settings, validate_genome
from fixed import run_fixed, run_random_search
from growing import run_growing

CONDITIONS = ("fixed_small", "fixed_large", "growing", "random_small")


def plot_network(genome: dict, seed: int, distance: float, initial_hidden: int):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    validate_genome(genome)
    symbols = {"tanh": r"$\tanh$", "sigmoid": r"$\sigma$",
               "gaussian": r"$e^{-x^2}$", "sine": r"$\sin$"}
    ni = genome["input_size"]
    nh = genome["hidden_size"]
    no = genome["output_size"]
    height = max(ni, nh, no) - 1
    input_y = np.linspace(height, 0, ni)
    hidden_y = np.linspace(height, 0, nh)
    output_y = np.linspace(height, 0, no)
    wi = np.asarray(genome["w_in"])
    wo = np.asarray(genome["w_out"])
    scale = max(float(np.max(np.abs(wi))), float(np.max(np.abs(wo))), 1e-12)
    fig, ax = plt.subplots(figsize=(8.5, max(8, height * 0.55 + 3)))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.19)

    for weights, start_y, end_y, left, right in ((wi, input_y, hidden_y, 0, 4),
                                               (wo, hidden_y, output_y, 4, 8)):
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weight = weights[i, j]
                if weight == 0:
                    colour, width, alpha, style = "#9ca3af", 0.5, 0.35, ":"
                else:
                    colour = "#2474a8" if weight > 0 else "#c96d21"
                    strength = abs(weight) / scale
                    width, alpha, style = 0.4 + 1.8 * strength, 0.2 + 0.5 * strength, "-"
                ax.plot([left, right], [start_y[i], end_y[j]], color=colour,
                        linewidth=width, alpha=alpha, linestyle=style, zorder=1)

    input_labels = []
    if ni == 2 * no + 5:
        for i in range(no):
            input_labels.append(rf"$q_{{{i + 1}}}$")
            input_labels.append(rf"$\dot{{q}}_{{{i + 1}}}$")
        input_labels.extend(["target x", "target y", r"$\sin(2\pi ft)$",
                             r"$\cos(2\pi ft)$", "bias = 1"])
    else:
        for i in range(ni):
            input_labels.append(f"input {i + 1}")
    for i, y in enumerate(input_y):
        ax.add_patch(Circle((0, y), 0.23, facecolor="#edf2f7",
                            edgecolor="#334155", zorder=3))
        ax.text(-0.45, y, input_labels[i], ha="right", va="center", fontsize=11)
    for i, y in enumerate(hidden_y):
        style = "--" if i >= initial_hidden else "-"
        ax.add_patch(Circle((4, y), 0.52, facecolor="white", edgecolor="#334155",
                            linewidth=1.6, linestyle=style, zorder=3))
        ax.text(4, y, symbols[genome["activations"][i]], ha="center", va="center",
                fontsize=12, zorder=4)
        ax.text(4, y - 0.65, f"H{i + 1}", ha="center", va="top", fontsize=10)
    for i, y in enumerate(output_y):
        ax.add_patch(Circle((8, y), 0.52, facecolor="#edf2f7",
                            edgecolor="#334155", linewidth=1.6, zorder=3))
        ax.text(8, y, r"$\tanh$", ha="center", va="center", fontsize=12, zorder=4)
        ax.text(8.65, y, rf"$\theta_{{{i + 1}}}$", ha="left", va="center", fontsize=11)
    for x, label in ((0, f"Inputs ({ni})"), (4, f"Hidden ({nh})"), (8, f"Outputs ({no})")):
        ax.text(x, height + 1.1, label, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-2.4, 9.4)
    ax.set_ylim(-1.5, height + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(f"Growing network - seed {seed}", fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935, f"Best evolved controller | remaining distance: {distance:.3f} m",
             ha="center", fontsize=11)
    handles = [Line2D([], [], color="#2474a8", label="Positive weight"),
               Line2D([], [], color="#c96d21", label="Negative weight"),
               Line2D([], [], color="#9ca3af", linestyle=":", label="Zero weight")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.135),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.11, "Connections flow left to right; thickness shows relative weight magnitude.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.083, "Dashed neuron outline = added during evolution.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.053, r"Activations: $\tanh(x)$   $\sigma(x)=1/(1+e^{-x})$   $e^{-x^2}$   $\sin(x)$",
             ha="center", fontsize=11)
    fig.text(0.5, 0.025, r"Inputs are normalized; target x/y are local. Outputs: $\theta=(\pi/2)\tanh(z)$.",
             ha="center", fontsize=10)
    return fig


def plot_results(root: Path, conditions=CONDITIONS) -> None:
    """Aggregate complete independent runs only; never invent absent results."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Select at least one condition without duplicates")

    grouped = {}
    reference_configs = {}
    summary = []
    networks = []
    for condition in conditions:
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
            if condition == "growing":
                best = json.loads(path.with_name("best.json").read_text())
                validate_genome(best["genome"])
                if not np.isclose(best["fitness"], matrix[-1, 2]):
                    raise ValueError(f"Best controller does not match final metrics: {path}")
                networks.append((int(config["seed"]), best, config["small_hidden_size"]))
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

    reference_condition = conditions[0]
    reference = grouped[reference_condition]
    for condition, runs in grouped.items():
        if {s for s, _ in runs} != {s for s, _ in reference}:
            raise ValueError("Use the same independent seed list in every condition")
        if reference_configs[condition] != reference_configs[reference_condition]:
            raise ValueError("Common experiment settings must match across conditions")
        if not np.array_equal(runs[0][1][:, :2], reference[0][1][:, :2]):
            raise ValueError("Use identical generation/evaluation logging grids")

    labels = {"fixed_small": "Fixed small networks", "fixed_large": "Fixed large networks",
              "growing": "Growing networks", "random_small": "Random search"}
    label = labels[conditions[0]] if len(conditions) == 1 else "Controller comparison"
    subtitle = f"{label} | Mean and shaded ±1 SD across {len(reference)} independent runs"
    generation_label = "Generation"
    if "random_small" in conditions:
        generation_label = "Generation (random search: batch)"

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for column, xlabel, filename, title in (
            (0, generation_label, "fitness_generations", "Best target distance across generations"),
            (1, "Fitness evaluations", "fitness_evaluations", "Best target distance across evaluations")):
        fig, ax = plt.subplots(figsize=(6.3, 4.0), layout="constrained")
        fig.suptitle(title, fontsize=12, fontweight="bold")
        ax.set_title(subtitle, fontsize=9, pad=10)
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

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.8), layout="constrained")
    fig.suptitle("Network size across generations", fontsize=12, fontweight="bold")
    fig.supxlabel(subtitle, fontsize=9)
    axes[0].set_title("Hidden neurons", fontsize=11)
    axes[1].set_title("Connections", fontsize=11)
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
    for seed, best, initial_hidden in networks:
        fig = plot_network(best["genome"], seed, best["fitness"], initial_hidden)
        fig.savefig(figures / f"network_growing_seed_{seed}.pdf")
        fig.savefig(figures / f"network_growing_seed_{seed}.png", dpi=180)
        plt.close(fig)
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
        plot_results(args.output, args.conditions)
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
