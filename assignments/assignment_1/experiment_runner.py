"""Experiment harness for Assignment 1: elitism vs. generational replacement.

RESEARCH QUESTION
-----------------
Does the survivor-selection scheme change how well the EA approaches the target
body set? We compare two variants of one and the same EA - identical encoding,
identical parent selection, identical crossover and mutation operators,
identical module budget - that differ in *one* aspect:

    replacement : the offspring population replaces the parent population
                  wholesale (a generational, (mu, lambda)-style scheme).
    elitism     : the `s` best of parents + offspring always survive; the rest
                  of the new population is drawn from the remainder.

Both are compared against a random-search control given exactly the same number
of fitness evaluations.

WHAT THIS FILE DOES
-------------------
    python experiment_runner.py final          # the experiment for the report
    python experiment_runner.py tune           # the preliminary parameter sweep
    python experiment_runner.py plot <tag>     # re-draw figures from saved data

Every run is an independent seed; nothing but the seed and the condition varies
within one experiment. Raw per-generation data, the summary table, the
significance tests and the figures are all written under
`__data__/A1_2026/<tag>/`, so the report can be rebuilt from disk without
re-running anything.

WHY THE RUNS ARE NOT "VARIED A BIT" BETWEEN REPETITIONS
-------------------------------------------------------
The five-plus repetitions the assignment asks for are there to estimate the
*variance of one configuration*, which is what the mean and the std band in the
figures report. Changing hyperparameters between repetitions would fold
parameter variance into seed variance and leave both unmeasurable. Parameter
optimisation is therefore a separate, earlier phase (`tune`), run on seeds that
are disjoint from the final ones, with its own reduced budget. The final runs
then use one frozen configuration per condition.
"""

from __future__ import annotations

# Standard library
import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Third-party libraries
import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Local scripts
import A1_2026 as ea
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from tree_edit_distance import distances_to_targets, tree_edit_distance

# ============================================================================ #
#  EXPERIMENT CONFIGURATION
# ============================================================================ #

#: The configuration reported in the paper. `tune` may override k / p_c / p_m /
#: s per condition; everything else is shared by every condition and every run.
DEFAULT_HYPERPARAMETERS: dict[str, float] = {
    "pop_size": 50,
    "generations": 100,
    "k": 3,  # tournament size (parent selection)
    "p_c": 0.5,  # crossover probability
    "p_m": 0.5,  # mutation probability
    "s": 2,  # number of elites (elitism only)
    "stagnation_patience": 0,  # 0 = run the full generation budget
    "fitness_improvement_threshold": 0.0,
}

#: Seeds for the final experiment. Ten independent runs per condition - the
#: assignment asks for at least five.
FINAL_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#: Seeds for the preliminary parameter sweep. Deliberately DISJOINT from
#: FINAL_SEEDS so that the configuration is not chosen on the same runs it is
#: later evaluated on.
TUNING_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)

#: Hyperparameters that index or slice and must therefore stay integral, even
#: after a round trip through JSON (which turns 5 into 5.0).
INTEGER_HYPERPARAMETERS: tuple[str, ...] = (
    "pop_size",
    "generations",
    "k",
    "s",
    "stagnation_patience",
)


def normalise_hyperparameters(hyperparameters: dict[str, float]) -> dict[str, float]:
    """Coerce the integral hyperparameters back to `int`."""
    clean = dict(hyperparameters)
    for key in INTEGER_HYPERPARAMETERS:
        if key in clean:
            clean[key] = int(clean[key])
    return clean


#: name -> selection_method argument of `A1_2026.evolve_population`.
#: `None` marks the random-search control, which has no selection at all.
CONDITIONS: dict[str, int | None] = {
    "replacement": 0,
    "elitism": 1,
    "random": None,
}

PLOT_STYLE: dict[str, dict[str, str]] = {
    "replacement": {"color": "#1f77b4", "label": "EA - generational replacement"},
    "elitism": {"color": "#d62728", "label": "EA - elitism"},
    "random": {"color": "#7f7f7f", "label": "random search (same budget)"},
}

def labels_in(frame: pd.DataFrame) -> list[str]:
    """Curve labels in a stable order: the fixed conditions first, then sweeps."""
    present = list(dict.fromkeys(frame["condition"]))
    known = [c for c in CONDITIONS if c in present]
    return known + [c for c in present if c not in CONDITIONS]


def styles_for(labels: list[str]) -> dict[str, dict[str, str]]:
    """Colour and legend text for every curve in one figure.

    The three fixed conditions keep their colour across every figure in the
    report. Sweep labels get a sequential ramp instead of arbitrary colours,
    so that "more elitism" reads as "darker" - the ramp is sized to the number
    of sweep curves rather than indexed into a fixed list, which would wrap
    around and put the darkest shade in the middle of the sweep.
    """
    sweeps = [label for label in labels if label not in PLOT_STYLE]
    ramp = plt.get_cmap("YlOrRd")
    styles = {}
    for label in labels:
        if label in PLOT_STYLE:
            styles[label] = PLOT_STYLE[label]
            continue
        position = sweeps.index(label)
        shade = 0.35 + 0.6 * (position / max(len(sweeps) - 1, 1))
        styles[label] = {"color": ramp(shade), "label": label}
    return styles


# ============================================================================ #
#  ONE RUN
# ============================================================================ #


@dataclass
class RunResult:
    """Everything one independent run contributes to the report."""

    condition: str
    seed: int
    hyperparameters: dict[str, float]
    #: How this run is grouped in the figures. Usually the condition name, but
    #: a sweep uses it to keep "elitism, s=2" and "elitism, s=25" apart.
    label: str = ""
    best_per_generation: list[float] = field(default_factory=list)
    mean_per_generation: list[float] = field(default_factory=list)
    worst_per_generation: list[float] = field(default_factory=list)
    #: Fraction of the population that is structurally unique, per generation.
    #: This is what actually separates the two survivor schemes: replacement
    #: throws the whole parent population away every generation, elitism keeps
    #: copies of its best, and copies cost diversity.
    diversity_per_generation: list[float] = field(default_factory=list)
    best_genome: dict | None = None
    evaluations: int = 0
    wall_seconds: float = 0.0

    @property
    def best_so_far(self) -> list[float]:
        """Cumulative best - the curve that is comparable to random search."""
        return list(np.minimum.accumulate(self.best_per_generation))

    @property
    def final_best(self) -> float:
        return float(np.min(self.best_per_generation))


def _seed_everything(seed: int) -> None:
    """Seed every RNG the tree encoding can touch.

    The tree operators draw from the `random` module; numpy and torch are
    seeded as well so that switching the encoding to "nde" later does not
    silently make runs irreproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    ea.RNG = np.random.default_rng(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is an ariel dependency
        pass


def run_once(
    condition: str,
    seed: int,
    hyperparameters: dict[str, float],
    label: str = "",
) -> RunResult:
    """Execute one independent run and reduce it to per-generation statistics.

    The full population history is discarded before returning: only the
    per-generation best/mean/worst and the single best genome survive, which
    keeps the result small enough to ship back from a worker process.
    """
    targets = ea.load_targets()
    selection_method = CONDITIONS[condition]

    _seed_everything(seed)
    ea.reset_evaluation_budget()
    started = time.perf_counter()

    if selection_method is None:
        history = ea.random_search(targets, hyperparameters)
    else:
        population = ea.generate_population(int(hyperparameters["pop_size"]))
        history = ea.run_evolution(
            population,
            targets,
            selection_method,
            hyperparameters,
        )

    wall = time.perf_counter() - started

    result = RunResult(
        condition=condition,
        seed=seed,
        hyperparameters=dict(hyperparameters),
        label=label or condition,
        evaluations=ea.evaluations_used(),
        wall_seconds=wall,
    )

    champion: TreeGenome | None = None
    champion_fitness = float("inf")
    for generation in history:
        fitness = np.array([ea.evaluate(x, targets) for x in generation])
        result.best_per_generation.append(float(fitness.min()))
        result.mean_per_generation.append(float(fitness.mean()))
        result.worst_per_generation.append(float(fitness.max()))
        unique = len({ea.genome_signature(x) for x in generation})
        result.diversity_per_generation.append(unique / len(generation))
        best_index = int(fitness.argmin())
        if fitness[best_index] < champion_fitness:
            champion_fitness = float(fitness[best_index])
            champion = generation[best_index]

    if champion is not None:
        result.best_genome = champion.to_dict()
    return result


def _run_once_star(args: tuple) -> RunResult:
    """Unpack helper - ProcessPoolExecutor.map passes a single argument."""
    return run_once(*args)


def run_many(
    jobs: list[tuple],
    workers: int | None,
) -> list[RunResult]:
    """Run every (condition, seed, hyperparameters) job, in parallel if asked.

    Runs are fully independent - each seeds its own RNGs and keeps its own
    fitness cache - so distributing them over processes changes nothing but
    the wall clock.
    """
    if workers == 1:
        results = []
        for index, job in enumerate(jobs, start=1):
            print(f"  [{index}/{len(jobs)}] {job[-1] or job[0]} seed={job[1]}", flush=True)
            results.append(_run_once_star(job))
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = []
        for index, result in enumerate(pool.map(_run_once_star, jobs), start=1):
            print(
                f"  [{index}/{len(jobs)}] {result.label} "
                f"seed={result.seed} best={result.final_best:.3f} "
                f"({result.wall_seconds:.1f}s)",
                flush=True,
            )
            results.append(result)
    return results


# ============================================================================ #
#  PERSISTENCE
# ============================================================================ #


def results_to_frame(results: list[RunResult]) -> pd.DataFrame:
    """Long-format table: one row per (condition, seed, generation)."""
    rows = []
    for result in results:
        best_so_far = result.best_so_far
        for generation, best in enumerate(result.best_per_generation):
            rows.append({
                "condition": result.label,
                "selection": result.condition,
                "seed": result.seed,
                "generation": generation,
                "best": best,
                "best_so_far": best_so_far[generation],
                "mean": result.mean_per_generation[generation],
                "worst": result.worst_per_generation[generation],
                "diversity": result.diversity_per_generation[generation],
            })
    return pd.DataFrame(rows)


def save_experiment(
    tag: str,
    results: list[RunResult],
    hyperparameters: dict[str, float],
    per_condition: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Write raw data, metadata and champion genomes under `__data__`."""
    out_dir = ea.DATA / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results_to_frame(results).to_csv(out_dir / "runs.csv", index=False)

    summary = pd.DataFrame([
        {
            "condition": r.label,
            "selection": r.condition,
            "seed": r.seed,
            "final_best": r.final_best,
            "final_mean": r.mean_per_generation[-1],
            "final_diversity": r.diversity_per_generation[-1],
            "generations_run": len(r.best_per_generation) - 1,
            "evaluations": r.evaluations,
            "wall_seconds": r.wall_seconds,
            **{k: r.hyperparameters[k] for k in ("k", "p_c", "p_m", "s")},
        }
        for r in results
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)

    genome_dir = out_dir / "best_genomes"
    genome_dir.mkdir(exist_ok=True)
    for result in results:
        if result.best_genome is not None:
            safe = result.label.replace(" ", "_").replace("=", "")
            path = genome_dir / f"{safe}_seed{result.seed}.json"
            path.write_text(json.dumps(result.best_genome, indent=2))

    metadata = {
        "tag": tag,
        "encoding": ea.GENOTYPE,
        "num_of_modules": ea.NUM_OF_MODULES,
        "targets": str(ea.TARGET_DIR),
        "hyperparameters": hyperparameters,
        "per_condition_hyperparameters": per_condition or {},
        "conditions": {k: v for k, v in CONDITIONS.items()},
        "seeds": sorted({r.seed for r in results}),
        "nominal_budget": int(
            hyperparameters["pop_size"] * (hyperparameters["generations"] + 1),
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


# ============================================================================ #
#  STATISTICS
# ============================================================================ #


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: the non-parametric effect size for two samples.

    Returns the probability that a random draw from `a` exceeds one from `b`,
    minus the probability of the reverse. 0 means the two are interchangeable;
    +-1 means no overlap at all.
    """
    greater = sum((x > y) for x in a for y in b)
    smaller = sum((x < y) for x in a for y in b)
    return (greater - smaller) / (len(a) * len(b))


def compare_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests on the final best fitness of each run.

    Five to ten runs per condition is far too small to lean on normality, so
    this is a rank test rather than a t-test, reported alongside Cliff's delta
    because a p-value on ten samples says nothing about how big the gap is.
    """
    rows = []
    names = list(summary["condition"].unique())
    for first, second in product(names, names):
        if names.index(first) >= names.index(second):
            continue
        a = summary.loc[summary["condition"] == first, "final_best"].to_numpy()
        b = summary.loc[summary["condition"] == second, "final_best"].to_numpy()
        statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "condition_a": first,
            "condition_b": second,
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "U": float(statistic),
            "p_value": float(p_value),
            "cliffs_delta": cliffs_delta(a, b),
            "n_a": len(a),
            "n_b": len(b),
        })
    return pd.DataFrame(rows)


def generations_to_threshold(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """First generation at which each run's best-so-far fitness reaches `threshold`.

    Final fitness alone cannot distinguish two schemes that both converge to the
    same place; this asks the other half of the question - which one gets there
    first. Runs that never reach the threshold are reported as NaN and counted
    separately, because filling them in with the generation budget would silently
    flatter whichever scheme failed most often.
    """
    rows = []
    for (condition, seed), group in frame.groupby(["condition", "seed"]):
        group = group.sort_values("generation")
        reached = group.loc[group["best_so_far"] <= threshold, "generation"]
        rows.append({
            "condition": condition,
            "seed": seed,
            "generations_to_threshold": (
                int(reached.iloc[0]) if len(reached) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def descriptive_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / median / min-max of final best fitness, per condition."""
    grouped = summary.groupby("condition", sort=False)["final_best"]
    table = grouped.agg(["count", "mean", "std", "median", "min", "max"])
    by_condition = summary.groupby("condition", sort=False)
    table["final_diversity"] = by_condition["final_diversity"].mean()
    table["evaluations"] = by_condition["evaluations"].mean()
    return table.reset_index()


# ============================================================================ #
#  FIGURES
# ============================================================================ #


def _curve(frame: pd.DataFrame, condition: str, column: str):
    """Mean and std across runs, per generation, for one condition."""
    subset = frame[frame["condition"] == condition]
    pivot = subset.pivot(index="generation", columns="seed", values=column)
    return (
        pivot.index.to_numpy(),
        pivot.mean(axis=1).to_numpy(),
        pivot.std(axis=1, ddof=1).to_numpy(),
    )


def plot_comparison(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """One line per condition: mean across runs, shaded +- one std."""
    figure, axis = plt.subplots(figsize=(6.0, 3.6))

    labels = labels_in(frame)
    styles = styles_for(labels)
    for condition in labels:
        style = styles[condition]
        generations, mean, std = _curve(frame, condition, column)
        axis.plot(generations, mean, color=style["color"], label=style["label"])
        axis.fill_between(
            generations,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )

    axis.set_xlabel("generation")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_final_distribution(summary: pd.DataFrame, path: Path) -> None:
    """Box plot of the final best fitness of every run, per condition."""
    conditions = labels_in(summary)
    data = [
        summary.loc[summary["condition"] == c, "final_best"].to_numpy()
        for c in conditions
    ]

    styles = styles_for(conditions)
    figure, axis = plt.subplots(figsize=(1.4 * len(conditions) + 1.6, 3.4))
    boxes = axis.boxplot(data, patch_artist=True, widths=0.55)
    for patch, condition in zip(boxes["boxes"], conditions):
        patch.set_facecolor(styles[condition]["color"])
        patch.set_alpha(0.35)
    for median in boxes["medians"]:
        median.set_color("black")

    for index, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(values))
        axis.scatter(
            index + jitter,
            values,
            s=14,
            color="black",
            alpha=0.6,
            zorder=3,
        )

    axis.set_xticks(range(1, len(conditions) + 1))
    axis.set_xticklabels(conditions, fontsize=8, rotation=20, ha="right")
    axis.set_ylabel("final best fitness (lower is better)")
    axis.set_title(f"Final best over {len(data[0])} runs", fontsize=10)
    axis.grid(alpha=0.25, axis="y", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"  wrote {path}")


#: Module type -> colour, for the body-graph figure.
MODULE_COLORS: dict[str, str] = {
    "CORE": "#2c3e50",
    "BRICK": "#e0a458",
    "HINGE": "#5b9bd5",
}


def _draw_body(axis, graph, title: str) -> None:
    """Draw one body graph as a rooted tree, coloured by module type.

    The MuJoCo renderer shows the body as it would stand in the world, which is
    what `show_body` is for. This draws the structure the fitness function
    actually compares: the tree, its module types and its branching.
    """
    import networkx as nx

    root = next(n for n, degree in graph.in_degree() if degree == 0)
    depths = nx.single_source_shortest_path_length(graph, root)

    # a simple layered layout: depth sets y, order within a depth sets x
    by_depth: dict[int, list] = {}
    for node in nx.dfs_preorder_nodes(graph, root):
        by_depth.setdefault(depths[node], []).append(node)
    positions = {}
    for depth, nodes in by_depth.items():
        for index, node in enumerate(nodes):
            offset = (index - (len(nodes) - 1) / 2) / max(len(nodes), 1)
            positions[node] = (offset, -depth)

    colors = [
        MODULE_COLORS.get(graph.nodes[n].get("type", ""), "#bbbbbb")
        for n in graph.nodes
    ]
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color="#999999", width=0.8)
    nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=colors, node_size=70)
    axis.set_title(title, fontsize=8)
    axis.set_axis_off()


def plot_bodies(tag: str) -> None:
    """Champion body of each condition next to every target body."""
    out_dir = ea.DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    targets = ea.load_targets()
    labels = [c for c in labels_in(summary) if c != "random"] + ["random"]

    columns = max(len(labels), len(targets))
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(1.7 * columns, 5.0),
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.set_axis_off()

    for column, label in enumerate(labels):
        subset = summary[summary["condition"] == label]
        if subset.empty:
            continue
        best_row = subset.loc[subset["final_best"].idxmin()]
        path = champion_path(out_dir, label, int(best_row["seed"]))
        genome = TreeGenome.from_dict(json.loads(path.read_text()))
        _draw_body(
            axes[0][column],
            genome.to_networkx(),
            f"{label}\nfitness {best_row['final_best']:.2f}",
        )

    for column, target in enumerate(targets):
        _draw_body(
            axes[1][column],
            target,
            f"target {column}\n{target.number_of_nodes()} modules",
        )

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=color, label=name, markersize=6,
        )
        for name, color in MODULE_COLORS.items()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle("Best evolved bodies (top) and the target set (bottom)", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    path = out_dir / "fig_bodies.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"  wrote {path}")


def make_figures(tag: str) -> None:
    """Rebuild every report figure from the saved CSVs."""
    out_dir = ea.DATA / tag
    frame = pd.read_csv(out_dir / "runs.csv")
    summary = pd.read_csv(out_dir / "summary.csv")
    runs = summary.groupby("condition").size().max()

    plot_comparison(
        frame,
        "best_so_far",
        "best fitness so far (lower is better)",
        f"Best-so-far fitness, mean +- std over {runs} independent runs",
        out_dir / "fig_best_so_far.png",
    )
    plot_comparison(
        frame,
        "best",
        "best fitness in generation (lower is better)",
        f"Best-of-generation fitness, mean +- std over {runs} runs",
        out_dir / "fig_best_per_generation.png",
    )
    plot_comparison(
        frame,
        "mean",
        "population mean fitness (lower is better)",
        f"Population mean fitness, mean +- std over {runs} runs",
        out_dir / "fig_population_mean.png",
    )
    plot_comparison(
        frame,
        "diversity",
        "fraction of population that is unique",
        f"Population diversity, mean +- std over {runs} runs",
        out_dir / "fig_diversity.png",
    )
    plot_final_distribution(summary, out_dir / "fig_final_distribution.png")
    plot_bodies(tag)


# ============================================================================ #
#  REPORTING
# ============================================================================ #


def champion_path(out_dir: Path, label: str, seed: int) -> Path:
    """Where `save_experiment` put the best genome of one run."""
    safe = label.replace(" ", "_").replace("=", "")
    return out_dir / "best_genomes" / f"{safe}_seed{seed}.json"


def report(tag: str) -> None:
    """Print - and save - the numbers the Results section needs."""
    out_dir = ea.DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    metadata = json.loads((out_dir / "meta.json").read_text())

    targets = ea.load_targets()
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1 :]
    ]

    descriptives = descriptive_table(summary)
    tests = compare_conditions(summary)

    descriptives.to_csv(out_dir / "table_descriptives.csv", index=False)
    tests.to_csv(out_dir / "table_significance.csv", index=False)

    print()
    print(f"experiment            : {tag}")
    print(f"budget per run        : {metadata['nominal_budget']} evaluations")
    print(f"target spread (floor) : mean pairwise distance {np.mean(spread):.2f}")
    print()
    print("FINAL BEST FITNESS PER CONDITION (lower is better)")
    print(descriptives.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("MANN-WHITNEY U ON FINAL BEST FITNESS")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Convergence speed, measured at a threshold every EA condition actually
    # reaches: the worst of the per-condition median final fitnesses, excluding
    # the random-search control (which reaches nothing near it).
    frame = pd.read_csv(out_dir / "runs.csv")
    evolutionary = descriptives[descriptives["condition"] != "random"]
    if not evolutionary.empty:
        threshold = float(evolutionary["median"].max())
        speed = generations_to_threshold(frame, threshold)
        grouped = speed.groupby("condition", sort=False)
        speed_table = grouped["generations_to_threshold"].agg(
            reached="count",  # NaN runs never got there, and count skips NaN
            median_generation="median",
            mean_generation="mean",
            std_generation="std",
        )
        speed_table.insert(0, "runs", grouped.size())
        speed_table = speed_table.reindex(labels_in(summary)).reset_index()
        speed.to_csv(out_dir / "table_convergence_speed.csv", index=False)
        print()
        print(f"GENERATIONS TO REACH FITNESS {threshold:.3f} (lower is faster)")
        print(
            "The threshold is the worst per-condition median, so roughly half "
            "the runs of the slowest EA condition reach it by construction; "
            "what the comparison is about is WHEN they do.",
        )
        print(
            speed_table.to_string(index=False, float_format=lambda v: f"{v:.1f}"),
        )
    print()

    # The single best body found anywhere, per condition - worth a figure in
    # the report, and a sanity check that the fitness is rewarding what we think.
    print("CHAMPION BODY PER CONDITION")
    for condition in labels_in(summary):
        subset = summary[summary["condition"] == condition]
        if subset.empty:
            continue
        best_row = subset.loc[subset["final_best"].idxmin()]
        path = champion_path(out_dir, condition, int(best_row["seed"]))
        genome = TreeGenome.from_dict(json.loads(path.read_text()))
        graph = genome.to_networkx()
        distances = ", ".join(
            f"{d:.1f}" for d in distances_to_targets(graph, targets)
        )
        print(
            f"  {condition:<12} fitness {best_row['final_best']:.3f} "
            f"| {graph.number_of_nodes()} modules | per-target {distances}",
        )


def render_champions(tag: str) -> None:
    """Render the best body of each condition to a PNG via MuJoCo."""
    out_dir = ea.DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    for condition in labels_in(summary):
        subset = summary[summary["condition"] == condition]
        if subset.empty:
            continue
        best_row = subset.loc[subset["final_best"].idxmin()]
        path = champion_path(out_dir, condition, int(best_row["seed"]))
        genome = TreeGenome.from_dict(json.loads(path.read_text()))
        ea.show_body(
            genome.to_networkx(),
            "frame",
            file_name=f"{tag}_champion_{condition.replace(' ', '_').replace('=', '')}",
        )


# ============================================================================ #
#  SUB-COMMANDS
# ============================================================================ #


def command_final(args: argparse.Namespace) -> None:
    """The experiment reported in the paper."""
    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
    hyperparameters["pop_size"] = args.pop_size
    hyperparameters["generations"] = args.generations

    per_condition: dict[str, dict[str, float]] = {}
    if args.config is not None:
        tuned = json.loads(Path(args.config).read_text())
        per_condition = tuned.get("per_condition", tuned)
        print(f"loaded tuned parameters from {args.config}")

    seeds = tuple(FINAL_SEEDS[: args.seeds])
    jobs = []
    for condition in CONDITIONS:
        condition_hp = dict(hyperparameters)
        condition_hp.update(per_condition.get(condition, {}))
        # keep the shared budget authoritative even if a tuning file differs
        condition_hp["pop_size"] = hyperparameters["pop_size"]
        condition_hp["generations"] = hyperparameters["generations"]
        condition_hp = normalise_hyperparameters(condition_hp)
        for seed in seeds:
            jobs.append((condition, seed, condition_hp, condition))

    budget = int(hyperparameters["pop_size"] * (hyperparameters["generations"] + 1))
    print(
        f"final experiment: {len(CONDITIONS)} conditions x {len(seeds)} seeds "
        f"= {len(jobs)} runs, {budget} evaluations each",
    )

    results = run_many(jobs, args.workers)
    out_dir = save_experiment(args.tag, results, hyperparameters, per_condition)
    print(f"raw data written to {out_dir}")

    make_figures(args.tag)
    report(args.tag)
    if args.render:
        render_champions(args.tag)


def command_tune(args: argparse.Namespace) -> None:
    """Preliminary parameter sweep on seeds disjoint from the final ones.

    A short, cheap grid: its job is to pick one configuration per condition
    before the real runs start, not to produce results for the report. Only
    the EA conditions are swept - random search has no parameters to tune.
    """
    grid = {
        "k": args.k,
        "p_c": args.p_c,
        "p_m": args.p_m,
        "s": args.s,
    }
    keys = list(grid)

    jobs = []
    for condition in ("replacement", "elitism"):
        for values in product(*(grid[key] for key in keys)):
            candidate = dict(DEFAULT_HYPERPARAMETERS)
            candidate["pop_size"] = args.pop_size
            candidate["generations"] = args.generations
            candidate.update(dict(zip(keys, values)))
            candidate = normalise_hyperparameters(candidate)
            # `s` does nothing under generational replacement; sweeping it there
            # would just run the same configuration several times.
            if condition == "replacement" and candidate["s"] != grid["s"][0]:
                continue
            for seed in TUNING_SEEDS[: args.seeds]:
                jobs.append((condition, seed, candidate, condition))

    print(
        f"tuning sweep: {len(jobs)} runs "
        f"({args.pop_size} individuals x {args.generations} generations)",
    )
    results = run_many(jobs, args.workers)

    rows = []
    for result in results:
        row = {"condition": result.condition, "seed": result.seed}
        row.update({key: result.hyperparameters[key] for key in keys})
        row["final_best"] = result.final_best
        rows.append(row)
    frame = pd.DataFrame(rows)

    out_dir = ea.DATA / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "tuning_runs.csv", index=False)

    aggregated = (
        frame.groupby(["condition", *keys])["final_best"]
        .agg(["mean", "std", "min"])
        .reset_index()
        .sort_values(["condition", "mean"])
    )
    aggregated.to_csv(out_dir / "tuning_summary.csv", index=False)

    print()
    print("TUNING RESULTS - best five configurations per condition")
    print("(mean final best over the tuning seeds; lower is better)")
    for condition in aggregated["condition"].unique():
        print(f"\n  {condition}")
        top = aggregated[aggregated["condition"] == condition].head(5)
        print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # A few seeds per cell make the argmin itself noisy: the top configurations
    # usually sit inside each other's spread. So rather than trusting the single
    # winner, take every configuration whose mean is within one standard error
    # of the best and pick the one with the lowest mean among those - and say in
    # the report that the choice was not sharply determined.
    best: dict[str, dict[str, float]] = {}
    for condition in aggregated["condition"].unique():
        subset = aggregated[aggregated["condition"] == condition]
        floor = subset["mean"].min()
        tolerance = subset.loc[subset["mean"].idxmin(), "std"] / np.sqrt(
            len(TUNING_SEEDS[: args.seeds]),
        )
        contenders = subset[subset["mean"] <= floor + tolerance]
        winner = contenders.loc[contenders["mean"].idxmin()]
        best[condition] = {key: float(winner[key]) for key in keys}
        print(
            f"\n  {condition}: {len(contenders)} configuration(s) within one "
            f"standard error ({tolerance:.3f}) of the best mean {floor:.3f}",
        )

    # Written twice on purpose: once beside the raw sweep data, and once next
    # to the code, because `__data__` is gitignored and the final experiment
    # takes this file as an input. A fresh clone has to be able to reproduce
    # step 2 without first re-running the sweep.
    payload = json.dumps({"per_condition": best}, indent=2)
    (out_dir / "tuned_parameters.json").write_text(payload)
    config_path = HERE / "tuned_parameters.json"
    config_path.write_text(payload)
    print()
    print(f"chosen configuration written to {config_path}")
    print(json.dumps(best, indent=2))
    print()
    print("run the final experiment with:")
    print(f"  python experiment_runner.py final --config {config_path}")


def command_sweep(args: argparse.Namespace) -> None:
    """Sweep the elitism strength `s` at the full experimental budget.

    The headline comparison pits generational replacement against elitism at one
    setting of `s`. That answers "is THIS elitism better", not "does elitism
    help" - and with s=2 out of a population of 50, the elitist scheme keeps two
    individuals and fills the other 48 at random from parents plus offspring,
    which is barely elitism at all. Sweeping `s` turns the yes/no question into a
    dose-response curve, with generational replacement (s = 0 elites, nothing
    survives) and random search as the two ends of the scale.
    """
    hyperparameters = normalise_hyperparameters({
        **DEFAULT_HYPERPARAMETERS,
        "pop_size": args.pop_size,
        "generations": args.generations,
    })
    if args.config is not None:
        tuned = json.loads(Path(args.config).read_text())
        per_condition = tuned.get("per_condition", tuned)
        hyperparameters.update(per_condition.get("elitism", {}))
        hyperparameters = normalise_hyperparameters(hyperparameters)
        hyperparameters["pop_size"] = args.pop_size
        hyperparameters["generations"] = args.generations

    seeds = tuple(FINAL_SEEDS[: args.seeds])
    jobs: list[tuple] = []

    # the two reference points
    for condition in ("replacement", "random"):
        for seed in seeds:
            jobs.append((condition, seed, dict(hyperparameters), condition))

    for elites in args.s:
        # s == pop_size is legal and is the interesting endpoint: every slot is
        # filled by rank and none at random, i.e. textbook (mu + lambda)
        # truncation selection. Above that there are not enough survivors to
        # fill the population.
        if elites > args.pop_size:
            print(f"  skipping s={elites}: above pop_size={args.pop_size}")
            continue
        candidate = dict(hyperparameters)
        candidate["s"] = int(elites)
        label = f"elitism s={elites}"
        for seed in seeds:
            jobs.append(("elitism", seed, candidate, label))

    print(
        f"elitism sweep: {len(jobs)} runs over s in {args.s}, "
        f"{len(seeds)} seeds each",
    )
    results = run_many(jobs, args.workers)
    out_dir = save_experiment(args.tag, results, hyperparameters)
    print(f"raw data written to {out_dir}")

    make_figures(args.tag)
    report(args.tag)


def command_plot(args: argparse.Namespace) -> None:
    """Re-draw the figures and re-print the tables from saved data."""
    make_figures(args.tag)
    report(args.tag)
    if args.render:
        render_champions(args.tag)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--tag", default=None, help="output folder name")
        sub.add_argument(
            "--workers",
            type=int,
            default=None,
            help="parallel processes (1 = serial, default = all cores)",
        )
        sub.add_argument(
            "--render",
            action="store_true",
            help="also render the champion bodies with MuJoCo",
        )

    final = subparsers.add_parser("final", help="the experiment for the report")
    add_common(final)
    final.add_argument("--pop-size", type=int, default=50)
    final.add_argument("--generations", type=int, default=100)
    final.add_argument(
        "--seeds",
        type=int,
        default=len(FINAL_SEEDS),
        help=f"how many of {FINAL_SEEDS} to use (assignment minimum: 5)",
    )
    final.add_argument(
        "--config",
        default=None,
        help="tuned_parameters.json from a previous `tune` run",
    )
    final.set_defaults(func=command_final, default_tag="final")

    tune = subparsers.add_parser("tune", help="preliminary parameter sweep")
    add_common(tune)
    # The sweep runs at the FINAL population size on purpose: `s` is a count
    # of elites, not a fraction, so a value tuned at a smaller population does
    # not carry over. Only the generation budget is cut, to keep it cheap.
    tune.add_argument("--pop-size", type=int, default=50)
    tune.add_argument("--generations", type=int, default=40)
    tune.add_argument("--seeds", type=int, default=len(TUNING_SEEDS))
    tune.add_argument("--k", type=int, nargs="+", default=[2, 3, 5])
    tune.add_argument("--p-c", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    tune.add_argument("--p-m", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    tune.add_argument("--s", type=int, nargs="+", default=[2, 5, 10])
    tune.set_defaults(func=command_tune, default_tag="tuning")

    sweep = subparsers.add_parser(
        "sweep",
        help="vary the elitism strength s at the full budget",
    )
    add_common(sweep)
    sweep.add_argument("--pop-size", type=int, default=50)
    sweep.add_argument("--generations", type=int, default=100)
    sweep.add_argument("--seeds", type=int, default=len(FINAL_SEEDS))
    sweep.add_argument("--s", type=int, nargs="+", default=[2, 5, 10, 25, 45, 50])
    sweep.add_argument(
        "--config",
        default=None,
        help="tuned_parameters.json - its elitism entry sets the other params",
    )
    sweep.set_defaults(func=command_sweep, default_tag="sweep_elitism")

    plot = subparsers.add_parser("plot", help="redraw figures from saved data")
    add_common(plot)
    plot.set_defaults(func=command_plot, default_tag="final")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.tag is None:
        args.tag = args.default_tag
    args.func(args)


if __name__ == "__main__":
    main()
