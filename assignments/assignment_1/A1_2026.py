"""EC A1 template code - evolving robot morphologies with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo* file for starting you out with assignment 1. 
It samples one body at random, decodes it, scores it
against a set of target bodies, and shows you the result. 

*Your Job* section at the bottom of this file summarises the programming task. Full assignment description can be found in the pdf file on Canvas.


THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve a robot BODY that is as structurally close as possible to a whole set
of given target bodies at once.

    fitness = mean tree edit distance to every body in TARGET_DIR,
              plus one standard deviation across those per-target distances
"""


# Standard library
import random
from pathlib import Path
from typing import Literal
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer
import pandas as pd
from scipy.stats import mannwhitneyu

import copy
import matplotlib.pyplot as plt
# Standard library

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

import ariel.ec.genotypes.tree.operators as op
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 43
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  # "nde" | "tree" 
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

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


#: Seeds for the final experiment. Ten independent runs per condition
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


#: Parameters per condition, previously read from shared_parameters.json.
#: Both variants identical, so survivor selection is the only difference.
SHARED_PARAMETERS: dict[str, dict[str, float]] = {
    "replacement": {"k": 5, "p_c": 0.6, "p_m": 0.8, "s": 2},
    "elitism": {"k": 5, "p_c": 0.6, "p_m": 0.8, "s": 2},
}

#: Per-variant tuning result, previously tuned_parameters.json. Used as a
#: robustness check. `tune` prints a replacement for this dict.
TUNED_PARAMETERS: dict[str, dict[str, float]] = {
    "replacement": {"k": 5, "p_c": 0.6, "p_m": 0.8, "s": 2},
    "elitism": {"k": 5, "p_c": 0.3, "p_m": 0.8, "s": 2},
}

#: `--config shared` / `--config tuned`, or a path to a JSON file.
NAMED_CONFIGS: dict[str, dict[str, dict[str, float]]] = {
    "shared": SHARED_PARAMETERS,
    "tuned": TUNED_PARAMETERS,
}


def load_config(name: str | None) -> dict[str, dict[str, float]]:
    """Resolve --config: a built-in name, or a path to a JSON file."""
    if name is None:
        return {}
    if name in NAMED_CONFIGS:
        return NAMED_CONFIGS[name]
    payload = json.loads(Path(name).read_text())
    return payload.get("per_condition", payload)



# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
# ============================================================================ #


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


# ============================================================================ #
#  2. THE GENOTYPE CONTRACT
# ============================================================================ #
#
# You may use EITHER of ARIEL's two body encodings below. You may NOT invent
# your own, and CPPN is not offered for this assignment.
# Whichever you pick, the contract is the same and it is very short:
#
#       your genotype  --(its decoder)-->  nx.DiGraph  -->  fitness
#
# That DiGraph is the phenotype, and it is all the fitness function ever sees:
#
#       nodes carry   type      : "CORE" | "BRICK" | "HINGE"
#                     rotation  : "DEG_0" | "DEG_45" | "DEG_90"
#       edges carry   face      : "FRONT" | "BACK" | "RIGHT" | "LEFT"
#                                 | "TOP" | "BOTTOM"
#
# THE TWO ENCODINGS
#
#   "nde"   NeuralDevelopmentalEncoding + HighProbabilityDecoder
#           Genotype: three fixed-length float vectors (type / connection /
#           rotation genes). An INDIRECT encoding - a small vector is expanded
#           by a fixed neural network into probability matrices, which are
#           then decoded greedily into a body.
#           -> Fixed-length real vector. Standard real-valued operators work
#              out of the box. But the genotype-phenotype map is wildly
#              non-linear: a small mutation can rebuild the robot entirely.
#           -> IMPORTANT: `NeuralDevelopmentalEncoding`'s internal network is
#              randomly (re-)initialised every time you construct it, and NOT
#              derived from the genotype you pass in. If your EA's decode step
#              builds a fresh `NeuralDevelopmentalEncoding(...)` per individual
#              (the natural way to write it - see `random_nde_body` below),
#              the SAME genotype decodes to a DIFFERENT random body every call,
#              and fitness stops reflecting the genotype at all. Construct it
#              ONCE for your whole run and reuse that one instance's
#              `.forward()` for every genotype you decode.
#           -> ALSO IMPORTANT: `NeuralDevelopmentalEncoding` is a
#              `torch.nn.Module`. Its weight initialisation uses torch's own
#              RNG, entirely separate from numpy/random. `np.random.seed(...)`
#              and `random.seed(...)` do NOT control it - you also need
#              `torch.manual_seed(...)`, or your results will not reproduce
#              across separate runs even with "the same" seed.
#
#   "tree"  TreeGenome + its operators
#           Genotype: the tree itself, nodes and edges.
#           A DIRECT encoding - genotype and phenotype are the same shape.
#           -> ariel.ec.genotypes.tree.operators already gives you
#              random_tree, add_node, remove_subtree, subtree_swap,
#              crossover_subtree, mutate_hoist, mutate_shrink,
#              mutate_replace_node, mutate_subtree_replacement.
#              Variable-length genotype, so watch for bloat.
#
#
# Below, each encoding gets ONE random genotype, decoded to a graph. That is
# your starting point, not your solution: your EA has to search this space,
# not sample it once.
#
# ============================================================================ #

# NDE settings
GENOTYPE_SIZE: int = 64  # length of each of the three NDE gene vectors


# Constructed ONCE, at import time, and reused for every decode call below and
# in your own EA. See the "IMPORTANT" note on "nde" in THE GENOTYPE CONTRACT
# above: rebuilding this per individual silently breaks the genotype -> body
# mapping, because its internal network randomises on construction.
_NDE = NeuralDevelopmentalEncoding(
    number_of_modules=NUM_OF_MODULES,
    genotype_size=GENOTYPE_SIZE,
)


def random_nde_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random NDE genotype and decode it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. The three vectors below are the
    genotype: that is what you mutate, recombine and select on. Note this
    function does NOT construct its own `NeuralDevelopmentalEncoding` - it
    reuses the module-level `_NDE` instance. Do the same in your EA.

    `num_modules` must match the value `_NDE` was built with (NUM_OF_MODULES).
    """
    genotype = [
        RNG.uniform(-1.0, 1.0, GENOTYPE_SIZE).astype(np.float32)  # module types
        for _ in range(3)  # types, connections, rotations
    ]

    type_p, conn_p, rot_p = _NDE.forward(genotype)

    decoder = HighProbabilityDecoder(num_modules)
    return decoder.probability_matrices_to_graph(type_p, conn_p, rot_p)


def random_tree_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random tree genotype and convert it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. Here the genotype IS the tree, so
    `TreeGenome` is what your population holds - call `.to_networkx()` only
    when it is time to compute fitness.
    """
    genome = random_tree(max_modules=num_modules)
    return genome.to_networkx()


def random_body(
    genotype: GenotypeTypes = GENOTYPE,
    num_modules: int = NUM_OF_MODULES,
) -> nx.DiGraph:
    """Sample one random body using the chosen encoding."""
    match genotype:
        case "nde":
            return random_nde_body(num_modules)
        case "tree":
            return random_tree_body(num_modules)


# ============================================================================ #
#  3. FITNESS
# ============================================================================ #
#
# Fitness is the MEAN tree edit distance to every target body, PLUS one
# standard deviation across those per-target distances. LOWER IS BETTER, and
# 0.0 would mean your body is identical to all of them at once - which, since
# the targets differ from each other, is impossible. There is a floor above
# zero here and you will not reach it. Work out roughly where it is: a body
# cannot be closer to a set than the set is to itself.
#
# The distance itself lives in tree_edit_distance.py.
# Read that file - you cannot reason about your EA's behaviour without knowing what it is climbing.
#
# ============================================================================ #


def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


# ============================================================================ #
#  4. LOOKING AT A BODY
# ============================================================================ #


def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


# ============================================================================ #
#  5. Evolutionary Algorithm
# ============================================================================ #

# Every genome is scored once and memoised. Without this, elitism (which keeps
# survivors) and tournament selection (which re-inspects parents) would each
# rescore the same bodies, so the evaluation count would depend on the
# selection scheme rather than on search effort.

_FITNESS_CACHE: dict[str, float] = {}
_EVAL_COUNT: int = 0


def genome_signature(genome: TreeGenome) -> str:
    """Canonical string key for a tree genome, used to memoise its fitness."""
    nodes = ";".join(
        f"{nid}:{attrs['type']}:{attrs['rotation']}"
        for nid, attrs in sorted(genome.nodes.items())
    )
    edges = ";".join(
        f"{e['parent']}>{e['child']}@{e['face']}"
        for e in sorted(
            genome.edges,
            key=lambda e: (e["parent"], e["child"], e["face"]),
        )
    )
    return nodes + "|" + edges


def evaluate(genome: TreeGenome, targets: list[nx.DiGraph]) -> float:
    """Fitness of one genome, computed once and cached. LOWER IS BETTER."""
    global _EVAL_COUNT
    key = genome_signature(genome)
    cached = _FITNESS_CACHE.get(key)
    if cached is not None:
        return cached
    value = fitness_function(genome.to_networkx(), targets)
    _FITNESS_CACHE[key] = value
    _EVAL_COUNT += 1
    return value


def reset_evaluation_budget() -> None:
    """Clear the fitness cache and counter. Call once at the start of a run."""
    global _EVAL_COUNT
    _FITNESS_CACHE.clear()
    _EVAL_COUNT = 0


def evaluations_used() -> int:
    """How many distinct genomes have been scored since the last reset."""
    return _EVAL_COUNT


def generate_population(pop_size: int,):
    population = []
    for _ in range(pop_size):
        #Fixed a little bug, tournament selection expects tree genome, not tuple
        population.append((random_tree(max_modules=NUM_OF_MODULES)))
    return population

def mutation(parent: nx.DiGraph, p: float):
    child = copy.deepcopy(parent)
    if random.random() > p:
        return child
    op.mutate_subtree_replacement(genome=child, max_modules=10)
    return child

def crossover(parent_1: nx.DiGraph, parent_2: nx.DiGraph, p: float):
    if random.random() > p:
            return parent_1, parent_2
    child_1, child_2 = op.crossover_subtree(parent_1, parent_2)
    return child_1, child_2

def tournament_selection(generation: list, targets: list[nx.DiGraph], k: int):
    # Randomly select k individuals from population and returns the one with the lowest fitness
    current_winner = random.choice(generation)
    fitness_cw = evaluate(current_winner, targets)
    for _ in range(k-1):
        candidate = random.choice(generation)
        fitness_can = evaluate(candidate, targets)
        if fitness_can < fitness_cw:
            fitness_cw = fitness_can
            current_winner = candidate
    return current_winner



def replacement_selection(parents: list, children: list, targets: list[nx.DiGraph] | None = None):
    mu = len(parents)

    if len(children) < mu:
        raise ValueError("Replacement selection needs number of children >= parents")
    elif len(children) == mu:
        return children
    elif targets is None:
        return random.sample(children, mu)

    fitness = []
    for child in children:
        fitness.append(evaluate(child, targets))

    scored = []
    for i in range(len(fitness)):
        scored.append((fitness[i], i))

    scored.sort()

    survivors = []
    for i in range(mu):
        fittest_index = scored[i][1]
        survivors.append(children[fittest_index])

    return survivors

def elitism_selection(parents: list, children: list, targets: list[nx.DiGraph], s: int):
    # Combines the previous generation and its children and takes s of the ones with the 
    # lowest fitness, and randomly selects from the remaining individuals to create the new
    # generation. 
    parent_child = parents.copy()
    parent_child.extend(children)
    parent_child_fitness = [evaluate(x, targets) for x in parent_child]
    #Another small edge-case fix, comparison wouldn't work if fitness was to be equal
    parent_child_sorted = [x for _, x in sorted(zip(parent_child_fitness, parent_child), key=lambda pair: pair[0])]
    new_generation = parent_child_sorted[:s]
    others = random.sample(parent_child_sorted[s:], len(parents)-s)
    new_generation.extend(others)
    return new_generation

def evolve_population(generation: list, targets: list[nx.DiGraph], selection_method: int, hyperparameters: dict[str, int| float]):
    # Evolves a generation once using either replacement (0) or elitism (1) as a selection method.

    children = []

    # ceil(N / 2) pairs give N children. Note the brackets: `N + 1 // 2` is N.
    for _ in range((len(generation) + 1) // 2):
        parent_1 = tournament_selection(generation, targets, int(hyperparameters["k"]))
        parent_2 = tournament_selection(generation, targets, int(hyperparameters["k"]))

        child_1, child_2 = crossover(parent_1, parent_2, float(hyperparameters["p_c"]))

        child_1 = mutation(child_1, float(hyperparameters["p_m"]))
        child_2 = mutation(child_2, float(hyperparameters["p_m"]))

        children.append(child_1)
        children.append(child_2)

    #Edge case with odd population sizes
    children = children[:len(generation)]
    if selection_method == 0:
        new_generation = replacement_selection(generation, children, targets)
        return new_generation
    elif selection_method == 1:
        new_generation = elitism_selection(generation, children, targets, int(hyperparameters["s"]))
        return new_generation

    
    return None

def run_evolution(starting_population: list, targets: list[nx.DiGraph], selection_method: int, hyperparameters: dict[str, int | float]):
    """One evolutionary run. `history[g]` is the population at generation `g`.

    Runs a fixed number of generations so that every run spends the same
    budget. A positive `stagnation_patience` also stops early once the best
    fitness has not improved by more than `fitness_improvement_threshold` for
    that many generations; it is 0 by default, since stopping early spends
    less than the budget the baseline is given.
    """
    generations = int(hyperparameters["generations"])
    patience = int(hyperparameters.get("stagnation_patience", 0))
    threshold = float(hyperparameters.get("fitness_improvement_threshold", 0.0))

    population_history = [starting_population]
    best_so_far = min(evaluate(x, targets) for x in starting_population)
    stagnant = 0

    for _ in range(generations):
        population = evolve_population(
            population_history[-1],
            targets,
            selection_method,
            hyperparameters,
        )
        population_history.append(population)

        best = min(evaluate(x, targets) for x in population)
        if best_so_far - best > threshold:
            stagnant = 0
        else:
            stagnant += 1
        best_so_far = min(best_so_far, best)

        if patience and stagnant >= patience:
            break

    return population_history


def random_search(targets: list[nx.DiGraph], hyperparameters: dict[str, int | float]):
    """Random-search baseline at exactly the EA's evaluation budget.

    Draws `pop_size` fresh genomes per "generation" for `generations + 1`
    generations and returns them in `run_evolution`'s history format, so both
    go through the same plotting and statistics code.
    """
    pop_size = int(hyperparameters["pop_size"])
    generations = int(hyperparameters["generations"])

    population_history = []
    for _ in range(generations + 1):
        batch = [random_tree(max_modules=NUM_OF_MODULES) for _ in range(pop_size)]
        for genome in batch:
            evaluate(genome, targets)
        population_history.append(batch)
    return population_history



# ============================================================================ #
#  6. PLOTTING
# ============================================================================ #

# evolutions is a list containing evolutions, these are lists containing generations(populations), these contain individuals
# An evolution may instead be a flat list of one fitness value per generation,
# which is what the experiment runner passes. `save_path` writes the figure to
# disk instead of opening a window; `selection_method` may be 0, 1 or a label.


def _fitness_per_generation(evolutions: list, targets, aggregate) -> list[list[float]]:
    """One fitness value per generation, from genomes or from floats."""
    curves = []
    for evo in evolutions:
        if len(evo) and isinstance(evo[0], (int, float, np.floating)):
            curves.append([float(value) for value in evo])
            continue
        curves.append([
            float(aggregate([evaluate(x, targets) for x in gen])) for gen in evo
        ])
    return curves


def _describe(selection_method: int | str) -> str:
    """Name the condition for a plot title."""
    if selection_method == 0:
        return "replacement"
    if selection_method == 1:
        return "elitism"
    return str(selection_method)


def _draw_evolution_curves(
    fitness_evolutions: list[list[float]],
    ylabel: str,
    title: str,
    save_path=None,
) -> None:
    """Mean +- std across runs per generation, with early-stop markers."""
    # rows = runs, columns = generations, short runs padded with NaN. Aggregate
    # down the columns (axis=0) to get one point per generation, not per run.
    pad = len(max(fitness_evolutions, key=len))
    array_fe = np.array([i + [np.nan]*(pad-len(i)) for i in fitness_evolutions])
    means = np.nanmean(array_fe, axis=0)
    std = np.nanstd(array_fe, axis=0)

    # mark the generation at which any early-stopping run broke off
    stamps = []
    values = []
    for run in fitness_evolutions:
        if len(run) < pad:
            stamps.append(len(run) - 1)
            values.append(means[len(run) - 1])

    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    generations = [x for x in range(len(means))]
    axis.plot(generations, means)
    axis.fill_between(generations, means+std, means-std, alpha=0.5)

    # plots the breakoff points of evolution runs
    axis.scatter(stamps, values, c="red", zorder=4)

    axis.set_xlabel('generation')
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()

    if save_path is None:
        plt.show()
        plt.close(figure)
        return
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    console.log(f"saved {save_path}")


def plot_means_evolutions(
    evolutions: list,
    targets: list[nx.DiGraph] | None,
    selection_method: int | str,
    save_path=None,
):
    # Generates a plot of means + std of multiple evolution runs in regards to the average fitness per generation
    mean_fitness_evolutions = _fitness_per_generation(evolutions, targets, np.mean)
    _draw_evolution_curves(
        mean_fitness_evolutions,
        'mean fitness (lower is better)',
        f"Average mean fitness per generation over {len(evolutions)} runs "
        f"using {_describe(selection_method)}",
        save_path,
    )
    return None


def plot_bests_evolutions(
    evolutions: list,
    targets: list[nx.DiGraph] | None,
    selection_method: int | str,
    save_path=None,
):
    # Generates a plot of means + std of multiple evolution runs in regards to the best fitness per generation
    fitness_evolutions = _fitness_per_generation(evolutions, targets, min)
    _draw_evolution_curves(
        fitness_evolutions,
        'best fitness (lower is better)',
        f"Average best fitness per generation over {len(evolutions)} runs "
        f"using {_describe(selection_method)}",
        save_path,
    )
    return None

PLOT_STYLE: dict[str, dict[str, str]] = {
    "replacement": {"color": "#1f77b4", "label": "EA - generational replacement"},
    "elitism": {"color": "#d62728", "label": "EA - elitism"},
    "random": {"color": "#7f7f7f", "label": "random search (same budget)"},
}


def safe_name(label: str) -> str:
    """Filename-safe form of a curve label ("elitism s=2" -> "elitism_s2")."""
    return label.replace(" ", "_").replace("=", "")


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


def evolution_curves(frame: pd.DataFrame, condition: str, column: str) -> list[list[float]]:
    """One list of per-generation fitness values per run, for one condition.

    This is the shape `plot_means_evolutions` and `plot_bests_evolutions` take:
    a list of runs, each already reduced to one number per generation. Runs are
    read back from the saved CSV, so a run that stopped early is simply shorter
    than the rest, which is exactly what their NaN padding is there to handle.
    """
    subset = frame[frame["condition"] == condition]
    curves = []
    for _, run in subset.groupby("seed"):
        run = run.sort_values("generation")
        curves.append(run[column].astype(float).tolist())
    return curves


def plot_each_condition(frame: pd.DataFrame, out_dir: Path) -> None:
    """Per-condition figures, drawn by the plotting functions in A1_2026.

    The overlay figures above put every condition on one axis to answer which
    is better; these are the single-condition view, one file per condition per
    metric.
    """
    for condition in labels_in(frame):
        plot_bests_evolutions(
            evolution_curves(frame, condition, "best"),
            None,  # fitness is already computed, so no targets are needed
            condition,
            save_path=out_dir / f"fig_best_{safe_name(condition)}.png",
        )
        plot_means_evolutions(
            evolution_curves(frame, condition, "mean"),
            None,
            condition,
            save_path=out_dir / f"fig_mean_{safe_name(condition)}.png",
        )


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
    out_dir = DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    targets = load_targets()
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
    out_dir = DATA / tag
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
    plot_each_condition(frame, out_dir)
    plot_bodies(tag)

# ============================================================================ #
#  7. RUNNING AN EXPERIMENT
# ============================================================================ #

@dataclass
class RunResult:
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
    RNG = np.random.default_rng(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError: 
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
    targets = load_targets()
    selection_method = CONDITIONS[condition]

    _seed_everything(seed)
    reset_evaluation_budget()
    started = time.perf_counter()

    if selection_method is None:
        history = random_search(targets, hyperparameters)
    else:
        population = generate_population(int(hyperparameters["pop_size"]))
        history = run_evolution(
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
        evaluations=evaluations_used(),
        wall_seconds=wall,
    )

    champion: TreeGenome | None = None
    champion_fitness = float("inf")
    for generation in history:
        fitness = np.array([evaluate(x, targets) for x in generation])
        result.best_per_generation.append(float(fitness.min()))
        result.mean_per_generation.append(float(fitness.mean()))
        result.worst_per_generation.append(float(fitness.max()))
        unique = len({genome_signature(x) for x in generation})
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
    out_dir = DATA / tag
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
            path = genome_dir / f"{safe_name(result.label)}_seed{result.seed}.json"
            path.write_text(json.dumps(result.best_genome, indent=2))

    metadata = {
        "tag": tag,
        "encoding": GENOTYPE,
        "num_of_modules": NUM_OF_MODULES,
        "targets": str(TARGET_DIR),
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

def champion_path(out_dir: Path, label: str, seed: int) -> Path:
    """Where `save_experiment` put the best genome of one run."""
    return out_dir / "best_genomes" / f"{safe_name(label)}_seed{seed}.json"


def report(tag: str) -> None:
    """Print - and save - the numbers the Results section needs."""
    out_dir = DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    metadata = json.loads((out_dir / "meta.json").read_text())

    targets = load_targets()
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
    out_dir = DATA / tag
    summary = pd.read_csv(out_dir / "summary.csv")
    for condition in labels_in(summary):
        subset = summary[summary["condition"] == condition]
        if subset.empty:
            continue
        best_row = subset.loc[subset["final_best"].idxmin()]
        path = champion_path(out_dir, condition, int(best_row["seed"]))
        genome = TreeGenome.from_dict(json.loads(path.read_text()))
        show_body(
            genome.to_networkx(),
            "frame",
            file_name=f"{tag}_champion_{safe_name(condition)}",
        )

def command_final(args: argparse.Namespace) -> None:
    """The experiment reported in the paper."""
    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
    hyperparameters["pop_size"] = args.pop_size
    hyperparameters["generations"] = args.generations

    per_condition = load_config(args.config)
    if per_condition:
        print(f"using parameters: {args.config}")

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

    out_dir = DATA / args.tag
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

    (out_dir / "tuned_parameters.json").write_text(
        json.dumps({"per_condition": best}, indent=2),
    )
    print()
    print("chosen configuration - paste over TUNED_PARAMETERS to adopt it:")
    print(f"TUNED_PARAMETERS = {json.dumps(best, indent=4)}")
    print()
    print("then: python A1_2026.py final --config tuned")


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
    per_condition = load_config(args.config)
    if per_condition:
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
        help="'shared', 'tuned', or a path to a JSON file",
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
        help="'shared', 'tuned', or a path to a JSON file",
    )
    sweep.set_defaults(func=command_sweep, default_tag="sweep_elitism")

    plot = subparsers.add_parser("plot", help="redraw figures from saved data")
    add_common(plot)
    plot.set_defaults(func=command_plot, default_tag="final")

    return parser

# ============================================================================ #
#  8. ENTRY POINT
# ============================================================================ #

def run_experiment_cli() -> None:
    """Dispatch `python A1_2026.py <sub-command>` to the experiment harness."""
    args = build_parser().parse_args()
    if args.tag is None:
        args.tag = args.default_tag
    args.func(args)

def main() -> None:
    """Score one randomly-sampled body against the target set."""
    targets = load_targets()

    console.log(f"encoding      : {GENOTYPE}")
    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
        "target sizes  : "
        + ", ".join(str(t.number_of_nodes()) for t in targets),
    )

    # Small on purpose: this is a smoke test, not the experiment.
    hyperparameters = {
        "pop_size": 10,
        "generations": 5,
        "k": 3,                  # tournament size
        "p_c": 0.5,              # crossover probability
        "p_m": 0.5,              # mutation probability
        "s": 2,                  # elites carried over (elitism only)
        "stagnation_patience": 0,
        "fitness_improvement_threshold": 0.0,
    }

    # How far apart are the targets from each other? Your fitness cannot go
    # below the best possible compromise, and this is the clue to where that is.
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1 :]
    ]
    console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")

    # --- One random body --------------------------------------------------- #
    body = random_body(GENOTYPE, NUM_OF_MODULES)
    fitness = fitness_function(body, targets)

    console.log("")
    console.log(f"random body   : {body.number_of_nodes()} modules")
    console.log(
        "per-target    : "
        + ", ".join(f"{d:.1f}" for d in distances_to_targets(body, targets)),
    )
    console.log(f"fitness       : {fitness:.4f}   (lower is better)")

    # --- Smoke test: one short run of each condition ------------------------ #
    console.log("")
    console.log("smoke test    : 10 individuals, 5 generations, 1 seed each")

    histories = {}
    for label, selection_method in (("replacement", 0), ("elitism", 1)):
        random.seed(SEED)
        reset_evaluation_budget()
        population = generate_population(int(hyperparameters["pop_size"]))
        history = run_evolution(population, targets, selection_method, hyperparameters)
        histories[label] = history
        best = min(evaluate(x, targets) for x in history[-1])
        console.log(
            f"  {label:<12}: best {best:.3f} after {len(history) - 1} gens "
            f"({evaluations_used()} evaluations)",
        )

    random.seed(SEED)
    reset_evaluation_budget()
    history = random_search(targets, hyperparameters)
    histories["random search"] = history
    best = min(evaluate(x, targets) for gen in history for x in gen)
    console.log(
        f"  {'random':<12}: best {best:.3f} at the same budget "
        f"({evaluations_used()} evaluations)",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_experiment_cli()
    else:
        main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above samples ONE body at random and scores it. Your task is to
# replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# For morphology-specific evolution with the tree encoding, read:
#
#     examples/c_genotypes/1_body_evolution_tree.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- GENOTYPE - DEPENDENT "GOTCHA"S -------------------------
#
#   TREE: VARIABLE LENGTH - Tree genotypes grow; without pressure against it they
#     will grow forever, and every extra module costs an edit.
#   NDE: REPRODUCIBILITY   If you're using "nde": construct
#     `NeuralDevelopmentalEncoding` ONCE for your whole run, never per
#     individual or per generation, AND call `torch.manual_seed(...)` in
#     addition to the numpy/random seeds. See the two "IMPORTANT" notes under
#     "nde" in THE GENOTYPE CONTRACT above - getting either wrong means your
#     your runs won't reproduce cleanly.
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing - repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline, a good standard is at least a random search.
#   Keep the encoding, module budget and target set identical across
#     everything you compare, change one thing at a time.
#
# ============================================================================ #
