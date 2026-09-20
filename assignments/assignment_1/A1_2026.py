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

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

import copy
import matplotlib.pyplot as plt

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
#
# EVALUATION BUDGET
# -----------------
# Every genome is scored exactly once and the result memoised. This matters for
# the comparison in this assignment: elitism carries survivors over unchanged,
# and tournament selection inspects the same parents repeatedly, so re-running
# the tree edit distance on them would make "number of fitness evaluations"
# depend on the selection scheme instead of on the search effort. With the cache
# in place, one run costs exactly pop_size * (generations + 1) evaluations
# regardless of the scheme - which is the budget the random-search baseline is
# given too.
#
# ============================================================================ #

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

    # One pair of parents produces one pair of children, so a population of
    # N needs ceil(N / 2) pairs. (Note the brackets: `N + 1 // 2` is `N + 0`,
    # which built twice as many children as the population can hold.)
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
    """Run one independent evolutionary run and return its population history.

    `history[g]` is the population at generation `g`; `history[0]` is
    `starting_population`. The run lasts a FIXED number of generations, so
    every run in an experiment spends the same evaluation budget and the
    per-generation curves of different runs line up on a shared x-axis.

    Setting `stagnation_patience` to a positive value additionally stops a run
    early once the best fitness has failed to improve by more than
    `fitness_improvement_threshold` for that many consecutive generations. It
    is 0 (disabled) by default, because an early stop spends less than the
    nominal budget and makes the comparison against random search unfair.
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
    """Random-search baseline, given exactly the EA's evaluation budget.

    Draws `pop_size` fresh random genomes per "generation" for
    `generations + 1` generations - the same number of genomes the EA creates
    and scores - and returns them in the same history format as
    `run_evolution`, so both go through the same plotting and statistics code.

    This is the control the assignment asks for: it says how much of the EA's
    progress comes from *search*, rather than from drawing several thousand
    random bodies and keeping the luckiest one.
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
#
# Both plotting functions also accept an evolution that is already a flat list
# of one fitness value per generation. The experiment harness needs that: it
# runs in worker processes and throws populations away as soon as it has
# scored them, because shipping thousands of genomes back per run costs far
# more than shipping the numbers they were reduced to.
#
# Pass `save_path` to write the figure to disk instead of opening a window;
# `selection_method` may be 0, 1, or a label string for anything else.


def _fitness_per_generation(evolutions: list, targets, aggregate) -> list[list[float]]:
    """Reduce each run to one fitness value per generation.

    Accepts either populations of genomes (reduced with `aggregate`) or a run
    that is already a list of floats, which is passed through unchanged.
    """
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
    # rows = runs, columns = generations. Runs that stopped early are
    # padded with NaN, so aggregate DOWN THE COLUMNS (axis=0): that is
    # "across runs, per generation". axis=1 averages each run with
    # itself and yields one point per run instead of one per generation.
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


# ============================================================================ #
#  7. ENTRY POINT
# ============================================================================ #


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

    # Default hyperparameters. The final experiment overrides these from
    # experiment_runner.py - see DEFAULT_HYPERPARAMETERS there, which is the
    # single source of truth for the numbers reported in the paper.
    hyperparameters = {
        "pop_size": 10,          # tiny: this is a smoke test, not an experiment
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
    # The real experiment (5+ independent runs per condition, statistics and
    # the report figures) lives in experiment_runner.py:
    #
    #     python experiment_runner.py final
    #
    # What follows is only the "test your algorithm first using a small
    # population for few generations" check from the assignment tips.
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
