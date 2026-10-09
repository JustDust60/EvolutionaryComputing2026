"""Shared contract. Person 1 owns EA TODOs; Person 2 owns simulation TODOs.

This is a starter, not a completed EA. Do not modify src/ariel.
Genomes are ordinary JSON-compatible dictionaries, not custom classes.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import copy
import csv
import json

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population, set_seed


@dataclass
class Settings:
    """Provisional settings: agree and freeze after the pilot, before final runs."""

    seed: int = 0
    population_size: int = 20
    offspring_size: int = 20
    evaluation_budget: int = 1020  # initial 20 + 50 batches of 20
    tournament_size: int = 3
    mutation_sigma: float = 0.1
    initial_weight_sigma: float = 0.5
    small_hidden_size: int = 2
    large_hidden_size: int = 8
    max_hidden_size: int = 8
    add_node_probability: float = 0.1
    simulation_seconds: float = 15.0
    target_x: float = 2.0
    target_y: float = 0.0
    rhythm_hz: float = 1.0
    body_name: str = "gecko"  # explicitly use john_set.gecko(), not another library
    control_mode: str = "direct"


# Every genotype has this schema (arrays must be saved as nested lists):
# {"input_size": int, "output_size": int, "hidden_size": int,
#  "w_in": [[float, ...], ...], "w_out": [[float, ...], ...]}
# w_in shape: (input_size, hidden_size)
# w_out shape: (hidden_size, output_size)
# Inputs include one constant bias feature, so no extra bias arrays are needed.


def clone_genome(genome: dict) -> dict:
    """Provided: offspring must not share mutable weights with their parent."""
    return copy.deepcopy(genome)


def validate_genome(genome: dict) -> None:
    """Provided: catch contract mistakes before launching a simulation."""
    ni, nh, no = (genome[k] for k in ("input_size", "hidden_size", "output_size"))
    if min(ni, nh, no) < 1:
        raise ValueError("All network dimensions must be positive")
    wi, wo = np.asarray(genome["w_in"]), np.asarray(genome["w_out"])
    if wi.shape != (ni, nh) or wo.shape != (nh, no):
        raise ValueError("Weight shapes do not match the genome dimensions")
    if not np.isfinite(wi).all() or not np.isfinite(wo).all():
        raise ValueError("Genome contains non-finite weights")


def forward(genome: dict, observations: np.ndarray) -> np.ndarray:
    """Provided: same tanh network and action scaling for every condition.

    For the final evaluator, convert weights to arrays once per evaluation,
    rather than once per physics step, if profiling shows this is expensive.
    """
    wi, wo = np.asarray(genome["w_in"]), np.asarray(genome["w_out"])
    inputs = np.asarray(observations, dtype=float)
    if inputs.shape != (genome["input_size"],):
        raise ValueError("Observation length does not match input_size")
    actions = np.tanh(np.tanh(inputs @ wi) @ wo) * (np.pi / 2)
    if not np.isfinite(actions).all():
        raise FloatingPointError("Network produced non-finite actions")
    return actions


def network_size(genome: dict) -> tuple[int, int]:
    """Provided: hidden neurons and connection count; zero-weight edges count."""
    h = genome["hidden_size"]
    return h, genome["input_size"] * h + h * genome["output_size"]


def save_json(path: Path, value: dict) -> None:
    """Provided: saves settings or the best genotype; refuses non-finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_metrics(path: Path, rows: list[dict]) -> None:
    """Provided: agreed columns, one row for generation 0 and every later batch.

    best_distance is the best distance seen so far in that independent run.
    population_mean_distance describes survivors, not the across-seed spread.
    mean_hidden/mean_connections describe survivors. best_hidden/best_connections
    describe the best-so-far genotype, including if it is no longer alive.
    Random-search generation is a batch index using the same batch size as EA.
    """
    columns = ["condition", "seed", "generation", "evaluations", "best_distance",
               "population_mean_distance", "mean_hidden", "mean_connections",
               "best_hidden", "best_connections", "elapsed_seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_observations(model, data, settings: Settings) -> np.ndarray:
    """TODO PERSON 2: implement once and use in ALL conditions.

    Recommended features: hinge positions and velocities; target displacement
    in the robot's local horizontal frame; sin/cos(2*pi*rhythm_hz*data.time);
    and a final constant 1.0 bias. Fix normalization/clipping for all runs.
    MuJoCo qpos[0:3] is core position and qpos[3:7] is its quaternion.
    Get hinge indices from the compiled model; do not assume qpos equals nu.
    Input size must be constant despite hidden-layer growth.
    """
    raise NotImplementedError("Person 2: build_observations")


def prepare_simulation(settings: Settings):
    """TODO PERSON 2: return (model, input_size, output_size).

    Adapt assignments/assignment_2/A2_template_2026.py. Use SimpleFlatWorld and
    the SAME john_set.gecko body, spawn [0,0,0.1], compile once per run.
    Clear mj.set_mjcb_control(None) before compiling/resetting.
    Derive input_size from build_observations and output_size from model.nu.
    A new MjData/reset for each individual is sufficient; no parallel threads.
    """
    raise NotImplementedError("Person 2: prepare_simulation")


def evaluate_genome(genome: dict, model, settings: Settings) -> float:
    """TODO PERSON 2: return final Euclidean XY distance; LOWER IS BETTER.

    Validate genotype. Clear the global callback; create/reset MjData; forward
    the model; install a callback using build_observations and forward().
    Run headlessly for simulation_seconds, reading model.nu action values.
    Use the same direct/delta command policy everywhere. Clear callback in a
    finally block, even if evaluation raises. Return norm(final_xy-target_xy).
    Count every simulation attempt against budget. Define a fixed finite worst
    score for numerical simulation failures, log them, and do not hide coding
    errors by catching every exception. Never add a complexity penalty here.
    """
    raise NotImplementedError("Person 2: evaluate_genome")


def tournament_parent(population: Population, settings: Settings,
                      rng: np.random.Generator) -> Individual:
    """TODO PERSON 1: sample tournament_size alive/evaluated individuals.

    Return the one with the LOWEST fitness. Randomize exact ties consistently.
    Return a parent reference; cloning happens when constructing offspring.
    """
    raise NotImplementedError("Person 1: tournament_parent")


def select_survivors(population: Population, settings: Settings,
                     rng: np.random.Generator) -> Population:
    """TODO PERSON 1: (mu+lambda) elitist survivor selection.

    Shuffle ties, keep the population_size lowest-fitness candidates alive,
    mark others alive=False. Preserve their IDs/history for ariel.ec logging.
    Return the whole Population with alive flags; subsequent operators must
    use .alive. Do not preferentially retain smaller networks on ties.
    """
    raise NotImplementedError("Person 1: select_survivors")


def run_ea(condition: str, settings: Settings, initialize: Callable,
           mutate: Callable, output_dir: Path) -> dict:
    """TODO PERSON 1: shared EA loop for fixed AND growing controllers.

    initialize(input_size, output_size, rng) -> genome
    mutate(parent_genome, rng) -> NEW genome (never changes the parent)

    Implementation checklist:
    1. Seed np.random.default_rng(settings.seed) and ariel.ec.set_seed.
    2. Create output_dir exclusively; refuse to overwrite an existing run.
    3. Prepare the model once, initialize and evaluate mu Individuals, count
       all mu evaluations, and record generation 0 before offspring exist.
    4. Use EA/EAOperation/Individual/Population from ariel.ec. Operations:
       produce_offspring -> evaluate_unevaluated -> select_survivors -> log.
       Nested operation signatures must use actual Population annotations
       (avoid `from __future__ import annotations` with this EAOperation API).
    5. Produce lambda new Individuals from tournament-selected parents,
       clone through mutate(), assign genotype, and requires_eval=True.
       Set is_maximisation=False and db_handling='halt', with a unique DB.
    6. Evaluate only unevaluated alive candidates. Cached surviving parents
       do not consume extra evaluations. Both EAs use identical operators
       except the initialize/mutate callbacks. No crossover or speciation.
    7. Stop at the common budget, including initialization. Require
       (budget-mu)%lambda==0. Pilot plateau guides choosing this common budget;
       do not stop individual final runs early on a plateau.
    8. Save config.json (asdict(settings) + condition), metrics.csv, best.json
       (fitness + genome), database.db, and return a summary dictionary.
       Keep best-so-far separately; ariel.ec histories may include dead rows.
    """
    raise NotImplementedError("Person 1: run_ea")
