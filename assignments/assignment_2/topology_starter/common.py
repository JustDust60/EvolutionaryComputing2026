"""Shared simulation, controller and evolutionary search."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import copy
import csv
import json
import logging
import time

import mujoco as mj
import numpy as np
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.ec import EA, EAOperation, Individual, Population, set_seed
from ariel.simulation.environments import SimpleFlatWorld


ACTIVATIONS = ("tanh", "sigmoid", "gaussian", "sine")


@dataclass
class Settings:
    """Shared experiment settings."""

    seed: int = 0
    population_size: int = 20
    offspring_size: int = 20
    evaluation_budget: int = 1020
    tournament_size: int = 3
    mutation_sigma: float = 0.1
    initial_weight_sigma: float = 0.5
    activation_mutation_probability: float = 0.1
    small_hidden_size: int = 2
    large_hidden_size: int = 8
    max_hidden_size: int = 8
    add_node_probability: float = 0.1
    simulation_seconds: float = 15.0
    target_x: float = 2.0
    target_y: float = 0.0
    rhythm_hz: float = 1.0
    body_name: str = "gecko"
    control_mode: str = "direct"


def clone_genome(genome: dict) -> dict:
    return copy.deepcopy(genome)


def validate_genome(genome: dict) -> None:
    ni = genome["input_size"]
    nh = genome["hidden_size"]
    no = genome["output_size"]
    if min(ni, nh, no) < 1:
        raise ValueError("All network dimensions must be positive")
    wi = np.asarray(genome["w_in"])
    wo = np.asarray(genome["w_out"])
    if wi.shape != (ni, nh) or wo.shape != (nh, no):
        raise ValueError("Weight shapes do not match the genome dimensions")
    if not np.isfinite(wi).all() or not np.isfinite(wo).all():
        raise ValueError("Genome contains non-finite weights")
    if len(genome["activations"]) != nh:
        raise ValueError("Each hidden neuron needs an activation function")
    for activation in genome["activations"]:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation function: {activation}")


def forward(genome: dict, observations: np.ndarray) -> np.ndarray:
    wi = np.asarray(genome["w_in"])
    wo = np.asarray(genome["w_out"])
    inputs = np.asarray(observations, dtype=float)
    if inputs.shape != (genome["input_size"],):
        raise ValueError("Observation length does not match input_size")
    values = inputs @ wi
    hidden = np.empty(genome["hidden_size"])
    for i in range(genome["hidden_size"]):
        activation = genome["activations"][i]
        value = values[i]
        if activation == "tanh":
            hidden[i] = np.tanh(value)
        elif activation == "sigmoid":
            hidden[i] = 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))
        elif activation == "gaussian":
            hidden[i] = np.exp(-value * value)
        elif activation == "sine":
            hidden[i] = np.sin(value)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    actions = np.tanh(hidden @ wo) * (np.pi / 2)
    if not np.isfinite(actions).all():
        raise FloatingPointError("Network produced non-finite actions")
    return actions


def network_size(genome: dict) -> tuple[int, int]:
    h = genome["hidden_size"]
    return h, genome["input_size"] * h + h * genome["output_size"]


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_metrics(path: Path, rows: list[dict]) -> None:
    columns = ["condition", "seed", "generation", "evaluations", "best_distance",
               "population_mean_distance", "mean_hidden", "mean_connections",
               "best_hidden", "best_connections", "elapsed_seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_progress(output_dir: Path, settings: Settings, metrics: dict) -> None:
    """Publish the last completed evaluation batch for local monitors."""
    temporary = output_dir / "progress.json.tmp"
    save_json(temporary, {**metrics, "evaluation_budget": settings.evaluation_budget})
    temporary.replace(output_dir / "progress.json")


def build_observations(model, data, settings: Settings) -> np.ndarray:
    observations = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE:
            position = data.qpos[model.jnt_qposadr[joint]]
            velocity = data.qvel[model.jnt_dofadr[joint]]
            observations.append(np.clip(position / (np.pi / 2), -1, 1))
            observations.append(np.clip(velocity / 10.0, -1, 1))

    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx = settings.target_x - data.qpos[0]
    dy = settings.target_y - data.qpos[1]
    local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
    local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
    observations.append(np.clip(local_x / 2.0, -1, 1))
    observations.append(np.clip(local_y / 2.0, -1, 1))
    phase = 2 * np.pi * settings.rhythm_hz * data.time
    observations.append(np.sin(phase))
    observations.append(np.cos(phase))
    observations.append(1.0)
    return np.asarray(observations, dtype=float)


def prepare_simulation(settings: Settings):
    mj.set_mjcb_control(None)
    if settings.body_name != "gecko":
        raise ValueError("This experiment uses john_set.gecko")
    world = SimpleFlatWorld()
    robot = john_set.gecko()
    world.spawn(robot.spec, position=[0.0, 0.0, 0.1],
                correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    input_size = len(build_observations(model, data, settings))
    return model, input_size, model.nu


def evaluate_genome(genome: dict, model, settings: Settings) -> float:
    mj.set_mjcb_control(None)
    validate_genome(genome)
    if settings.control_mode not in ("direct", "delta"):
        raise ValueError("Control mode must be direct or delta")
    if not np.isfinite(settings.simulation_seconds) or settings.simulation_seconds <= 0:
        raise ValueError("Simulation duration must be finite and positive")
    if genome["output_size"] != model.nu:
        raise ValueError("Output size does not match the robot")
    controller_genome = dict(genome)
    controller_genome["w_in"] = np.asarray(genome["w_in"], dtype=float)
    controller_genome["w_out"] = np.asarray(genome["w_out"], dtype=float)

    def control(model, data):
        observations = build_observations(model, data, settings)
        actions = forward(controller_genome, observations)
        if settings.control_mode == "direct":
            data.ctrl[:] = actions
        else:
            data.ctrl[:] = np.clip(data.ctrl + actions * 0.05, -np.pi / 2, np.pi / 2)

    numerical_warnings = (mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL,
                          mj.mjtWarning.mjWARN_BADQACC, mj.mjtWarning.mjWARN_BADCTRL)
    try:
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        if genome["input_size"] != len(build_observations(model, data, settings)):
            raise ValueError("Input size does not match the observations")
        mj.set_mjcb_control(control)
        steps = int(np.ceil(settings.simulation_seconds / model.opt.timestep))
        for _ in range(steps):
            mj.mj_step(model, data)
            for warning in numerical_warnings:
                if data.warning[warning].number > 0:
                    raise FloatingPointError(f"MuJoCo numerical warning: {warning.name}")
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("Non-finite simulation state")
        target = np.array([settings.target_x, settings.target_y])
        distance = float(np.linalg.norm(data.qpos[:2] - target))
        if not np.isfinite(distance):
            raise FloatingPointError("Non-finite final distance")
        return distance
    except FloatingPointError as error:
        logging.warning("Simulation failed: %s", error)
        return 1_000_000.0
    finally:
        mj.set_mjcb_control(None)


def tournament_parent(population: Population, settings: Settings,
                      rng: np.random.Generator) -> Individual:
    candidates = []
    for individual in population.alive:
        if not individual.requires_eval:
            candidates.append(individual)
    if not candidates or settings.tournament_size < 1:
        raise ValueError("A tournament needs evaluated parents and a positive size")
    count = min(settings.tournament_size, len(candidates))
    indices = rng.choice(len(candidates), size=count, replace=False)
    winner = candidates[int(indices[0])]
    for index in indices[1:]:
        candidate = candidates[int(index)]
        if candidate.fitness < winner.fitness:
            winner = candidate
    return winner


def select_survivors(population: Population, settings: Settings,
                     rng: np.random.Generator) -> Population:
    candidates = list(population.alive)
    rng.shuffle(candidates)
    candidates.sort(key=lambda individual: individual.fitness)
    for i, individual in enumerate(candidates):
        individual.alive = i < settings.population_size
    return population


def run_ea(condition: str, settings: Settings, initialize: Callable,
           mutate: Callable, output_dir: Path) -> dict:
    if min(settings.population_size, settings.offspring_size, settings.tournament_size) < 1:
        raise ValueError("Population, offspring and tournament sizes must be positive")
    remaining = settings.evaluation_budget - settings.population_size
    if remaining < 0 or remaining % settings.offspring_size:
        raise ValueError("Budget must include initialization and whole offspring batches")
    if not 0 <= settings.add_node_probability <= 1:
        raise ValueError("Node addition probability must be between zero and one")
    if not 0 <= settings.activation_mutation_probability <= 1:
        raise ValueError("Activation mutation probability must be between zero and one")
    if condition == "growing" and settings.small_hidden_size > settings.max_hidden_size:
        raise ValueError("Initial hidden size exceeds the growth cap")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(settings.seed)
    set_seed(settings.seed)
    started = time.perf_counter()
    model, input_size, output_size = prepare_simulation(settings)
    evaluations = 0
    best = None
    rows = []

    def evaluate_population(population: Population) -> Population:
        nonlocal evaluations, best
        for individual in population.alive:
            if individual.requires_eval:
                evaluations += 1
                individual.fitness = evaluate_genome(individual.genotype, model, settings)
                if best is None or individual.fitness < best["fitness"]:
                    best = {"fitness": individual.fitness,
                            "genome": clone_genome(individual.genotype)}
        return population

    def produce_offspring(population: Population) -> Population:
        parents = population.alive
        for _ in range(settings.offspring_size):
            parent = tournament_parent(parents, settings, rng)
            child = Individual()
            child.genotype = mutate(parent.genotype, rng)
            child.requires_eval = True
            population.append(child)
        return population

    def survivors(population: Population) -> Population:
        return select_survivors(population, settings, rng)

    def record_metrics(population: Population) -> Population:
        distances = []
        hidden_sizes = []
        connections = []
        for individual in population.alive:
            hidden, edges = network_size(individual.genotype)
            distances.append(individual.fitness)
            hidden_sizes.append(hidden)
            connections.append(edges)
        best_hidden, best_connections = network_size(best["genome"])
        rows.append({"condition": condition, "seed": settings.seed,
                     "generation": (evaluations - settings.population_size) // settings.offspring_size,
                     "evaluations": evaluations, "best_distance": best["fitness"],
                     "population_mean_distance": float(np.mean(distances)),
                     "mean_hidden": float(np.mean(hidden_sizes)),
                     "mean_connections": float(np.mean(connections)),
                     "best_hidden": best_hidden, "best_connections": best_connections,
                     "elapsed_seconds": time.perf_counter() - started})
        save_progress(output_dir, settings, rows[-1])
        return population

    initial = Population([])
    for _ in range(settings.population_size):
        individual = Individual()
        individual.genotype = initialize(input_size, output_size, rng)
        initial.append(individual)
    evaluate_population(initial)
    record_metrics(initial)
    config = asdict(settings)
    config["condition"] = condition
    save_json(output_dir / "config.json", config)
    operations = [EAOperation(produce_offspring), EAOperation(evaluate_population),
                  EAOperation(survivors), EAOperation(record_metrics)]
    ea = EA(initial, operations, num_steps=remaining // settings.offspring_size,
            first_generation_id=0, is_maximisation=False,
            db_file_path=output_dir / "database.db", db_handling="halt", quiet=True)
    try:
        ea.run()
    finally:
        ea.engine.dispose()
        mj.set_mjcb_control(None)
    save_metrics(output_dir / "metrics.csv", rows)
    save_json(output_dir / "best.json", best)
    return {"condition": condition, "seed": settings.seed,
            "evaluations": evaluations, "best_distance": best["fitness"],
            "best_hidden": best["genome"]["hidden_size"],
            "elapsed_seconds": time.perf_counter() - started}
