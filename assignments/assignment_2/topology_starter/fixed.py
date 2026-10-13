"""Fixed networks and the random-search baseline."""

from dataclasses import asdict
from pathlib import Path
import numpy as np
import time

from ariel.ec import Individual, Population, set_seed
from common import (ACTIVATIONS, Settings, clone_genome, evaluate_genome,
                    network_size, prepare_simulation, run_ea, save_json,
                    save_metrics, save_progress, validate_genome)


def initialize_fixed(input_size: int, output_size: int, hidden_size: int,
                     settings: Settings, rng: np.random.Generator) -> dict:
    activations = []
    for _ in range(hidden_size):
        activations.append("tanh")
    genome = {"input_size": input_size, "output_size": output_size,
              "hidden_size": hidden_size, "activations": activations,
              "w_in": rng.normal(0.0, settings.initial_weight_sigma,
                                 (input_size, hidden_size)).tolist(),
              "w_out": rng.normal(0.0, settings.initial_weight_sigma,
                                  (hidden_size, output_size)).tolist()}
    validate_genome(genome)
    return genome


def mutate_weights(genome: dict, settings: Settings,
                   rng: np.random.Generator) -> dict:
    validate_genome(genome)
    child = clone_genome(genome)
    count = child["input_size"] * child["hidden_size"]
    count += child["hidden_size"] * child["output_size"]
    for name in ("w_in", "w_out"):
        for row in child[name]:
            for i in range(len(row)):
                if rng.random() < 1.0 / count:
                    row[i] += float(rng.normal(0.0, settings.mutation_sigma))
    validate_genome(child)
    return child


def mutate_activations(genome: dict, settings: Settings,
                       rng: np.random.Generator) -> dict:
    child = clone_genome(genome)
    for i in range(child["hidden_size"]):
        if rng.random() < settings.activation_mutation_probability:
            choices = list(ACTIVATIONS)
            choices.remove(child["activations"][i])
            child["activations"][i] = str(rng.choice(choices))
    validate_genome(child)
    return child


def mutate_fixed(genome: dict, settings: Settings,
                 rng: np.random.Generator) -> dict:
    child = mutate_weights(genome, settings, rng)
    return mutate_activations(child, settings, rng)


def run_fixed(condition: str, settings: Settings, output_dir: Path) -> dict:
    sizes = {"fixed_small": settings.small_hidden_size,
             "fixed_large": settings.large_hidden_size}
    if condition not in sizes:
        raise ValueError(f"Unknown fixed condition: {condition}")
    h = sizes[condition]
    return run_ea(condition, settings,
                  lambda ni, no, rng: initialize_fixed(ni, no, h, settings, rng),
                  lambda g, rng: mutate_fixed(g, settings, rng), output_dir)


def run_random_search(settings: Settings, output_dir: Path) -> dict:
    if min(settings.population_size, settings.offspring_size) < 1:
        raise ValueError("Population and offspring sizes must be positive")
    remaining = settings.evaluation_budget - settings.population_size
    if remaining < 0 or remaining % settings.offspring_size:
        raise ValueError("Budget must include initialization and whole offspring batches")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(settings.seed)
    set_seed(settings.seed)
    condition = "random_small"
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
                     "generation": len(rows),
                     "evaluations": evaluations, "best_distance": best["fitness"],
                     "population_mean_distance": float(np.mean(distances)),
                     "mean_hidden": float(np.mean(hidden_sizes)),
                     "mean_connections": float(np.mean(connections)),
                     "best_hidden": best_hidden, "best_connections": best_connections,
                     "elapsed_seconds": time.perf_counter() - started})
        save_progress(output_dir, settings, rows[-1])
        return population

    config = asdict(settings)
    config["condition"] = condition
    save_json(output_dir / "config.json", config)
    while evaluations < settings.evaluation_budget:
        batch_size = settings.population_size if evaluations == 0 else settings.offspring_size

        indivs = Population([])
        for _ in range(batch_size):
            individual = Individual()
            individual.genotype = initialize_fixed(input_size, output_size,
                                                   settings.small_hidden_size, settings, rng)
            individual.requires_eval = True
            indivs.append(individual)

        evaluate_population(indivs)
        record_metrics(indivs)

    save_metrics(output_dir / "metrics.csv", rows)
    save_json(output_dir / "best.json", best)
    return {"condition": condition, "seed": settings.seed,
            "evaluations": evaluations, "best_distance": best["fitness"],
            "elapsed_seconds": time.perf_counter() - started}
