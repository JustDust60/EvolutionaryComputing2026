"""PERSON 1: fixed networks, random baseline, and shared EA/selection TODOs."""

from dataclasses import asdict
from pathlib import Path
import numpy as np

from common import Settings, clone_genome, run_ea, validate_genome


def initialize_fixed(input_size: int, output_size: int, hidden_size: int,
                     settings: Settings, rng: np.random.Generator) -> dict:
    """TODO PERSON 1: return a dense, one-hidden-layer genome.

    w_in shape (input_size, hidden_size), w_out (hidden_size, output_size).
    Draw weights from N(0, initial_weight_sigma); convert arrays to .tolist().
    Use common.py's exact schema and validate_genome before returning.
    The growing condition imports this initializer for identical starts.
    """
    raise NotImplementedError("Person 1: initialize_fixed")


def mutate_weights(genome: dict, settings: Settings,
                   rng: np.random.Generator) -> dict:
    """TODO PERSON 1: deep-copy and perturb weights; never change topology.

    Each weight independently changes with p=1/number_of_weights. Add
    N(0, mutation_sigma) to selected weights. This gives the SAME expected
    number of changes (one) across differently sized networks. An unchanged
    offspring is allowed; do not force an extra change in only one condition.
    Return JSON lists. Person 2 MUST reuse this function for weight mutation.
    """
    raise NotImplementedError("Person 1: mutate_weights")


def run_fixed(condition: str, settings: Settings, output_dir: Path) -> dict:
    """Provided: fixed_small/fixed_large share the same EA as growing."""
    sizes = {"fixed_small": settings.small_hidden_size,
             "fixed_large": settings.large_hidden_size}
    if condition not in sizes:
        raise ValueError(f"Unknown fixed condition: {condition}")
    h = sizes[condition]
    return run_ea(condition, settings,
                  lambda ni, no, rng: initialize_fixed(ni, no, h, settings, rng),
                  lambda g, rng: mutate_weights(g, settings, rng), output_dir)


def run_random_search(settings: Settings, output_dir: Path) -> dict:
    """TODO PERSON 1: random search of the SAME fixed_small representation.

    Use the shared simulation and initialization distribution. Independently
    sample and evaluate exactly evaluation_budget genomes; no selection or
    mutation. Record best-so-far after the first mu evaluations and every
    lambda thereafter, using common.py's CSV schema and identical batches.
    Here population_mean_distance/mean sizes describe the current batch.
    Save config.json, metrics.csv, best.json; condition='random_small'.
    Seed ariel.ec as well as numpy. Refuse to overwrite existing output.
    """
    raise NotImplementedError("Person 1: run_random_search")
