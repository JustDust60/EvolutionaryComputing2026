"""Growing networks with one dense hidden layer."""

from pathlib import Path
import numpy as np

from common import ACTIVATIONS, Settings, clone_genome, run_ea, validate_genome
from fixed import initialize_fixed, mutate_fixed


def initialize_growing(input_size: int, output_size: int, settings: Settings,
                       rng: np.random.Generator) -> dict:
    return initialize_fixed(input_size, output_size, settings.small_hidden_size,
                            settings, rng)


def add_hidden_neuron(genome: dict, settings: Settings,
                      rng: np.random.Generator) -> dict:
    validate_genome(genome)
    child = clone_genome(genome)
    if child["hidden_size"] >= settings.max_hidden_size:
        return child
    for row in child["w_in"]:
        row.append(float(rng.normal(0.0, settings.initial_weight_sigma)))
    outgoing = []
    for _ in range(child["output_size"]):
        outgoing.append(0.0)
    child["w_out"].append(outgoing)
    child["activations"].append(str(rng.choice(ACTIVATIONS)))
    child["hidden_size"] += 1
    validate_genome(child)
    return child


def mutate_growing(genome: dict, settings: Settings,
                   rng: np.random.Generator) -> dict:
    child = mutate_fixed(genome, settings, rng)
    if child["hidden_size"] < settings.max_hidden_size:
        if rng.random() < settings.add_node_probability:
            child = add_hidden_neuron(child, settings, rng)
    return child


def run_growing(settings: Settings, output_dir: Path) -> dict:
    return run_ea("growing", settings,
                  lambda ni, no, rng: initialize_growing(ni, no, settings, rng),
                  lambda g, rng: mutate_growing(g, settings, rng), output_dir)
