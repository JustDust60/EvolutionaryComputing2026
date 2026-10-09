"""PERSON 2: growing networks and common.py simulation/observation TODOs. Only hidden neurons are added; topology stays a
single dense hidden layer. This is a focused topology-growth experiment,
not full NEAT: no species, innovation numbers, crossover, recurrence or deletion.
"""

from pathlib import Path
import numpy as np

from common import Settings, clone_genome, run_ea, validate_genome
from fixed import initialize_fixed, mutate_weights


def initialize_growing(input_size: int, output_size: int, settings: Settings,
                       rng: np.random.Generator) -> dict:
    """Provided: exactly the same starting architecture/distribution as small.

    Person 2 can develop growth with a hand-built valid genome before Person 1
    finishes initialize_fixed. Both people freeze the schema immediately.
    """
    return initialize_fixed(input_size, output_size, settings.small_hidden_size,
                            settings, rng)


def add_hidden_neuron(genome: dict, settings: Settings,
                      rng: np.random.Generator) -> dict:
    """TODO PERSON 2: deep-copy and append one hidden neuron, respecting cap.

    If hidden_size >= max_hidden_size, return an unchanged deep copy.
    Append a column to w_in, sampled from N(0, initial_weight_sigma).
    Append a ZERO row to w_out, increment hidden_size, preserve existing
    weights exactly, and validate shapes. New connections are real genes,
    including zero weights, and are eligible for future weight mutation.
    Zero outgoing weights make this structural operation alone preserve the
    parent's output exactly; assert this on several observation vectors.
    Return ordinary nested lists. No innovation IDs or graph library needed.
    """
    raise NotImplementedError("Person 2: add_hidden_neuron")


def mutate_growing(genome: dict, settings: Settings,
                    rng: np.random.Generator) -> dict:
    """TODO PERSON 2: common weight mutation, then optional node addition.

    Call fixed.mutate_weights first. With add_node_probability, call
    add_hidden_neuron on the result. This is the only reproduction difference
    from fixed_small/fixed_large. Return a new valid genome.
    At the cap, mutation is weights-only. No extra randomness in initialization.
    """
    raise NotImplementedError("Person 2: mutate_growing")


def run_growing(settings: Settings, output_dir: Path) -> dict:
    """Provided: use Person 1's shared evaluation, selection, budget and logs."""
    return run_ea("growing", settings,
                  lambda ni, no, rng: initialize_growing(ni, no, settings, rng),
                  lambda g, rng: mutate_growing(g, settings, rng), output_dir)
