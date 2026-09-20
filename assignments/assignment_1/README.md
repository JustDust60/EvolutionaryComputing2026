# Assignment 1 — elitism vs. generational replacement

**Research question.** Within one otherwise identical EA for evolving a robot
body towards a set of target morphologies, does the *survivor-selection scheme*
change the quality of the body found, or the speed at which it is found?

Two variants of the same algorithm are compared, differing in one aspect only:

| variant | survivor selection |
| --- | --- |
| `replacement` | the offspring population replaces the parent population wholesale |
| `elitism` | the `s` best of parents ∪ offspring always survive; the remaining `μ − s` slots are filled by a uniform sample of the rest |

Both are compared against a **random-search control at the same evaluation
budget**.

## Files

| file | what it is |
| --- | --- |
| `A1_2026.py` | the EA itself: encoding, fitness, operators, both selection schemes, `run_evolution`, `random_search`. Running it directly executes a small smoke test, not an experiment. |
| `experiment_runner.py` | the experiment harness: tuning sweep, final runs, statistics, figures. |
| `tree_edit_distance.py` | the fitness metric (given by the course; unmodified). |
| `target_bodies/` | the five target morphologies (7, 11, 15, 19 and 25 modules). |
| `shared_parameters.json` | the configuration used by both variants in the headline experiment. |
| `tuned_parameters.json` | per-variant configuration chosen by `tune`; overwritten each time the sweep is re-run. |
| `__data__/A1_2026/<tag>/` | everything an experiment produced: raw per-generation CSV, summary CSV, statistics tables, champion genomes, figures. |

## Reproducing the results

All commands are run from this directory, with the project's virtual
environment active.

```bash
# 0. smoke test - 10 individuals, 5 generations, seconds
python A1_2026.py

# 1. preliminary parameter sweep (disjoint seeds 101-105, 40 generations)
python experiment_runner.py tune

# 2. the two final experiments, 10 independent seeds each, 100 generations
python experiment_runner.py final --tag final_shared \
    --config shared_parameters.json
python experiment_runner.py final --tag final_tuned \
    --config tuned_parameters.json

# 3. the elitism-strength sweep
python experiment_runner.py sweep --tag sweep_elitism \
    --config shared_parameters.json

# redraw every figure and reprint every table from saved data, no re-running
python experiment_runner.py plot --tag final_shared
```

Runs are distributed over CPU cores by default; `--workers 1` forces serial
execution. Every run seeds `random`, `numpy` and `torch` from its seed number,
so a given (condition, seed) pair reproduces exactly.

## Experimental setup

| setting | value |
| --- | --- |
| encoding | `tree` (direct; `ariel.ec.genotypes.tree`) |
| module budget | 20 |
| population | 50 |
| generations | 100 |
| **evaluation budget** | **5050 per run** = 50 initial + 50 × 100 offspring |
| parent selection | tournament, `k = 5` |
| crossover | subtree crossover, `p_c = 0.6` |
| mutation | subtree replacement, `p_m = 0.8` |
| elites | `s = 2` (headline comparison); swept over 2, 5, 10, 25, 45 |
| independent runs | 10 per condition (assignment minimum: 5) |
| statistics | Mann–Whitney U, two-sided, plus Cliff's δ |

### Why the budget is counted the way it is

Every genome is scored **exactly once**; the result is memoised by a canonical
signature of its nodes and edges (`A1_2026.genome_signature`). Without this,
the number of fitness evaluations would depend on the selection scheme rather
than on the search effort: elitism carries survivors over unchanged and
tournament selection inspects the same parents repeatedly, and re-scoring them
would have charged elitism for work it did not do. With the cache in place both
variants create the same number of new individuals per generation, and random
search draws exactly as many genomes in total.

In practice the EA conditions spend *fewer* distinct evaluations than the
nominal 5050 (roughly 3700–4600), because some offspring are structurally
identical to an individual already scored — the tree operators roll back to a
copy of the parent when a mutation or crossover would produce an invalid body.
The EA is therefore compared against random search from a position of
disadvantage, never advantage.

### Why the five-plus repetitions all use the same parameters

The repetitions exist to estimate the variance *of one configuration*, which is
what the mean and the shaded ±1 std band in every figure report. Varying
hyperparameters between repetitions would fold parameter variance into seed
variance and leave neither measurable. Parameter choice is therefore a separate,
earlier phase (`tune`) on **seeds 101–105, disjoint from the final seeds 1–10**,
at a reduced generation budget. The final runs then use one frozen
configuration per condition.

Because the tuning phase found several configurations within one standard error
of each other, the comparison was run twice:

* `final_shared` — both variants use the *same* parameters (`k=5, p_c=0.6,
  p_m=0.8, s=2`), so selection scheme is strictly the only difference. This is
  the headline experiment.
* `final_tuned` — each variant uses its own tuned parameters, i.e. best against
  best. Reported as a robustness check; it reaches the same conclusion.

## Results

Final best fitness over 10 runs, lower is better (`final_shared`):

| condition | mean | std | median | best run | evaluations |
| --- | --- | --- | --- | --- | --- |
| replacement | 12.588 | 0.284 | 12.560 | 12.049 | 4550 |
| elitism (s=2) | 12.627 | 0.358 | 12.627 | 12.000 | 4272 |
| random search | 17.133 | 0.315 | 17.207 | 16.378 | 5050 |

1. **Both EA variants beat random search decisively.** U = 0, p = 0.0002,
   Cliff's δ = −1.0 (no overlap between the two samples) for each variant.
   At an identical budget, random search never comes within 3.5 fitness points
   of either EA.
2. **The survivor-selection scheme does not change the final quality.**
   replacement vs. elitism: p = 0.82, δ = −0.07 under shared parameters;
   p = 0.38, δ = 0.24 under per-variant tuned parameters. Every pairwise
   comparison in the `s` sweep is non-significant too. All conditions converge
   to a plateau around 12.5.
3. **It does change how fast that plateau is reached.** Generations needed to
   reach fitness 12.735 (median over the runs that reach it): 56 for
   replacement, 42 at s=2, 26 at s=5, 24 at s=10, 31 at s=25 and s=45.
4. **Population diversity is the mechanism.** The fraction of structurally
   unique individuals stays at ~0.99 under generational replacement and falls
   monotonically with `s` — to ~0.60 at s=25 and ~0.34 at s=45. Elitism buys
   convergence speed by spending diversity, and on this landscape the trade is
   roughly fitness-neutral over 100 generations.
5. **The fitness landscape has a floor and the EA finds it.** The mean pairwise
   distance within the target set is 15.75, so no single body can be close to
   all five at once. Every evolved champion converges to 13–14 modules, between
   the target sizes of 7 and 25, while random search's best body keeps the ~21
   modules that `random_tree` produces by default.

## Figures

Each experiment folder contains:

| figure | content |
| --- | --- |
| `fig_best_so_far.png` | best-so-far fitness per generation, mean ± std over runs — the assignment's required line plot |
| `fig_best_per_generation.png` | best individual of each generation |
| `fig_population_mean.png` | population mean fitness |
| `fig_diversity.png` | fraction of the population that is structurally unique |
| `fig_final_distribution.png` | box plot of the final best fitness of every run |
| `fig_bodies.png` | champion body of each condition drawn against all five targets |
