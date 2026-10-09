# Assignment 2: three-person completion plan

## Research question and scope

**Does incrementally adding hidden neurons improve target-reaching performance
compared with fixed-topology weight evolution, at the same simulation-evaluation
budget?**

Lecture connection: simple weight evolution (slide 12), topology-and-weight
evolution (slide 14), and NEAT's minimality/incremental complexification (slides
15–16). This experiment isolates hidden-layer growth. It does not implement or
claim to evaluate full NEAT, speciation, innovation alignment, or arbitrary
topology search.

The template explicitly permits changes to layers and activations, but does not
explicitly address structural mutations during a run. Person 3 should resolve
that remaining interpretation with the instructor early. Do not wait to build
the common evaluator and fixed conditions; those are useful either way. If
structural evolution is ruled out, use the already planned fixed-small versus
fixed-large comparison and change the research question accordingly.

Administrative note: the handout says groups of four. This is the requested
division of work among three active contributors; it does not change the
registered group or the requirement to list all actual group members.

## Keep the project simple

Use one flat folder next to the provided assignment template. Four Python files,
one plan, and one report source are enough. No Git submodules, new packages,
plugin systems, graph frameworks, or custom neural-network library.

| File | Owner | Purpose |
|---|---|---|
| `common.py` | Persons 1/2, assigned function blocks | Shared settings, observation/evaluation contract, EA loop, selection and logs |
| `fixed.py` | Person 1 | Fixed-small, fixed-large, common weight mutation, random search |
| `growing.py` | Person 2 | Adding hidden neurons and the growing-condition mutation |
| `run_experiments.py` | Provided; Person 1 integrates | Runs seeds/configurations and creates verified summaries/figures |
| `main.tex` | Person 3 | Overleaf report; coders supply accurate methods notes |
| `PLAN.md` | Everyone; Person 3 tracks | Work allocation, dates and experiment decisions |

Use the existing repository's ARIEL environment. Never edit `src/ariel`.
Only generated output goes in `results/condition/seed_N/`.

## Experimental design

| Condition | Initial hidden neurons | Evolution | Purpose |
|---|---:|---|---|
| `fixed_small` | 2 | Weights only | Same starting capacity as growing |
| `fixed_large` | 8 | Weights only | Tests whether having more capacity from the start is enough |
| `growing` | 2, cap 8 | Weights and node addition | Main treatment |
| `random_small` | 2 | Independent random weights | Required budget-matched baseline |

All use one dense hidden layer with tanh activations and the same controller
output scaling. Growth appends a hidden neuron with random incoming weights and
zero outgoing weights, preserving the pre-growth function. Later weight
mutations can make the new neuron useful. New zero-weight connections still
count as genes/connections. This is a restricted topology search, not arbitrary
rewiring. Random search is matched to the small architecture; do not claim it
also controls the larger architecture's random-search performance.

Fix one body from `john_set` (provisional `john_set.gecko()`), SimpleFlatWorld,
spawn [0,0,0.1], target XY [2,0], and 15 seconds per final simulation. The template's
other `gecko` import is not a reason to change the handout's John Set requirement.
Provide joint-state, relative-target and sin/cos time features, plus a bias.
Use exactly the same features, normalizations, command policy and fitness for
every condition. Fitness is final XY Euclidean distance, minimized. Do not add
complexity, speed or energy penalties: these would change the experiment.

Provisional shared settings: mu=20 parents, lambda=20 offspring, tournament size
3, (mu+lambda) elitist survivors, Gaussian weight mutation sigma=0.1, expected
one changed weight per offspring, node-addition probability 0.1, initial weight
sigma=0.5. Randomize fitness ties so function-preserving additions can survive.
No crossover or speciation in any condition. The cap and probability are fixed
before final runs; do not tune only the treatment on the final seeds.

The initial budget proposal is 1,020 simulation evaluations per seed: 20 initial
evaluations plus 50 batches of 20. Use five independent final seeds (0–4) per
condition, giving 20 runs / 20,400 simulations at this budget. These are
provisional numbers, not a claim that 50 generations is enough to converge.
Use pilot seeds 100 and 101 to measure runtime and inspect plateaus. Pick one
common final cap after the pilot. If curves still improve materially at that
cap, increase the cap for every condition if time permits, or report explicitly
that results are budget-limited rather than converged. Do not stop individual
final runs early: unequal plateau stopping would undermine the equal-budget
question. The assignment's plateau guidance informs the common budget choice.

## Work allocation and functions

### Person 1 — fixed networks and shared evolutionary search

Implement `initialize_fixed`, `mutate_weights`, and `run_random_search` in
`fixed.py`. In `common.py`, implement `tournament_parent`, `select_survivors`,
and `run_ea`. Person 2 owns the three simulation functions in that file.
Detailed input/output contracts and implementation checklists are in the stubs.

Start with initialization and the shared EA so Person 2 can plug in quickly.
Use `ariel.ec.EA`, `EAOperation`, `Individual` and `Population`; do not substitute
a black-box optimizer or maintain a separate growing-network search loop.
Remember that the framework defaults to maximization: set minimization.
Initialization counts against the budget; surviving cached parents do not get
free extra simulations. Reset both NumPy and ARIEL random generators.

Deliver: functioning fixed-small and fixed-large runs, random baseline,
reproducible settings, unique SQLite database per EA run, agreed CSV logs and
best-controller JSON. Run final fixed and baseline configurations. Give Person
3 approximately half a page of methods facts and the actual parameter count.

### Person 2 — growing networks and shared simulation

Implement `add_hidden_neuron` and `mutate_growing` in `growing.py`. In
`common.py`, implement `build_observations`, `prepare_simulation`, and
`evaluate_genome` once for ALL conditions, adapting the course template.
`initialize_growing` and `run_growing` are already connected to shared code.
Reuse Person 1's weight mutation exactly; apply weight mutation first and node
addition second. Adding neurons is the only reproduction difference.

Build the evaluator early so Person 1 can run fixed networks. Use hand-made
valid dictionaries to check
growth offline. Check old weights are unchanged; input/output sizes remain
fixed; each addition increments hidden size and connection counts; parent
objects are unmodified; the cap is respected; structural addition alone
preserves outputs because the new outgoing weights are zero. Make a meaningful
multi-step check: grow, perturb a new outgoing weight, and confirm the added
unit can actually affect output.

Deliver: working growing condition through the shared runner, cap and shape
checks, evidence that topology can change and be selected during a pilot, and
final growing runs. Inspect the best controller and growth curves, including
whether zero-output additions become useful or remain neutral. Give Person 3
the exact mutation procedure and its limitations. Cross-check Person 1's
evaluation counts and callback cleanup; Person 1 reviews your mutation logic.

### Person 3 — report, interpretation and final submission

Own the Overleaf source, cover page, Introduction, hypotheses, experimental
description, Results/Discussion, Conclusions and bibliography. Obtain scope
clarification early and maintain the decisions table in the report. The two
coders remain responsible for validating the accuracy of their Methods text.

Do not wait for final results: write the Introduction and Methods from verified
implementation notes, and prepare table/figure slots. Use the provided
`--plot` command when all final runs exist; it checks completeness, common
settings and seed grids before plotting. Interpret sample standard deviation
across independent run best-fitness trajectories, not variation within one
population. Keep each seed's final value available; with five runs emphasize
effect sizes and spread, and do not overclaim statistical significance.

Discuss both growing-versus-small and growing-versus-large. If larger fixed
networks explain the advantage, avoid attributing it entirely to staged growth.
If growth fails, a clearly supported negative result still answers the question.
Report restricted one-layer topology, no speciation, the hard cap and compute
budget as limits. Include wall-clock cost descriptively; equal simulation count
does not mean equal wall-clock time.

Deliver: maximum six content pages (cover and bibliography excluded), GECCO19
ACM SigConf layout, accurate group roster and student IDs, reproducible methods,
figures, conclusions, code package and correctly named submission archive.

## Timeline (Europe/Amsterdam)

The handout states a deadline of 13 October 2026 at 09:00. Check the course portal
for its timezone; the schedule below uses the user's Europe/Amsterdam timezone.

| Date | Person 1 | Person 2 | Person 3 | Checkpoint |
|---|---|---|---|---|
| Fri 9 Oct | Freeze schema; initializer + shared EA | Shared evaluator + observations; growth against dummy genomes | Scope check; cover, Introduction and report skeleton | Both network types produce valid actions through one interface |
| Sat 10 Oct | Finish EA + random search; short smoke runs | Integrate growth; meaningful mutation checks | Draft Methods; inspect pilot logs | All four conditions run; pilot seeds 100/101; choose common final budget from timing/plateaus |
| Sun 11 Oct | Launch fixed-small, fixed-large and baseline, seeds 0–4 | Launch growing, seeds 0–4; monitor failures | Finish verified Methods; start summaries as runs complete | Final settings frozen; complete runs logged or runtime problem addressed early |
| Mon 12 Oct | Audit budgets, reproducibility and code | Audit topology traces and best controllers | Final figures, discussion, bibliography and page count | Freeze results and complete report by evening; submit if ready |
| Tue 13 Oct, before 08:30 | Final code/package check | Final results/package check | Submit and verify receipt | Buffer before 09:00 deadline |

Integration rule: each coder edits only their assigned function blocks in
`common.py`; both agree changes to its schema before committing. Use normal commits in the existing team repository.
No new Git workflow, submodules or parallel threads are needed. Run different
conditions on separate people's computers if useful; record environment versions
and keep settings identical. Do not run different hardware as if it were a
controlled wall-clock comparison.

## Commands after implementation

From the existing repository, use its installed environment, for example:

```bash
cd assignments/assignment_2/topology_starter
../../../.venv/bin/python run_experiments.py --plan
../../../.venv/bin/python run_experiments.py --conditions fixed_small fixed_large random_small --seeds 100 --population 4 --offspring 4 --budget 12 --duration 2 --output smoke_fixed
../../../.venv/bin/python run_experiments.py --conditions growing --seeds 100 --population 4 --offspring 4 --budget 12 --duration 2 --output smoke_growing
../../../.venv/bin/python run_experiments.py --conditions fixed_small fixed_large random_small --seeds 0 1 2 3 4 --budget 1020 --output results
../../../.venv/bin/python run_experiments.py --conditions growing --seeds 0 1 2 3 4 --budget 1020 --output results
../../../.venv/bin/python run_experiments.py --plot --output results
```

Replace 1,020 with the agreed final budget. Run folders refuse overwrite; use a
new output root for pilots or reruns. Before implementation, simulation commands
intentionally raise `NotImplementedError`; `--plan` already works. Plotting needs
five complete runs for every condition. If runs were performed on two computers,
collect their condition directories under the same results root first.

## Submission checklist

- [ ] Instructor interpretation of topology evolution resolved and described honestly.
- [ ] Own EA built on `ariel.ec`; no changes in `src/ariel`; no black-box optimizer.
- [ ] Fixed John Set body and supported fixed world; inputs, duration, target and fitness shared.
- [ ] At least five independent complete runs per configuration and budget-matched baseline.
- [ ] Mean and standard deviation across seeds; required generation plot plus evaluation plot.
- [ ] Final-distance table and topology growth figure; no fabricated placeholder values.
- [ ] Report accurately describes implemented code and specifies seeds/settings/budget.
- [ ] Six-page content limit checked; all actual members/IDs on the separate cover.
- [ ] Final `groupnumber.zip` contains `groupnumber/`, `groupnumber.pdf`, code and run instructions.
- [ ] Submission uploaded and receipt checked before the course deadline.

## Sources

- Local handout: `assignments/assignment_2/Assignment2.html`, sections 5–9.
- Local template: `assignments/assignment_2/A2_template_2026.py`, controller
  contract, architecture permission at lines 142–145, and YOUR JOB section.
- Course lecture: `EC_NeuroEvolution_2026_v2.pdf`, slides 12, 14–16 and 21–24.
- [GECCO 2019 template instructions](https://gecco-2019.sigevo.org/index.html/Papers%2BSubmission%2BInstructions): ACM SigConf format. Course limits/cover requirements override conference submission rules.
- [Stanley & Miikkulainen (2002)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf): original NEAT paper; background, not a claim that our restricted experiment is full NEAT.
