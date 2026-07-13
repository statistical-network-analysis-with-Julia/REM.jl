# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REM.jl is a Julia implementation of Relational Event Models for statistical analysis of time-stamped relational events in networks. It is a port of [eventnet](https://github.com/juergenlerner/eventnet).

## Development Commands

```bash
# Run tests from shell
julia --project -e 'using Pkg; Pkg.test()'
```

```julia
# Run tests from Julia REPL
using Pkg; Pkg.test("REM")

# Load the package in development
using Pkg; Pkg.develop(path=".")
using REM

# Run specific test file directly
include("test/runtests.jl")
```

## Architecture

### Core Data Types (`src/types.jl`)
- `Event{T}` - Single relational event with sender, receiver, timestamp, type, and weight
- `EventSequence{T}` - Time-sorted collection of events with an actor universe. Declare the universe with `EventSequence(events; actors=ActorSet(...))` (isolates and noncontiguous IDs included); when omitted it falls back to the observed event endpoints and `actors_declared` is `false` (fitting against such a sequence warns)
- `ActorSet` - Set of actors with optional ID-to-name mapping
- `NodeAttribute{T}` - Actor-level attribute storage with default values
- `RiskSet` - Defines potential dyads (sender × receiver) for case-control sampling

### Network State (`src/network.jl`)
- `EventNetworkState{T}` - Tracks cumulative network state for efficient statistic computation
- Maintains dyad counts, degrees, and event history with optional exponential decay; decayed counts are stored as `(value, last_update_time)` pairs and decayed lazily on read relative to `current_time`, so advancing the clock is O(1) and each event is absorbed exactly once

### Statistics (`src/statistics/`)
All statistics implement the `compute(stat, state, sender, receiver) -> Float64` interface:

- **Dyad** (`dyad.jl`): `Repetition`, `Reciprocity`, `InertiaStatistic`, `RecencyStatistic`
- **Degree** (`degree.jl`): `SenderActivity`, `ReceiverActivity`, `SenderPopularity`, `ReceiverPopularity`
- **Triangle** (`triangle.jl`): `TransitiveClosure`, `CyclicClosure`, `SharedSender`, `SharedReceiver`
- **FourCycle** (`fourcycle.jl`): `FourCycle` with various cycle type configurations
- **Node** (`node.jl`): `AttributeMatch`, `ActorMix`, `NodeDifference`, `SenderAttribute`, `ReceiverAttribute`

`compute`, `name` and `compute_all` are **not REM's generics** — they are the shared statistic protocol defined in Networks.jl (`src/statistics.jl`) and imported by name, exactly as `gof` is. Every model package extends the same three functions with methods for its own statistic types, so `REM.compute === ERGM.compute` and the co-load that the whole statnet workflow depends on —

```julia
using ERGM, REM     # cross-sections with ERGM, dynamics with REM
compute(Edges(), net)                 # ERGM's method
compute(Repetition(), state, 1, 2)    # REM's method — same generic
```

— leaves the verbs usable unqualified instead of undefined by Julia's conflicting-export rule. Adding a statistic anywhere (Relevent.jl does exactly this) means `import REM: compute, name` and adding methods; never define a local `compute`. The rule is pinned by the "Namespace: co-loading ERGM and REM" testset, which also runs the check in a fresh process (ERGM is a test-only dependency).

A name that means something *different* in another package gets renamed, never re-exported: `NodeMatch` → `AttributeMatch`, `NetworkState` → `EventNetworkState`, `NodeMix` → `ActorMix` (vs `ERGM.NodeMix`, a cross-sectional mixing term). `REM.NodeMix` survives as a deprecated, **non-exported** alias — exporting it would recreate the collision. A name that means the *same* thing, in contrast, becomes a method on the shared generic and stays exported: `has_edge(state, s, r)` is a method of `Graphs.has_edge` (via `import Networks: has_edge`), because asking "is there a tie from i to j" of an accumulated event network is the question Graphs already asks of a graph.

`StatisticSet` (in `statistics/base.jl`) is tuple-backed: `compute_all`/`compute_all!` over a set compile to statically dispatched per-statistic calls (no dynamic dispatch in the observation/likelihood inner loop). Vectors of statistics passed to `generate_observations`/`compute_statistics`/`fit_rem` are converted to a `StatisticSet` internally; reuse a set across calls to avoid recompiling per tuple type.

### Estimation Pipeline (`src/observation.jl`, `src/estimation.jl`)
1. `CaseControlSampler` - Generates case-control observations from event sequence
2. `generate_observations()` - Computes statistics for cases and sampled controls
3. `fit_rem()` - Fits stratified conditional logistic regression via Newton-Raphson

### Standard errors: three estimators, three different assumptions (REM#2)

The partial likelihood is evaluated on ONE draw of the controls, and it assumes the observed information equals the score variance. `fit_rem(...; se=...)` relaxes those separately; `REMResult.se_type` records which was used and `Networks.se_method(fit)` reports it, so a `show` method can never claim an estimator that was not run. **The point estimates are identical under all three** — they stay those of the original (`seed`) control draw; only the covariance is replaced.

- `se=:hessian` (default) — inverse negative Hessian. Conditional on the one control draw, so it **understates** the uncertainty.
- `se=:sandwich` — **event-clustered Godambe sandwich** `H⁻¹BH⁻¹`, meat `B = Σ_e u_e u_eᵀ` from the per-event score contributions (`_clogit_sandwich_se`). Each event IS one stratum, so the event is the clustering unit and there is no `cluster` argument to pass. Robust to misspecification of the within-stratum conditional model; still on the one drawn control set. Works on both the `EventSequence` and the `DataFrame` methods.
- `se=:bootstrap` — **repeated control sampling**, on the shared `Networks.bootstrap_cov` loop. A *parametric* bootstrap would be the wrong tool here and must not be added: what the Hessian misses is variance from the **sampling design**, not from the model. Redraw the risk set `n_boot` times (seeds from `rng`), refit, and combine by the **law of total variance**: `V = W̄ + (1 + 1/B)·B_between` — the mean of the per-draw inverse-information covariances plus the empirical covariance of the refits. **The between-draw term alone is NOT the answer**: it is one component of the total and is *smaller* than the Hessian SE, so reporting it as the standard error would understate the uncertainty even further than `:hessian` does. With the combination, the SEs necessarily exceed the `:hessian` ones, and the excess is exactly the risk-set sampling variability. Refused on the `DataFrame` method (the controls are already drawn there) with a pointer to the `EventSequence` method.

The control inclusion probabilities that all of this is about are exposed as `sampling_probs(fit)` (and `risk_set_sizes(fit)`) — part of the estimand, not an implementation detail. With the full risk set (`sampling_probs` all `1.0`) nothing is conditioned away: `is_exact` holds and `show` prints no caveat at all.

### Tied event times: `ties=:error|:ordered|:breslow|:efron` (REM#2, finding 12)

The partial likelihood is a **Cox partial likelihood** (one stratum per event), so a tied timestamp is the classical Cox tie problem and Breslow/Efron are the classical answers — but note what a tie does *here*: statistics are read off the network state **before** the event, so ordering two simultaneous events lets the one placed first enter the **statistics** of the one placed second (its `Repetition`, its `Reciprocity`, its degrees). That is not a tie-break, it is invented information — which is why the default is refusal, not a sort.

The vocabulary is **`Networks.TIE_POLICIES`**, defined once in Networks.jl (`src/results.jl`) and shared with Relevent.jl; `Networks.check_tie_policy` is the guard, and it makes an option a model cannot honour **fail loudly instead of no-op**.

- `:error` (**default**) — names the tie (which events, which timestamp, how many in all) and throws.
- `:ordered` — sequence order, no correction: the pre-0.2 behaviour, now opt-in.
- `:breslow` — the network state is **frozen across the tie block** (the block is absorbed as a whole afterwards), each tied event is its own stratum, all share one denominator.
- `:efron` — as `:breslow`, plus the denominator weights `1 − (j−1)/d` on the `d` tied cases, carried per row in the **`tie_weight` column** and applied by `_fit_stratified_clogit` (`tie_weights=`) as `Σ_a w_a·exp(η_a)` in the stratum denominator, numerator unchanged. Requires the tied cases to be **distinct dyads** — a dyad competing with itself has no fractional weight, and inventing one can even go negative, so it throws. Efron on the same design *unweighted* is exactly Breslow (tested).
- `:batch` — **refused**, pointing at `:breslow`: with the risk set held fixed, a "simultaneous batch" in an ordinal likelihood IS the Breslow correction. (It is `Relevent.fit_timing`'s policy, where there is an exposure interval for a batch to consume.)

The policy is applied where the **design is built** (`generate_observations`), and rides with it: the applied policy is attached as the `"tie_method"` DataFrame metadata (`:note` style), so `fit_rem(::DataFrame, ...)` reports the truth without being told twice. A frame marked `:efron` that has *lost* its `tie_weight` column is refused rather than fitted unweighted while claiming Efron.

`tie_method(fit)` reports **what actually happened**, so it is `:none` when the data had no ties (a correction on tie-free data corrected nothing) and never `:error` (a tie under `:error` throws instead of returning). `is_exact(fit)` requires *both* the full risk set and `:none`. On tie-free data all four policies produce the identical design, row for row — the sharpest available check that a correction is a correction.

**Golden fixture** `test/fixtures/rem_ties.toml` (`test/fixtures/r/rem_ties.R`) pins `:breslow`/`:efron` against **`survival::coxph(..., ties="breslow"/"efron")`** on a sequence observed on a coarse clock (25 of 53 timestamps tied, up to 4 deep). The R script builds the counting-process design — one interval per distinct time, all `n(n−1)` dyads at risk, covariates frozen across each block — from the raw edgelist in plain R. Agreement is < 1e-11 on coefficients, standard errors and log-likelihood. That is what makes "we implemented Breslow" checkable.

### Package Extensions (`ext/`)
- `REMNetworkDynamicExt` (weak dep on NetworkDynamic.jl): `EventSequence(::DynamicNetwork)` converts edge activation spells to events at their onset times (onset-censored spells skipped by default). Tested by adding NetworkDynamic to the test target.

  It honours the **ecosystem conversion contract** (Networks.jl `src/conversion.jl`; per-path table in `Networks.jl/docs/src/guide/conversion_invariants.md`). An `Event` is an instant and has no way to say a dyad is *unobserved*, so a `DynamicNetwork` whose base network carries a missing-dyad mask is **rejected** (`missing=:error`, the ecosystem default; `missing=:face` is the auditable opt-in). This is not fussiness: silently turning an unobserved dyad into a never-happened non-event biases a likelihood that is **conditional on the risk set**, which is the estimand. `report=true` returns `(seq, ::Networks.ConversionReport)` naming what an event sequence cannot carry — spell termini (a dissolution is not an event), terminus censoring, vertex spells (the declared actor universe is flat over time, not a time-varying risk set), attributes, and the observation window. Pinned by the "NetworkDynamic extension: conversion invariants" testset.

## Modeling Assumptions and Behaviors

- **Ordinal likelihood only**: each event is one stratum of a conditional logistic regression; exact waiting times enter only through optional decay weighting, not as a hazard term (unlike `relevent::rem.dyad`'s interval likelihood).
- **The risk set is the estimand**: the likelihood is conditional on it, so the actor universe must be declared, never inferred from outcomes. `generate_observations`/`fit_rem` take `at_risk` (alias `riskset` in `fit_rem`) as a static actor set, a static `RiskSet` (asymmetric sender/receiver), a vector of per-event risk sets, or a callback `(event_index, state) -> RiskSet`. Every case is validated against its own risk set and each risk set must admit ≥ 1 control — both throw before fitting. Risk-set size and control sampling probability are recorded per stratum (`risk_set_size`/`sampling_prob` columns; `REMResult.risk_set_sizes`/`.sampling_probs`).
- **Sampling is without replacement** with a local seeded RNG; when fewer distinct dyads exist than `n_controls`, the full risk set is enumerated instead (one-time warning).
- `EventNetworkState` maintains incremental `out_neighbors`/`in_neighbors` adjacency sets so neighbor queries are O(degree); adjacency records "ever had an event" and does not expire under decay (counts do decay).
- `_fit_stratified_clogit` uses Newton-Raphson with step-halving; convergence requires both a small log-likelihood change and a small gradient norm. The per-iteration derivatives accumulate the Hessian in place via BLAS (gemm on sqrt-probability-weighted stratum rows plus a `ger!` rank-1 update) into preallocated workspace buffers — no per-row `x*x'` outer products. It takes `se=:hessian|:sandwich` (the point estimate does not depend on it) and always returns `var_cov`, the inverse observed information, because the repeated-control-sampling SEs need it per draw as the within-draw variance component.

## Golden fixture (R)

`test/fixtures/rem_clogit.toml` freezes `survival::clogit` (survival 3.8-3, R 4.6.1) on a simulated 10-actor / 80-event sequence; `test/fixtures/r/rem_clogit.R` regenerates it (`Rscript test/fixtures/r/rem_clogit.R > test/fixtures/rem_clogit.toml`). Loaded with Networks.jl's `load_golden`, which refuses a fixture without provenance.

The R script rebuilds the **whole stratified design matrix from the raw edgelist in plain R** — it does not import anything Julia computed — so the fixture checks the *statistics* (`Repetition`, `Reciprocity`, `SenderActivity`, `ReceiverPopularity`, `TransitiveClosure`) as well as the estimator. The risk set is enumerated in **full** on both sides (pass `n_controls = n(n-1) - 1`, which makes `generate_observations` enumerate rather than sample), so there are no sampled controls to reconcile across two RNGs and the comparison is exact rather than distributional.

Tolerance **1e-8**: both sides maximize the same exact conditional-logit likelihood by Newton-Raphson, so nothing may differ but floating-point summation order. Observed agreement is <1e-13. Case-control *sampling* is a variance/compute tradeoff on top of this likelihood; it is not what R would disagree with, and it is not what this fixture tests.

## Key Design Patterns

- Statistics are computed lazily using `EventNetworkState` which updates incrementally
- Exponential decay of network effects via configurable halflife/decay parameters
- Case-control sampling enables efficient estimation for large networks
- All statistics return `Float64` for consistent matrix operations

## Example Usage

```julia
using REM

# Load events
events = [
    Event(1, 2, 1.0),
    Event(2, 1, 2.0),
    Event(1, 3, 3.0)
]
# Declare the actor universe (actor 4 is an eligible isolate)
seq = EventSequence(events; actors=ActorSet([1, 2, 3, 4]))

# Define statistics
stats = [Repetition(), Reciprocity(), SenderActivity()]

# Fit model with case-control sampling
result = fit_rem(seq, stats; n_controls=100, seed=42)
```
