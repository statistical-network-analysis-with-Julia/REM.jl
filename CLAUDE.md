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
- `EventSequence{T}` - Time-sorted collection of events with actor tracking
- `ActorSet` - Set of actors with optional ID-to-name mapping
- `NodeAttribute{T}` - Actor-level attribute storage with default values
- `RiskSet` - Defines potential dyads for case-control sampling

### Network State (`src/network.jl`)
- `EventNetworkState{T}` - Tracks cumulative network state for efficient statistic computation
- Maintains dyad counts, degrees, and event history with optional exponential decay

### Statistics (`src/statistics/`)
All statistics implement the `compute(stat, state, sender, receiver) -> Float64` interface:

- **Dyad** (`dyad.jl`): `Repetition`, `Reciprocity`, `InertiaStatistic`, `RecencyStatistic`
- **Degree** (`degree.jl`): `SenderActivity`, `ReceiverActivity`, `SenderPopularity`, `ReceiverPopularity`
- **Triangle** (`triangle.jl`): `TransitiveClosure`, `CyclicClosure`, `SharedSender`, `SharedReceiver`
- **FourCycle** (`fourcycle.jl`): `FourCycle` with various cycle type configurations
- **Node** (`node.jl`): `AttributeMatch`, `NodeMix`, `NodeDifference`, `SenderAttribute`, `ReceiverAttribute`

`StatisticSet` (in `statistics/base.jl`) is tuple-backed: `compute_all`/`compute_all!` over a set compile to statically dispatched per-statistic calls (no dynamic dispatch in the observation/likelihood inner loop). Vectors of statistics passed to `generate_observations`/`compute_statistics`/`fit_rem` are converted to a `StatisticSet` internally; reuse a set across calls to avoid recompiling per tuple type.

### Estimation Pipeline (`src/observation.jl`, `src/estimation.jl`)
1. `CaseControlSampler` - Generates case-control observations from event sequence
2. `generate_observations()` - Computes statistics for cases and sampled controls
3. `fit_rem()` - Fits stratified conditional logistic regression via Newton-Raphson

### Package Extensions (`ext/`)
- `REMNetworkDynamicExt` (weak dep on NetworkDynamic.jl): `EventSequence(::DynamicNetwork)` converts edge activation spells to events at their onset times (onset-censored spells skipped by default). Tested by adding NetworkDynamic to the test target.

## Modeling Assumptions and Behaviors

- **Ordinal likelihood only**: each event is one stratum of a conditional logistic regression; exact waiting times enter only through optional decay weighting, not as a hazard term (unlike `relevent::rem.dyad`'s interval likelihood). Tied timestamps are ordered arbitrarily (a one-time warning fires).
- **Sampling is without replacement** with a local seeded RNG; when fewer distinct dyads exist than `n_controls`, the full risk set is enumerated instead (one-time warning).
- `EventNetworkState` maintains incremental `out_neighbors`/`in_neighbors` adjacency sets so neighbor queries are O(degree); adjacency records "ever had an event" and does not expire under decay (counts do decay).
- `_fit_stratified_clogit` uses Newton-Raphson with step-halving; convergence requires both a small log-likelihood change and a small gradient norm. The per-iteration derivatives accumulate the Hessian in place via BLAS (gemm on sqrt-probability-weighted stratum rows plus a `ger!` rank-1 update) into preallocated workspace buffers — no per-row `x*x'` outer products.

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
seq = EventSequence(events)

# Define statistics
stats = [Repetition(), Reciprocity(), SenderActivity()]

# Fit model with case-control sampling
result = fit_rem(seq, stats; n_controls=100, seed=42)
```
