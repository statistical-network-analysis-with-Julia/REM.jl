# REM.jl

*Relational Event Models for Julia*

A Julia package for statistical analysis of time-stamped relational events in networks.

## Overview

Relational Event Models (REM) are statistical models for analyzing sequences of time-stamped relational events. An event is a directed interaction from a sender to a receiver at a specific point in time. REMs uncover factors explaining why certain actors interact at higher rates than others.

REM.jl is a port of [eventnet](https://github.com/juergenlerner/eventnet), providing efficient tools for modeling sequences of directed interactions between actors.

### What is a Relational Event?

A relational event is a time-stamped directed interaction:

```text
Actor A → Actor B at time t
```

Examples include:

- Emails sent between colleagues
- Messages in a chat application
- Trade transactions between countries
- Collaboration events between scientists

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Relational Event** | A time-stamped directed interaction (sender → receiver) |
| **Event Sequence** | A chronologically ordered sequence of relational events |
| **Network State** | The cumulative state of interactions up to a point in time |
| **Statistic** | A computed feature that may predict event occurrence |
| **Hazard Rate** | The instantaneous probability of an event occurring |

### Applications

REMs are widely used in:

- **Social network analysis**: Understanding communication patterns
- **Organizational studies**: Modeling collaboration and coordination
- **International relations**: Analyzing diplomatic interactions
- **Animal behavior**: Studying social hierarchies and mating patterns
- **Online platforms**: Modeling user interactions and content sharing

## Features

- **Rich statistic library**: 25+ statistics covering dyadic, degree, triadic, four-cycle, and attribute effects
- **Efficient computation**: Incremental network state updates for fast statistic calculation
- **Temporal decay**: Support for exponential decay of network effects
- **Case-control sampling**: Efficient estimation for large networks via stratified conditional logistic regression
- **Flexible timestamps**: Works with numeric, DateTime, and Date timestamps
- **Easy data loading**: Load events from DataFrames or CSV files

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Networks.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/NetworkDynamic.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/REM.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/REM.jl")
```

## Quick Start

```julia
using REM

# Create events
events = [
    Event(1, 2, 1.0),  # Actor 1 → Actor 2 at time 1.0
    Event(2, 1, 2.0),  # Actor 2 → Actor 1 at time 2.0 (reciprocation)
    Event(1, 3, 3.0),  # Actor 1 → Actor 3 at time 3.0
    Event(1, 2, 4.0),  # Actor 1 → Actor 2 at time 4.0 (repetition)
]
# Declare the actor universe (actor 4 is an eligible isolate); omitting
# `actors` infers it from the observed events and changes the estimand
seq = EventSequence(events; actors=ActorSet([1, 2, 3, 4]))

# Define model statistics
stats = [
    Repetition(),        # Tendency to repeat past interactions
    Reciprocity(),       # Tendency to reciprocate
    SenderActivity(),    # Sender's past activity level
    ReceiverPopularity() # Receiver's past popularity
]

# Fit model with case-control sampling
result = fit_rem(seq, stats; n_controls=100, seed=42)

# View results
println(result)
```

## Choosing Statistics

| Use Case | Recommended Statistics |
|----------|----------------------|
| Basic dyadic effects | [`Repetition`](@ref), [`Reciprocity`](@ref) |
| Actor heterogeneity | [`SenderActivity`](@ref), [`ReceiverPopularity`](@ref) |
| Triadic closure | [`TransitiveClosure`](@ref), [`CyclicClosure`](@ref) |
| Homophily effects | [`AttributeMatch`](@ref), [`NodeDifference`](@ref) |
| Time-varying effects | [`RecencyStatistic`](@ref) + temporal decay |
| Complex clustering | [`FourCycle`](@ref), [`GeometricWeightedTriads`](@ref) |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/events.md",
    "guide/statistics.md",
    "guide/estimation.md",
    "guide/decay.md",
    "api/types.md",
    "api/statistics.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### The Relational Event Model

REMs model the rate at which events occur as a function of the network's history. For a potential event from sender $s$ to receiver $r$ at time $t$, the hazard rate is:

$$\lambda_{sr}(t) = \exp\left(\sum_k \beta_k x_k(s, r, t)\right)$$

Where:

- $x_k(s, r, t)$ are statistics computed from the network history
- $\beta_k$ are coefficients to be estimated

### Case-Control Sampling

For large networks, computing statistics for all possible dyads is expensive. REM.jl uses case-control sampling:

1. For each observed event (case), sample non-events (controls) from the risk set
2. Compute statistics for both cases and controls
3. Estimate via stratified conditional logistic regression

This approach is statistically consistent and dramatically reduces computation time.

## References

1. Butts, C.T. (2008). A relational event framework for social action. *Sociological Methodology*, 38(1), 155-200.

2. Lerner, J., Lomi, A. (2020). Reliability of relational event model estimates under sampling: How to fit a relational event model to 360 million dyadic events. *Network Science*, 8(1), 97-135.

3. Lerner, J., Bussmann, M., Snijders, T.A.B., Brandes, U. (2013). Modeling frequency and type of interaction in event networks. *Corvinus Journal of Sociology and Social Policy*, 4(1), 3-32.

4. Brandes, U., Lerner, J., Snijders, T.A.B. (2009). Networks evolving step by step: Statistical analysis of dyadic event data. *2009 International Conference on Advances in Social Network Analysis and Mining*, 200-205.

5. Perry, P.O., Wolfe, P.J. (2013). Point process modelling for directed interaction networks. *Journal of the Royal Statistical Society: Series B*, 75(5), 821-849.

## Module

```@docs
REM
```
