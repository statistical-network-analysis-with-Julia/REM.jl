# REM.jl

[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/REM.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/REM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/REM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/REM.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/REM.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="REM.jl icon" width="160">
</p>

A Julia implementation of **Relational Event Models** for statistical analysis of time-stamped relational events in networks.

## Overview

Relational Event Models (REM) are statistical models for analyzing sequences of time-stamped relational events. They uncover factors explaining why certain actors interact at higher rates than others, accounting for:

- **Dyadic effects**: Repetition, reciprocity, and inertia between actor pairs
- **Actor effects**: Activity and popularity patterns at the node level
- **Structural effects**: Triadic closure, four-cycles, and local clustering
- **Attribute effects**: Homophily and covariate-based selection

REM.jl is a port of [eventnet](https://github.com/juergenlerner/eventnet), providing efficient tools for modeling sequences of directed interactions between actors.

**Modeling assumptions.** REM.jl estimates the *ordinal* relational event model: only the order of events enters the likelihood (each event is one stratum of a conditional logistic regression against sampled non-events). Exact inter-event waiting times are used only for optional decay weighting, not as a hazard term — the interval-timing likelihood of R `relevent::rem.dyad` is not implemented. Tied timestamps are processed in arbitrary sequence order without a tie correction, so coarse/tied time data deserve caution. Controls are sampled without replacement; when a small actor set makes fewer distinct dyads available than `n_controls`, the full risk set is used instead.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/REM.jl")
```

## Statistics Implemented

### 1. Dyad Statistics

Statistics based on the history of events between specific actor pairs.

```julia
Repetition(; directed=true)      # Past events from sender to receiver
Reciprocity()                    # Past events from receiver to sender
InertiaStatistic()               # Combined repetition and reciprocity
RecencyStatistic()               # Inverse time since last event on dyad
DyadCovariate(attr)              # Dyad-level covariate value
```

### 2. Degree Statistics

Statistics measuring actor activity and popularity levels.

```julia
SenderActivity()                 # Sender's out-degree (past sending activity)
ReceiverActivity()               # Receiver's out-degree (receiver's sending history)
SenderPopularity()               # Sender's in-degree (how often sender receives)
ReceiverPopularity()             # Receiver's in-degree (how often receiver receives)
TotalDegree(; role=:sender)      # Combined in-degree and out-degree
DegreeDifference()               # Difference between sender and receiver degrees
LogDegree(; role=:sender)        # Log-transformed degree statistic
```

### 3. Triangle Statistics

Statistics capturing triadic closure patterns in directed networks.

```julia
TransitiveClosure()              # s→k→r patterns (transitive triads)
CyclicClosure()                  # r→k→s patterns (cyclic triads)
SharedSender()                   # k→s and k→r patterns (common sender)
SharedReceiver()                 # s→k and r→k patterns (common receiver)
CommonNeighbors()                # Any shared third party connections
GeometricWeightedTriads(α=0.5)   # Geometrically weighted triadic effects
```

### 4. Four-Cycle Statistics

Statistics measuring local clustering through four-cycle configurations.

```julia
FourCycle(; cycle_types=[:all])           # Various four-cycle configurations
GeometricWeightedFourCycles(α=0.5)        # Geometrically weighted four-cycles
```

### 5. Node Attribute Statistics

Statistics based on actor-level attributes for homophily and covariate effects.

```julia
NodeMatch(attr)                  # Binary: 1 if sender and receiver match
NodeMix(attr; sender_val, receiver_val)  # Specific attribute combinations
NodeDifference(attr; absolute=false)     # Numeric difference between actors
NodeSum(attr)                    # Sum of sender and receiver attributes
NodeProduct(attr)                # Product of sender and receiver attributes
SenderAttribute(attr)            # Effect of sender's attribute value
ReceiverAttribute(attr)          # Effect of receiver's attribute value
SenderCategorical(attr, val)     # Sender has specific categorical value
ReceiverCategorical(attr, val)   # Receiver has specific categorical value
```

## Usage

### Basic Example

```julia
using REM

# Create an event sequence
events = [
    Event(1, 2, 1.0),  # Actor 1 sends to Actor 2 at time 1.0
    Event(2, 1, 2.0),  # Actor 2 sends to Actor 1 at time 2.0
    Event(1, 3, 3.0),  # Actor 1 sends to Actor 3 at time 3.0
    Event(3, 2, 4.0),  # Actor 3 sends to Actor 2 at time 4.0
]
seq = EventSequence(events)

# Define statistics to model
stats = [
    Repetition(),        # Tendency to repeat past interactions
    Reciprocity(),       # Tendency to reciprocate
    SenderActivity(),    # Sender's past activity level
    ReceiverPopularity() # Receiver's past popularity
]

# Fit the model
result = fit_rem(seq, stats; n_controls=100, seed=42)

# View results
println(result)
```

### Result Structure

The `fit_rem` function returns a `REMResult` with:

- `coefficients::Vector{Float64}`: Estimated coefficients (log hazard ratios)
- `std_errors::Vector{Float64}`: Standard errors of coefficients
- `z_values::Vector{Float64}`: Z-statistics for hypothesis testing
- `p_values::Vector{Float64}`: Two-sided p-values
- `stat_names::Vector{String}`: Names of statistics in the model
- `n_events::Int`: Number of events in the model
- `log_likelihood::Float64`: Log-likelihood at convergence
- `converged::Bool`: Whether optimization converged

```julia
# Access results
coef(result)       # Coefficient estimates
stderror(result)   # Standard errors
coeftable(result)  # Full coefficient table as DataFrame
```

### Case-Control Sampling

REM.jl uses case-control sampling with stratified conditional logistic regression for efficient estimation:

```julia
# Configure sampler
sampler = CaseControlSampler(
    n_controls=100,           # Controls per case
    exclude_self_loops=true,  # Exclude i→i from risk set
    seed=42                   # Reproducibility
)

# Generate observations
obs = generate_observations(seq, stats, sampler)

# Fit model from observations
result = fit_rem(obs, [name(s) for s in stats])
```

### Temporal Decay

Support for exponential decay of network effects, where older events contribute less:

```julia
# Convert halflife to decay rate (e.g., 10 time units)
decay = halflife_to_decay(10.0)

# Create network state with decay
state = NetworkState(seq; decay=decay)

# Fit model with decay
result = fit_rem(seq, stats; n_controls=100, decay=decay)

# Utility functions
decay_to_halflife(decay)              # Convert decay rate back to halflife
compute_decay_weight(decay, elapsed)  # Compute weight for elapsed time
```

### Node Attributes

Define and use actor-level attributes:

```julia
# Create attribute with default value
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "M", 3 => "F"), "Unknown")
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0, 3 => 28.0), 0.0)

# Use in statistics
stats = [
    NodeMatch(gender),            # Homophily: same gender
    NodeDifference(age),          # Age difference effect
    SenderAttribute(age),         # Sender's age effect
    NodeMix(gender; sender_val="M", receiver_val="F")  # M→F pattern
]
```

### Loading Data

```julia
using DataFrames

# From DataFrame
df = DataFrame(
    sender = [1, 2, 1, 3],
    receiver = [2, 1, 3, 2],
    time = [1.0, 2.0, 3.0, 4.0]
)
seq = load_events(df)

# From CSV file
seq = load_events("events.csv")

# With string actor names (automatically converted to integer IDs)
df_names = DataFrame(
    sender = ["Alice", "Bob", "Alice"],
    receiver = ["Bob", "Alice", "Carol"],
    time = [1.0, 2.0, 3.0]
)
seq = load_events(df_names; actor_names=true)

# With event types and weights
df_typed = DataFrame(
    sender = [1, 2, 1],
    receiver = [2, 1, 3],
    time = [1.0, 2.0, 3.0],
    type = [:email, :email, :phone],
    weight = [1.0, 2.0, 1.5]
)
seq = load_events(df_typed; type_col=:type, weight_col=:weight)
```

### Computing Statistics Without Fitting

```julia
# Compute statistics for all events (without case-control sampling)
stats_df = compute_statistics(seq, stats)

# Access NetworkState for custom computation
state = NetworkState(seq; decay=0.0)
for event in seq
    # Compute statistics before updating state
    values = compute_all(stats, state, event.sender, event.receiver)

    # Update state with event
    update!(state, event)
end
```

## Utility Functions

```julia
# Time decay utilities
halflife_to_decay(halflife)          # Convert halflife to decay parameter
decay_to_halflife(decay)             # Convert decay to halflife

# NetworkState accessors
get_dyad_count(state, s, r)          # Events from s to r
get_undirected_count(state, i, j)    # Events between i and j (either direction)
get_out_degree(state, actor)         # Actor's out-degree
get_in_degree(state, actor)          # Actor's in-degree
get_out_neighbors(state, actor)      # Set of actors receiving from actor
get_in_neighbors(state, actor)       # Set of actors sending to actor
has_edge(state, s, r)                # Whether s→r exists

# Risk set utilities
n_dyads(risk_set)                    # Number of dyads in risk set
```

## Running Tests

```julia
include("test/runtests.jl")
```

## Documentation

For more detailed documentation, see:

- [Stable Documentation](https://statistical-network-analysis-with-Julia.github.io/REM.jl/stable/)
- [Development Documentation](https://statistical-network-analysis-with-Julia.github.io/REM.jl/dev/)

## References

1. Butts, C.T. (2008). A relational event framework for social action. *Sociological Methodology*, 38(1), 155-200.

2. Lerner, J., Lomi, A. (2020). Reliability of relational event model estimates under sampling: How to fit a relational event model to 360 million dyadic events. *Network Science*, 8(1), 97-135.

3. Lerner, J., Bussmann, M., Snijders, T.A.B., Brandes, U. (2013). Modeling frequency and type of interaction in event networks. *Corvinus Journal of Sociology and Social Policy*, 4(1), 3-32.

4. Brandes, U., Lerner, J., Snijders, T.A.B. (2009). Networks evolving step by step: Statistical analysis of dyadic event data. *2009 International Conference on Advances in Social Network Analysis and Mining*, 200-205.

5. Perry, P.O., Wolfe, P.J. (2013). Point process modelling for directed interaction networks. *Journal of the Royal Statistical Society: Series B*, 75(5), 821-849.

## License

MIT License - see [LICENSE](LICENSE) for details.
