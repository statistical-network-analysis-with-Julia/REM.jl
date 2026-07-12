# Getting Started

This tutorial walks through common use cases for REM.jl, from basic event modeling to advanced analysis.

## Installation

Install REM.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Network.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/NetworkDynamic.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/REM.jl")
```

## Basic Workflow

The typical REM.jl workflow consists of four steps:

1. **Load or create events** - Prepare your relational event data
2. **Define statistics** - Choose which effects to model
3. **Fit the model** - Estimate coefficients via case-control sampling
4. **Interpret results** - Analyze the fitted model

## Step 1: Create an Event Sequence

Events represent directed interactions between actors at specific times:

```julia
using REM

# Create individual events: Event(sender, receiver, time)
events = [
    Event(1, 2, 1.0),   # Actor 1 → Actor 2 at time 1.0
    Event(2, 1, 2.0),   # Actor 2 → Actor 1 at time 2.0
    Event(1, 3, 3.0),   # Actor 1 → Actor 3 at time 3.0
    Event(3, 2, 4.0),   # Actor 3 → Actor 2 at time 4.0
    Event(2, 3, 5.0),   # Actor 2 → Actor 3 at time 5.0
    Event(1, 2, 6.0),   # Actor 1 → Actor 2 at time 6.0 (repetition)
]

# Create an EventSequence (automatically sorted by time)
seq = EventSequence(events)

println("Number of events: ", length(seq))      # 6
println("Number of actors: ", seq.n_actors)     # 3
println("Actors: ", seq.actors)                  # Set([1, 2, 3])
```

### Loading from a DataFrame

More commonly, you'll load events from existing data:

```julia
using DataFrames

df = DataFrame(
    sender = [1, 2, 1, 3, 2, 1],
    receiver = [2, 1, 3, 2, 3, 2],
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
)

seq = load_events(df)
```

### Loading from CSV

<!-- skip-check -->
```julia
seq = load_events("path/to/events.csv")
```

### String Actor Names

When actors are identified by names:

```julia
df = DataFrame(
    sender = ["Alice", "Bob", "Alice"],
    receiver = ["Bob", "Alice", "Carol"],
    time = [1.0, 2.0, 3.0]
)

seq = load_events(df; actor_names=true)
# Actors are assigned numeric IDs internally
```

## Step 2: Define Statistics

Statistics capture different mechanisms that might drive event occurrence:

```julia
# Basic dyadic and degree statistics
stats = [
    Repetition(),           # Past events from sender to receiver
    Reciprocity(),          # Past events from receiver to sender
    SenderActivity(),       # Sender's overall activity (out-degree)
    ReceiverPopularity(),   # Receiver's overall popularity (in-degree)
]
```

### Exploring Available Statistics

REM.jl provides many statistics organized by type:

| Category | Statistics | Description |
|----------|------------|-------------|
| **Dyad** | `Repetition`, `Reciprocity`, `InertiaStatistic`, `RecencyStatistic` | History between focal dyad |
| **Degree** | `SenderActivity`, `ReceiverPopularity`, `TotalDegree`, `LogDegree` | Actor activity/popularity |
| **Triangle** | `TransitiveClosure`, `CyclicClosure`, `SharedSender`, `SharedReceiver` | Triadic closure patterns |
| **Four-Cycle** | `FourCycle`, `GeometricWeightedFourCycles` | Local clustering effects |
| **Node Attribute** | `AttributeMatch`, `NodeMix`, `NodeDifference`, `SenderAttribute` | Homophily and covariate effects |

### Example: Comprehensive Model

```julia
# Create node attributes
gender = NodeAttribute(:gender, Dict(1=>"M", 2=>"F", 3=>"M"), "Unknown")

# Build a comprehensive model
stats = [
    # Dyadic effects
    Repetition(),
    Reciprocity(),

    # Actor effects
    SenderActivity(),
    ReceiverPopularity(),

    # Structural effects
    TransitiveClosure(),
    CyclicClosure(),

    # Attribute effects
    AttributeMatch(gender),
]
```

## Step 3: Fit the Model

Use `fit_rem` to estimate the model:

```julia
result = fit_rem(seq, stats; n_controls=100, seed=42)
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_controls` | Control samples per event | 50-200 |
| `seed` | Random seed for reproducibility | Any integer |
| `decay` | Exponential decay rate | 0.0 (no decay) |
| `exclude_self_loops` | Exclude self-events from risk set | `true` |

### Choosing Number of Controls

More controls provide more accurate estimates but increase computation:

```julia
# Quick exploratory analysis
result_quick = fit_rem(seq, stats; n_controls=50, seed=42)

# Final analysis
result_final = fit_rem(seq, stats; n_controls=200, seed=42)
```

## Step 4: Interpret Results

The result object contains coefficient estimates and test statistics:

```julia
# Print formatted summary table
println(result)

# Output:
# Relational Event Model Results
# ==============================
# Events: 6, Observations: 606
# Log-likelihood: -12.3456
# Converged: true
#
# Coefficients:
# ------------------------------------------------------------
# Statistic                  Coef    Std.Err          z      P>|z|
# ------------------------------------------------------------
# repetition               0.4523     0.0812     5.5700     0.0000 ***
# reciprocity              0.3156     0.0923     3.4200     0.0006 ***
# ...
```

### Accessing Results Programmatically

```julia
# Coefficient vector
coef(result)

# Standard errors
stderror(result)

# Full table as DataFrame
df = coeftable(result)
```

### Interpreting Coefficients

Coefficients are **log-hazard ratios**:

| Coefficient | Interpretation |
|-------------|----------------|
| β > 0 | Statistic increases event rate |
| β < 0 | Statistic decreases event rate |
| β = 0 | No effect |
| exp(β) | Multiplicative effect on rate |

**Example interpretations:**

- `repetition = 0.5` → Each past s→r event increases rate by 65% (exp(0.5) ≈ 1.65)
- `reciprocity = 0.8` → Events are 2.2× more likely when r→s has occurred (exp(0.8) ≈ 2.23)
- `sender_activity = -0.1` → High-activity senders have slightly lower per-dyad rates

## Complete Example

```julia
using REM
using DataFrames

# Create event data with clear patterns
events = [
    # Initial interactions
    Event(1, 2, 1.0),
    Event(2, 3, 2.0),

    # Reciprocity: 2→1 after 1→2
    Event(2, 1, 3.0),

    # Transitive closure: 1→3 after 1→2 and 2→3
    Event(1, 3, 4.0),

    # Repetition: 1→2 again
    Event(1, 2, 5.0),

    # More activity
    Event(3, 1, 6.0),
    Event(2, 1, 7.0),
    Event(1, 3, 8.0),
]
seq = EventSequence(events)

# Define model with structural effects
stats = [
    Repetition(),
    Reciprocity(),
    SenderActivity(),
    ReceiverPopularity(),
    TransitiveClosure(),
]

# Fit with case-control sampling
result = fit_rem(seq, stats; n_controls=50, seed=123)

# View results
println(result)

# Get coefficient table
df = coeftable(result)
println("\nCoefficient Table:")
println(df)

# Check model convergence
if result.converged
    println("\nModel converged successfully")
    println("Log-likelihood: ", result.log_likelihood)
else
    println("\nWarning: Model did not converge")
end
```

## Working with Temporal Decay

For time-sensitive effects where older events matter less:

```julia
# Set halflife: events lose half their weight after 10 time units
decay = halflife_to_decay(10.0)

result = fit_rem(seq, stats;
    n_controls = 100,
    decay = decay,
    seed = 42
)
```

See [Temporal Decay](guide/decay.md) for more details.

## Comparing Models

```julia
# Model 1: Basic effects
stats1 = [Repetition(), Reciprocity()]

# Model 2: Add structural effects
stats2 = [Repetition(), Reciprocity(), TransitiveClosure(), CyclicClosure()]

result1 = fit_rem(seq, stats1; n_controls=100, seed=42)
result2 = fit_rem(seq, stats2; n_controls=100, seed=42)

# Compare log-likelihoods
println("Model 1 LL: ", result1.log_likelihood)
println("Model 2 LL: ", result2.log_likelihood)

# Higher log-likelihood (less negative) indicates better fit
if result2.log_likelihood > result1.log_likelihood
    println("Model 2 fits better")
end
```

## Best Practices

1. **Start simple**: Begin with basic dyadic effects before adding structural terms
2. **Check convergence**: Always verify `result.converged == true`
3. **Use adequate controls**: At least 50-100 controls per case
4. **Set random seed**: For reproducibility across runs
5. **Scale appropriately**: For large networks, use log-transformed degree statistics
6. **Sufficient events**: Rule of thumb - at least 10 events per parameter

## Next Steps

- Learn about [Events and Data](guide/events.md) for data handling
- Explore all [Statistics](guide/statistics.md) available
- Understand [Model Estimation](guide/estimation.md) in detail
- Use [Temporal Decay](guide/decay.md) for time-weighted effects
