# Temporal Decay

REM.jl supports exponential decay of network effects, allowing past events to have diminishing influence over time. This captures the intuition that recent interactions are more relevant than older ones.

## Why Use Decay?

In many applications, recent events are more relevant than older ones:

- A communication last week matters more than one from a year ago
- Relationships may weaken without recent interaction
- Network effects fade over time
- Memory and attention are finite

Temporal decay captures this by down-weighting older events when computing statistics.

## The Exponential Decay Model

The weight of an event decays exponentially with elapsed time:

$$w(t) = \exp(-\lambda \cdot \Delta t)$$

Where:

- $\lambda$ is the decay rate (larger = faster decay)
- $\Delta t$ is the elapsed time since the event
- At $\Delta t = 0$: weight = 1.0 (full weight)
- At $\Delta t = $ halflife: weight = 0.5

## Setting the Decay Rate

### Using Halflife (Recommended)

The most intuitive approach is to specify a halflife - the time after which an event has half its original weight:

```julia
using REM

# Events lose half their weight after 10 time units
decay = halflife_to_decay(10.0)
```

### Direct Decay Rate

Alternatively, specify the decay rate directly:

```julia
# Decay rate of 0.1 per time unit
decay = 0.1
```

### Converting Between Forms

```julia
# Halflife to decay rate
decay = halflife_to_decay(10.0)

# Decay rate to halflife
halflife = decay_to_halflife(decay)   # 10.0 again

# Relationship: decay = log(2) / halflife
```

## Using Decay in Models

### With fit_rem

```julia
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
          Event(3, 2, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
seq = EventSequence(events)
stats = [Repetition(), Reciprocity()]

result = fit_rem(seq, stats;
    n_controls = 100,
    decay = halflife_to_decay(10.0),
    seed = 42
)
```

### With EventNetworkState

```julia
# Create state with decay
state = EventNetworkState(seq; decay=halflife_to_decay(10.0))

# Process events - decay is applied automatically as time advances
for event in seq
    update!(state, event)
end
```

### With generate_observations

```julia
sampler = CaseControlSampler(n_controls=100, seed=42)
obs = generate_observations(seq, stats, sampler;
    decay = halflife_to_decay(10.0)
)
```

## Decay with Different Time Types

### Numeric Timestamps

For numeric timestamps, decay is applied directly in the same units:

```julia
# If time is in hours
events = [
    Event(1, 2, 0.0),   # Hour 0
    Event(2, 1, 24.0),  # Hour 24 (1 day later)
]
seq = EventSequence(events)

# Halflife of 24 hours = one day decay
decay = halflife_to_decay(24.0)
state = EventNetworkState(seq; decay=decay)
```

### DateTime Timestamps

For DateTime, time differences are converted to **seconds** internally:

```julia
using Dates

events = [
    Event(1, 2, DateTime(2024, 1, 1, 10, 0)),  # 10:00 AM
    Event(2, 1, DateTime(2024, 1, 1, 11, 0)),  # 11:00 AM (1 hour later)
]
seq = EventSequence(events)

# Halflife of 1 hour = 3600 seconds
decay = halflife_to_decay(3600.0)
state = EventNetworkState(seq; decay=decay)
```

### Date Timestamps

For Date, differences are converted to days, then to **seconds**:

```julia
using Dates

events = [
    Event(1, 2, Date(2024, 1, 1)),   # Day 1
    Event(2, 1, Date(2024, 1, 8)),   # Day 8 (one week later)
]
seq = EventSequence(events)

# Halflife of 7 days = 7 * 86400 seconds
decay = halflife_to_decay(7.0 * 86400)
state = EventNetworkState(seq; decay=decay)
```

## How Decay Affects Statistics

### Dyad Counts

Without decay:

<!-- skip-check -->
```julia
get_dyad_count(state, s, r)  # = total number of s→r events
```

With decay:

<!-- skip-check -->
```julia
get_dyad_count(state, s, r)  # = Σ exp(-λ × elapsed_time_i)
```

### Example

```julia
using REM

events = [
    Event(1, 2, 0.0),   # First event at t=0
    Event(1, 2, 10.0),  # Second event at t=10
]
seq = EventSequence(events)

# Halflife of 10 time units
decay = halflife_to_decay(10.0)
state = EventNetworkState(seq; decay=decay)

# After first event
update!(state, seq[1])
println(get_dyad_count(state, 1, 2))  # 1.0

# After second event
# First event has decayed: 10 time units = 1 halflife → weight = 0.5
# Second event is fresh: weight = 1.0
update!(state, seq[2])
println(get_dyad_count(state, 1, 2))  # 1.5 (0.5 + 1.0)
```

### Degrees

Out-degree and in-degree are similarly weighted:

```julia
# Without decay: count of events sent
# With decay: Σ exp(-λ × elapsed) × event_weight
get_out_degree(state, 1)
get_in_degree(state, 1)
```

### All Statistics

Decay affects **all** statistics that depend on counts:

| Statistic | Effect of Decay |
|-----------|-----------------|
| Repetition | Weighted count of past s→r events |
| Reciprocity | Weighted count of past r→s events |
| SenderActivity | Weighted out-degree |
| ReceiverPopularity | Weighted in-degree |
| TransitiveClosure | Weighted count of two-paths |
| etc. | All use weighted counts |

## Choosing the Right Halflife

### Domain Guidelines

The appropriate halflife depends on your domain:

| Domain | Typical Halflife |
|--------|------------------|
| Real-time chat | Minutes to hours |
| Email communication | Hours to days |
| Social media | Days to weeks |
| Business relationships | Weeks to months |
| Organizational ties | Months to years |
| Stable institutions | Years |

### Practical Guidelines

1. **Domain knowledge**: What timeframe makes interactions "stale"?
2. **Event frequency**: Halflife should be comparable to typical inter-event times
3. **Observation period**: Halflife should be much smaller than total observation time
4. **Sensitivity analysis**: Try different values and compare results

### Sensitivity Analysis

```julia
halflifes = [1.0, 5.0, 10.0, 50.0, 100.0]
results = Dict()

for hl in halflifes
    decay = halflife_to_decay(hl)
    result = fit_rem(seq, stats; n_controls=100, decay=decay, seed=42)
    results[hl] = coef(result)
    println("Halflife $hl: ", round.(coef(result), digits=3))
end
```

## Recency Statistic vs Global Decay

There are two ways to model time effects:

### Global Decay

Affects **all** statistics through EventNetworkState:

```julia
# All statistics use decayed counts
result = fit_rem(seq, stats; decay=halflife_to_decay(10.0))
```

### RecencyStatistic

A **specific** statistic measuring time since last dyad event:

```julia
RecencyStatistic(transform=:inverse)     # 1/elapsed
RecencyStatistic(transform=:log)         # 1/log(1+elapsed)
RecencyStatistic(transform=:exp_decay, decay=0.1)  # exp(-0.1*elapsed)
```

### Key Differences

| Aspect | Global Decay | RecencyStatistic |
|--------|--------------|------------------|
| Affects | All statistics | Only recency |
| Measures | Weighted history | Time to last event |
| Parameters | Decay rate | Transform type |
| Use case | General fading | Dyad-specific timing |

### Combining Both

You can use both simultaneously:

```julia
stats = [
    Repetition(),           # Affected by global decay
    Reciprocity(),          # Affected by global decay
    RecencyStatistic(),     # Additional dyad-specific recency
    SenderActivity(),       # Affected by global decay
]

result = fit_rem(seq, stats;
    n_controls = 100,
    decay = halflife_to_decay(10.0),  # Global decay
    seed = 42
)
```

This allows modeling:

- General decay of all network effects (via global decay)
- Specific recency effects for focal dyads (via RecencyStatistic)

## No Decay (Default)

When decay = 0.0 (the default), all past events have equal weight:

```julia
# These are equivalent
result = fit_rem(seq, stats; n_controls=100)
result = fit_rem(seq, stats; n_controls=100, decay=0.0)
```

This is appropriate when:

- All historical interactions are equally relevant
- The observation period is short
- You want to maximize statistical power

## Example: Email Network

```julia
using REM
using Dates

# Load email data with DateTime timestamps
events = [
    Event(1, 2, DateTime(2024, 1, 1, 9, 0)),
    Event(2, 1, DateTime(2024, 1, 1, 9, 30)),
    Event(1, 3, DateTime(2024, 1, 1, 14, 0)),
    # ... more events
]
seq = EventSequence(events)

# Define statistics
stats = [
    Repetition(),
    Reciprocity(),
    SenderActivity(),
    ReceiverPopularity(),
    TransitiveClosure(),
]

# Model with 1-week halflife (in seconds)
one_week_seconds = 7 * 24 * 60 * 60
decay = halflife_to_decay(Float64(one_week_seconds))

result = fit_rem(seq, stats;
    n_controls = 100,
    decay = decay,
    seed = 42
)

println(result)
```

## Computational Notes

- Decay is applied incrementally as `update!` is called
- Time differences are computed relative to `state.current_time`
- Very fast decay (small halflife) may reduce effective sample size
- Very slow decay (large halflife) approaches no-decay case
