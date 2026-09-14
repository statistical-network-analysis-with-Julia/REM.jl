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
- At $\Delta t = h$, where $h$ is the halflife: weight = 0.5

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

On the bundled WTC calls (an ordinal clock, so a halflife of 50 means "50
calls later an event counts half"):

```julia
using Networks
wtc = load_dataset(:wtc_police_calls)
events = [Event(wtc.events[k, 2], wtc.events[k, 3], Float64(wtc.events[k, 1]))
          for k in 1:size(wtc.events, 1)]
seq = EventSequence(events; actors=ActorSet(1:37))
stats = [Repetition(), Reciprocity(), SenderActivity(), ReceiverPopularity()]

result = fit_rem(seq, stats;
    n_controls = 37 * 36 - 1,          # the full risk set
    decay = halflife_to_decay(50.0)
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
    decay = halflife_to_decay(50.0)
)
```

## Decay with Different Time Types

### Numeric Timestamps

For numeric timestamps, decay is applied directly in the same units:

```julia
# If time is in hours
hourly = EventSequence([
    Event(1, 2, 0.0),   # Hour 0
    Event(2, 1, 24.0),  # Hour 24 (1 day later)
]; actors=ActorSet(1:2))

# Halflife of 24 hours = one day decay
decay = halflife_to_decay(24.0)
state = EventNetworkState(hourly; decay=decay)
```

### DateTime Timestamps

For DateTime, time differences are converted to **seconds** internally. Pass
the halflife as a `Dates.Period` and the conversion is done for you:

```julia
using Dates

stamped = EventSequence([
    Event(1, 2, DateTime(2024, 1, 1, 10, 0)),  # 10:00 AM
    Event(2, 1, DateTime(2024, 1, 1, 11, 0)),  # 11:00 AM (1 hour later)
]; actors=ActorSet(1:2))

# Halflife of 1 hour = 3600 seconds
decay = halflife_to_decay(Hour(1))
decay == halflife_to_decay(3600.0)    # true
state = EventNetworkState(stamped; decay=decay)
state                        # EventNetworkState{DateTime}(2 actors, 0 events absorbed, halflife 3600.0 (decay 0.0001925), current_time = 0000-01-01T00:00:00)
```

### Date Timestamps

For Date, differences are converted to days, then to **seconds** — so a
halflife or a window given as a bare number is in seconds too, and `window=2.0`
on a Date clock is two *seconds* (which expires everything and yields all-zero
statistics). Use a `Dates.Period` for both:

```julia
using Dates

daily = EventSequence([
    Event(1, 2, Date(2024, 1, 1)),   # Day 1
    Event(2, 1, Date(2024, 1, 8)),   # Day 8 (one week later)
]; actors=ActorSet(1:2))

# Halflife of 7 days = 7 * 86400 seconds
decay = halflife_to_decay(Day(7))
decay == halflife_to_decay(7.0 * 86400)    # true
state = EventNetworkState(daily; decay=decay)

# The sliding-window alternative, likewise: events older than two days expire
windowed = EventNetworkState(daily; window=Day(2))
windowed.window                            # 172800.0 — seconds
compute_statistics(daily, [Repetition()]; window=Day(2)).repetition   # [0.0, 0.0]
```

`window=Day(2)` (or any `Dates.Period`) is accepted on every entry point —
`EventNetworkState`, `compute_statistics`, `generate_observations`, `fit_rem`,
`control_draw_cov` — **on a `Date`/`DateTime` clock only**. On a numeric clock
(an event index, minutes, whatever the data count in) a `Period` is refused
with an `ArgumentError`: converting `Day(2)` to 172 800 clock units would
silently be no window at all, so pass a number in the clock's units instead.

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

pair = EventSequence([
    Event(1, 2, 0.0),   # First event at t=0
    Event(1, 2, 10.0),  # Second event at t=10
]; actors=ActorSet(1:2))

# Halflife of 10 time units
decay = halflife_to_decay(10.0)
state = EventNetworkState(pair; decay=decay)

# After first event
update!(state, pair[1])
println(get_dyad_count(state, 1, 2))  # 1.0

# After second event
# First event has decayed: 10 time units = 1 halflife → weight = 0.5
# Second event is fresh: weight = 1.0
update!(state, pair[2])
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

Decay affects every statistic that reads a **count**; a statistic that reads
the **adjacency** (whether a dyad ever had an event) or a **last-event time**
does not decay:

| Statistic | Effect of decay | Effect of a window |
|-----------|-----------------|--------------------|
| Repetition, Reciprocity, InertiaStatistic | decayed count of past s→r / r→s events | only events inside the window count |
| SenderActivity, ReceiverPopularity, TotalDegree, DegreeDifference, LogDegree | decayed degrees | windowed degrees |
| TransitiveClosure, CyclicClosure, SharedSender, SharedReceiver, CommonNeighbors, FourCycle (default, `weighted=true`) | `Σ_k agg(w(s,k), w(k,r))` over the **decayed** dyad weights: a two-path fades with its weaker leg | over the windowed weights; an expired leg removes the two-path |
| the same with `weighted=false` | **none** — the count of distinct third parties reads the adjacency, which never expires under decay | a third party drops out once the dyad's events have all left the window |
| GeometricWeightedTriads, GeometricWeightedFourCycles | none (count-based: `n` distinct closing paths) | as above |
| RecencyStatistic | none (reads the last-event time, which is a fact about the clock) | none |
| AttributeMatch, ActorMix, NodeDifference, … (attribute statistics) | none | none |

The distinction matters when comparing with eventnet: its triadic statistics
are the weighted (decaying) form, which is why that is the default since 0.2.

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
halflifes = [10.0, 50.0, 100.0, 500.0]
results = Dict()

for hl in halflifes
    decay = halflife_to_decay(hl)
    result = fit_rem(seq, stats; n_controls=37 * 36 - 1, decay=decay)
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
RecencyStatistic(transform=:inverse)         # 1/elapsed
RecencyStatistic(transform=:inverse_log)     # 1/log(1+elapsed)
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
    n_controls = 37 * 36 - 1,
    decay = halflife_to_decay(50.0),  # Global decay
)
```

This allows modeling:

- General decay of all network effects (via global decay)
- Specific recency effects for focal dyads (via RecencyStatistic)

## The Alternative: a Sliding Window

Halflife decay is the memory model eventnet offers, and the only one. The other memory model in use in relational-event software (e.g. `remstats`' `memory = "window"`) is a **sliding window**: an event older than `current_time − window` simply stops counting. REM.jl offers it as `window=` on `EventNetworkState`, `generate_observations`, `compute_statistics` and `fit_rem`:

```julia
using REM

win = EventSequence([Event(1, 2, 0.0), Event(2, 1, 1.0), Event(1, 2, 2.5), Event(1, 3, 4.0)];
                    actors=ActorSet(1:3))

state = EventNetworkState(win; window=2.0)
for e in win; update!(state, e); end     # the clock is now at t = 4
get_dyad_count(state, 1, 2)             # 1.0 — only the event at t = 2.5 is inside the window
get_dyad_count(state, 2, 1)             # 0.0 — t = 1 is 3.0 old
has_edge(state, 2, 1)                   # false: adjacency expires too
get_out_degree(state, 1)                # 2.0

df = compute_statistics(win, [Repetition(), SenderActivity()]; window=2.0)

# On the WTC calls: only the last 50 calls count
result = fit_rem(seq, stats; n_controls=37 * 36 - 1, window=50.0)
```

What a window does that decay does not:

| | Halflife decay (`decay=`) | Sliding window (`window=`) |
|---|---|---|
| Past event's weight | `exp(−λ·Δ)`, never exactly 0 | `1` while `Δ ≤ window`, then `0` |
| Adjacency (`has_edge`, neighbour sets, `weighted=false` closure counts) | never expires | expires with the dyad's last live event |
| Units | clock units (seconds for `Date`/`DateTime`) | the same |
| `RecencyStatistic`, `event_history` | unaffected | unaffected (the last-event time is a fact about the clock) |
| Cost | O(1) per event (lazy) | amortized O(1) per event (FIFO + expiry cursor) |

An event exactly `window` old still counts; `window=Inf` (or `nothing`, the default) is no window and gives the identical design row for row. The two models are **mutually exclusive**: `decay > 0` together with a window is an `ArgumentError`. The windowed counts are pinned against `survival::clogit` on a design rebuilt in plain R (`test/fixtures/rem_eventnet.toml`), alongside the decayed ones.

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

A calendar clock: 80 emails among 6 colleagues, one every three hours (a
deterministic stand-in for a loaded mailbox), with a one-week halflife in
**seconds**:

```julia
using REM
using Dates

emails = Event{DateTime}[]
for k in 1:80
    s = 1 + (k % 6); r = 1 + ((5k + 2) % 6); r == s && (r = 1 + (r % 6))
    k % 4 == 0 && ((s, r) = (r, s))
    push!(emails, Event(s, r, DateTime(2024, 1, 1) + Hour(3k)))
end
mail = EventSequence(emails; actors=ActorSet(1:6))

# Define statistics
mail_stats = [
    Repetition(),
    Reciprocity(),
    SenderActivity(),
    ReceiverPopularity(),
]

# Model with 1-week halflife (in seconds)
one_week_seconds = 7 * 24 * 60 * 60
decay = halflife_to_decay(Float64(one_week_seconds))

result = fit_rem(mail, mail_stats;
    n_controls = 29,          # 6 actors: 30 dyads, the full risk set
    decay = decay,
)

println(result)
```

## Computational Notes

- Decay is applied incrementally as `update!` is called
- Time differences are computed relative to `state.current_time`
- Very fast decay (small halflife) may reduce effective sample size
- Very slow decay (large halflife) approaches no-decay case
