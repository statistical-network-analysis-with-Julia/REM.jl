# Statistics

Statistics in REM.jl capture different mechanisms that may drive event occurrence. All statistics implement a common interface and can be freely combined in models.

## Statistics Interface

All statistics implement two methods:

<!-- skip-check -->
```julia
compute(stat, state, sender, receiver) -> Float64
name(stat) -> String
```

The `compute` function calculates the statistic value for a potential event from `sender` to `receiver`, given the current `EventNetworkState`.

## Statistic Categories

REM.jl organizes statistics into five categories:

| Type | Description | Examples |
|------|-------------|----------|
| `DyadStatistic` | History between sender and receiver | Repetition, Reciprocity |
| `DegreeStatistic` | Actor activity and popularity | SenderActivity, ReceiverPopularity |
| `TriangleStatistic` | Triadic closure effects | TransitiveClosure, CyclicClosure |
| `FourCycleStatistic` | Four-cycle clustering effects | FourCycle |
| `NodeStatistic` | Node attribute effects | AttributeMatch, NodeDifference |

## Dyad Statistics

These capture the history of events between the focal sender-receiver pair.

### Repetition

Tendency to repeat past interactions:

```julia
using REM

# Count of past s→r events (directed)
Repetition()

# Count of s↔r events in either direction (undirected)
Repetition(directed=false)
```

**Interpretation**: A positive coefficient indicates actors tend to interact repeatedly with the same partners.

### Reciprocity

Tendency to reciprocate interactions:

```julia
# Count of past r→s events
Reciprocity()
```

**Interpretation**: A positive coefficient indicates actors tend to respond to those who contacted them.

### Inertia

Combined repetition and reciprocity:

```julia
# Default: equal weights
InertiaStatistic()

# Custom weights
InertiaStatistic(repetition_weight=2.0, reciprocity_weight=1.0)
```

**Formula**: `inertia = rep_weight × repetition + recip_weight × reciprocity`

### Recency

How recently the last event occurred on the dyad, as a **decreasing**
transform of the elapsed time `Δ` since it (`0.0` when there is no prior
event):

```julia
# Inverse of elapsed time since last s→r event
RecencyStatistic()

# With different transforms
RecencyStatistic(transform=:inverse)        # 1/Δ (default)
RecencyStatistic(transform=:inverse_log)    # 1/log(1+Δ)  (called :log before 0.2 — it is not a log transform)
RecencyStatistic(transform=:exp_decay, decay=0.1)  # exp(-0.1 Δ)

# The last event in EITHER direction between s and r
RecencyStatistic(directed=false)
```

**Interpretation**: Captures whether recent contact increases likelihood of future contact, beyond the cumulative count. Elapsed time is in the clock's units for numeric timestamps and in **seconds** for `Date`/`DateTime`. Unlike the state-level `decay`/`window` (below), recency reads the dyad's last-event time, which no window expires.

### Dyad Covariate

Pre-specified dyad-level covariate:

```julia
# Geographic distance between actors
distances = Dict(
    (1,2) => 10.0,
    (1,3) => 20.0,
    (2,3) => 15.0
)
DyadCovariate(distances; default=100.0, name="distance")
```

**Use case**: Include exogenous dyad-level variables like geographic distance, organizational distance, or prior relationship strength.

## Degree Statistics

These capture actor activity (out-degree) and popularity (in-degree).

### Activity (Out-degree)

```julia
SenderActivity()     # Sender's past sending activity
ReceiverActivity()   # Receiver's past sending activity
```

**Interpretation**:

- `SenderActivity` > 0: Active senders are more likely to send (Matthew effect)
- `ReceiverActivity` > 0: Active people are more likely to be contacted

### Popularity (In-degree)

```julia
SenderPopularity()   # Sender's past receiving (popularity)
ReceiverPopularity() # Receiver's past receiving (popularity)
```

**Interpretation**:

- `ReceiverPopularity` > 0: Popular actors continue to attract interactions
- `SenderPopularity` > 0: Popular actors are more likely to initiate contact

### Total Degree

```julia
TotalDegree(role=:sender)    # Sender's in + out degree
TotalDegree(role=:receiver)  # Receiver's in + out degree
```

### Degree Difference

```julia
DegreeDifference()                        # Sender out-degree - Receiver out-degree
DegreeDifference(degree_type=:in)         # In-degree difference
DegreeDifference(degree_type=:total)      # Total degree difference
DegreeDifference(absolute=true)           # |difference|
```

**Interpretation**: Tests whether events flow from high-degree to low-degree actors (or vice versa).

### Log Degree

For networks where degree effects may be non-linear:

```julia
LogDegree(role=:sender, degree_type=:out)   # log(1 + sender out-degree)
LogDegree(role=:receiver, degree_type=:in)  # log(1 + receiver in-degree)
```

**Use case**: Prevents very high-degree nodes from dominating the model.

## Triangle Statistics

These capture triadic closure - the tendency for events to "close" triangles in the network.

### eventnet's definition: weighted two-paths, aggregated then summed

REM.jl is a port of [eventnet](https://github.com/juergenlerner/eventnet), and since 0.2.0 the triadic statistics **default to eventnet's definition**. For a candidate event `s → r`, every two-path through a third actor `k ∉ {s, r}` contributes the two (decayed, event-weighted) dyad counts of its legs combined by an **aggregation function**, and the values of the parallel two-paths are added:

```math
\text{TransitiveClosure}(s, r) = \sum_{k \ne s, r} \operatorname{agg}\big(w(s, k),\; w(k, r)\big)
```

with `agg = min` by default — eventnet's choice — or `max`, `sum`, `product`. A two-path exists only when both of its legs have a positive count, so with `:sum` or `:max` a missing leg contributes nothing rather than the other leg's weight. The weights `w` are the state's counts under whichever memory model is in force — decayed under `decay=`, restricted to the live events under `window=` (see [Temporal Decay](decay.md)). The pre-0.2 definition, the **number of distinct** third parties `k` closing the pattern (read off the adjacency, so it never decays and, under a window, expires only with the dyad's last live event), is `weighted=false`.

```julia
using REM
# 1→2 twice, 2→3 once: one two-path 1→2→3 with weights (2, 1)
seq = EventSequence([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end

compute(TransitiveClosure(), state, 1, 3)                       # min(2, 1) = 1.0 (eventnet's default)
compute(TransitiveClosure(aggregation=:max), state, 1, 3)       # 2.0
compute(TransitiveClosure(aggregation=:sum), state, 1, 3)       # 3.0
compute(TransitiveClosure(aggregation=:product), state, 1, 3)   # 2.0
compute(TransitiveClosure(weighted=false), state, 1, 3)         # 1.0 — one third party (the 0.1 definition)
```

Under halflife decay the weighted forms fade while the count does not:

```julia
decayed = EventNetworkState(seq; decay=halflife_to_decay(1.0))
update!(decayed, Event(1, 2, 0.0))
update!(decayed, Event(2, 3, 1.0))          # read at t = 1: w(1,2) = 0.5, w(2,3) = 1
compute(TransitiveClosure(), decayed, 1, 3)                 # 0.5
compute(TransitiveClosure(weighted=false), decayed, 1, 3)   # 1.0
```

The same `weighted`/`aggregation` keywords apply to `CyclicClosure`, `SharedSender`, `SharedReceiver`, `CommonNeighbors` (on the undirected counts) and `FourCycle` (over its three weights). `GeometricWeightedTriads` and `GeometricWeightedFourCycles` are count-based by construction. All of them are pinned against `survival::clogit` on a design rebuilt in plain R (`test/fixtures/rem_eventnet.toml`), under each aggregation, with decay and with a window.

### Transitive Closure

Events that close a two-path s→k→r:

```julia
# Σ_k min(w(s,k), w(k,r)) — eventnet's definition
TransitiveClosure()

# Other aggregations of the two weights
TransitiveClosure(aggregation=:sum)

# Count of k where s→k and k→r (adjacency only)
TransitiveClosure(weighted=false)
```

Visual representation:

```text
  k
 ↗ ↘
s → r  ← new event closes the triangle
```

**Interpretation**: "Friends of friends become friends" - actors are more likely to interact if they share common contacts.

### Cyclic Closure

Events that form a cycle r→k→s:

```julia
# Σ_k min(w(r,k), w(k,s))
CyclicClosure()
CyclicClosure(weighted=false)   # count of k where r→k and k→s
```

Visual representation:

```text
  k
 ↗ ↙
r ← s  ← new s→r event closes the cycle
```

**Interpretation**: Tendency to complete directed cycles (common in reciprocal exchange networks).

### Shared Sender

Common sender k who sent to both s and r:

```julia
# Σ_k min(w(k,s), w(k,r))
SharedSender()
SharedSender(weighted=false)    # count of k where k→s and k→r
```

**Interpretation**: Actors contacted by the same third party are more likely to interact.

### Shared Receiver

Common receiver k who received from both s and r:

```julia
# Σ_k min(w(s,k), w(r,k))
SharedReceiver()
SharedReceiver(weighted=false)  # count of k where s→k and r→k
```

**Interpretation**: Actors who contacted the same third party are more likely to interact.

### Common Neighbors (Undirected)

The SYM form: `k` is a common neighbour when it has had an event with `s` in either direction and with `r` in either direction, and the weights are the **undirected** counts (both directions added):

```julia
# Σ_k min(u(s,k), u(k,r))
CommonNeighbors()
CommonNeighbors(aggregation=:sum)
CommonNeighbors(weighted=false)   # number of common neighbours
```

### Geometrically Weighted Triads

Down-weights additional shared partners (similar to GWESP in ERGM). With `n` the number of distinct third parties closing the pattern, the value is `exp(α)·(1 − (1 − exp(−α))^n)`: `0` for none, `1` for one, and each further partner adds less than the one before.

```julia
GeometricWeightedTriads(closure_type=:transitive, alpha=0.5)
GeometricWeightedTriads(closure_type=:cyclic, alpha=0.5)
GeometricWeightedTriads(closure_type=:shared_sender, alpha=0.5)
GeometricWeightedTriads(closure_type=:shared_receiver, alpha=0.5)
```

**Parameter α**: Controls how quickly additional shared partners are down-weighted. Lower α = stronger down-weighting.

## Four-Cycle Statistics

These capture clustering through pairs of intermediaries, forming four-node structures.

### Four-Cycle

Various four-cycle configurations. As for the triangles, the default is eventnet's weighted form — the three dyad weights of each closing three-path `(j, k)` combined by `aggregation` (`:min`) and the parallel three-paths added — and `weighted=false` counts the distinct three-paths:

```julia
# Different cycle types
FourCycle(cycle_type=:out_out)  # s→j←k→r (shared out-neighbor pattern)
FourCycle(cycle_type=:in_in)    # s←j→k←r (shared in-neighbor pattern)
FourCycle(cycle_type=:out_in)   # s→j→k→r (chain pattern)
FourCycle(cycle_type=:in_out)   # s←j←k←r (reverse chain)
FourCycle(cycle_type=:mixed)    # All patterns combined

FourCycle(cycle_type=:out_out, aggregation=:sum)   # sum of the three weights per three-path
FourCycle(cycle_type=:out_out, weighted=false)     # number of closing three-paths
```

Visual representation (out_out):

```text
s → j
    ↑
    k → r
```

**Interpretation**: Captures higher-order clustering beyond triangles.

### Geometrically Weighted Four-Cycles

`exp(α)·(1 − (1 − exp(−α))^n)` with `n` the number of distinct closing three-paths:

```julia
GeometricWeightedFourCycles(cycle_type=:out_out, alpha=0.5)
```

### Participation shifts live in Relevent.jl

Participation shifts (`PSAB-BA`, `PSAB-BY`, ... — "A calls B, then B calls A back") are **not** eventnet statistics: they are relevent's, and they live in [Relevent.jl](https://github.com/statistical-network-analysis-with-Julia/Relevent.jl) as `PShift`, together with the decay-weighted `LocalInertia`, `PriorInteraction`, `Momentum` and the capacity statistics. Because every statistic in the ecosystem is a method of the shared `compute` generic, a Relevent statistic goes straight into `fit_rem` beside REM's own — `fit_rem(seq, [Repetition(), PShift(:AB_BA)])` — and reads the state's event log through `needs_history` (see [`needs_history`](@ref)). Relevent.jl's test suite pins that dispatch; REM.jl cannot (Relevent depends on REM, not the reverse).

## Node Attribute Statistics

These incorporate actor-level attributes for homophily and covariate effects.

### Node attributes: no silent zero-fill

A `NodeAttribute` built **without** a default refuses to invent a value for an actor it lacks — a statistic reading such an actor throws an `ArgumentError` naming the attribute and the actor. A default is the explicit opt-in:

```julia
gender = NodeAttribute(:gender, Dict(1=>"M", 2=>"F", 3=>"M"))          # every actor listed; no default
gender_filled = NodeAttribute(:gender, Dict(1=>"M", 2=>"F"), "Unknown")  # actor 3 reads "Unknown", deliberately
has_default(gender), has_default(gender_filled)                        # (false, true)
```

Filling a missing covariate silently (with `0.0`, say) is the classic trap — a zero that looks like data — which is why the fill has to be asked for. Mismatches are caught at construction: `ActorMix(gender, 1, 2)` (an `Int` for a `String` attribute) and `NodeSum(gender)` (a numeric statistic on a categorical attribute) both throw an `ArgumentError` that says which statistics handle the case.

### Homophily (Node Match)

Indicator for matching attributes:

```julia
gender = NodeAttribute(:gender, Dict(1=>"M", 2=>"F", 3=>"M"), "Unknown")

# Returns 1.0 if sender and receiver have same gender, 0.0 otherwise
AttributeMatch(gender)
```

**Interpretation**: Positive coefficient = homophily (like attracts like).

### Mixing Patterns (Node Mix)

Indicator for specific sender-receiver attribute combinations:

```julia
# Returns 1.0 if sender is "M" and receiver is "F"
ActorMix(gender, "M", "F")
```

**Use case**: Test for asymmetric patterns (e.g., do men contact women more than vice versa?).

### Attribute Difference

For numeric attributes:

```julia
age = NodeAttribute(:age, Dict(1=>25.0, 2=>30.0), 0.0)

# sender_age - receiver_age
NodeDifference(age)

# |sender_age - receiver_age|
NodeDifference(age; absolute=true)
```

**Interpretation**:

- `NodeDifference` < 0: Events flow from younger to older (or similar attribute direction)
- `NodeDifference(absolute=true)` < 0: Events are more likely between similar actors

### Attribute Sum and Product

```julia
NodeSum(age)      # sender_age + receiver_age
NodeProduct(age)  # sender_age * receiver_age
```

### Main Effects

Include attribute as a main effect on sender or receiver:

```julia
# Numeric attributes
SenderAttribute(age)    # Sender's age affects rate
ReceiverAttribute(age)  # Receiver's age affects rate

# Categorical attributes (indicator for specific value)
SenderCategorical(gender, "M")    # 1.0 if sender is "M"
ReceiverCategorical(gender, "F")  # 1.0 if receiver is "F"
```

## Using Statistics in Practice

### Building a Model

```julia
# Create node attributes
gender = NodeAttribute(:gender, Dict(1=>"M", 2=>"F", 3=>"M"), "Unknown")
tenure = NodeAttribute(:tenure, Dict(1=>5.0, 2=>10.0, 3=>3.0), 0.0)

# Build comprehensive model
stats = [
    # Dyadic effects
    Repetition(),
    Reciprocity(),

    # Degree effects
    SenderActivity(),
    ReceiverPopularity(),

    # Structural effects
    TransitiveClosure(),
    CyclicClosure(),

    # Attribute effects
    AttributeMatch(gender),
    NodeDifference(tenure; absolute=true),
]
```

### Computing Statistics Manually

```julia
# A small demo sequence
demo_events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
               Event(3, 2, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
seq = EventSequence(demo_events)

# Create network state
state = EventNetworkState(seq)

# Process some events to build history
for i in 1:5
    update!(state, seq[i])
end

# Compute a single statistic for a potential event (sender 1, receiver 2)
rep = Repetition()
value = compute(rep, state, 1, 2)

# Compute all statistics
values = compute_all(stats, state, 1, 2)
```

### Custom Statistic Names

All statistics accept a `name` parameter for custom naming:

```julia
Repetition(name="past_interactions")
TransitiveClosure(name="friends_of_friends")
AttributeMatch(gender; name="same_gender")
```

This is useful when fitting multiple versions of the same statistic:

```julia
stats = [
    Repetition(directed=true, name="repetition_directed"),
    Repetition(directed=false, name="repetition_undirected"),
]
```

## Statistic Sets

For convenient handling of multiple statistics:

```julia
ss = StatisticSet([
    Repetition(),
    Reciprocity(),
    TransitiveClosure(),
])

# Access
length(ss)     # 3
ss[1]          # Repetition()
ss.names       # ["repetition", "reciprocity", "transitive_closure"]
ss             # StatisticSet(3 statistics: repetition, reciprocity, transitive_closure)

# Compute all (for candidate sender 1, receiver 2)
values = compute_all(ss, state, 1, 2)

# In-place variant for hot loops
dest = Vector{Float64}(undef, length(ss))
compute_all!(dest, ss, state, 1, 2)
```

`StatisticSet` stores the statistics as a tuple, so `compute_all` on a
set compiles to statically dispatched calls per statistic — no dynamic
dispatch in the observation-generation/likelihood inner loop. Passing a
plain `Vector` of statistics to `generate_observations`,
`compute_statistics`, or `fit_rem` still works: it is converted to a
`StatisticSet` internally. If you call these repeatedly with the same
statistics, construct the `StatisticSet` once and reuse it to avoid
recompiling for each new tuple type.

## Choosing Statistics

### By Research Question

| Question | Statistics |
|----------|------------|
| Do past interactions predict future ones? | Repetition, Reciprocity |
| Is there preferential attachment? | SenderActivity, ReceiverPopularity |
| Does the network cluster? | TransitiveClosure, CyclicClosure, FourCycle |
| Is there homophily? | AttributeMatch, NodeDifference |
| Do attributes affect sending/receiving? | SenderAttribute, ReceiverAttribute |

### Best Practices

1. **Start with dyadic effects**: Repetition and Reciprocity are almost always relevant
2. **Add degree effects**: Control for baseline activity/popularity differences
3. **Test structural effects carefully**: Triadic statistics can be correlated with degree
4. **Include relevant attributes**: Based on domain knowledge
5. **Avoid multicollinearity**: Don't include highly correlated statistics
6. **Use log transforms for degree**: Especially in networks with high-degree hubs
