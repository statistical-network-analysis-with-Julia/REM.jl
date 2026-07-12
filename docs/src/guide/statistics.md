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

How recently the last event occurred on the dyad:

```julia
# Inverse of elapsed time since last s→r event
RecencyStatistic()

# With different transforms
RecencyStatistic(transform=:inverse)    # 1/elapsed (default)
RecencyStatistic(transform=:log)        # 1/log(1+elapsed)
RecencyStatistic(transform=:exp_decay, decay=0.1)  # exp(-0.1*elapsed)
```

**Interpretation**: Captures whether recent contact increases likelihood of future contact, beyond the cumulative count.

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

### Transitive Closure

Events that close a two-path s→k→r:

```julia
# Count of k where s→k and k→r
TransitiveClosure()

# Weighted by minimum edge weight
TransitiveClosure(weighted=true)
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
# Count of k where r→k and k→s
CyclicClosure()
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
# Count of k where k→s and k→r
SharedSender()
```

**Interpretation**: Actors contacted by the same third party are more likely to interact.

### Shared Receiver

Common receiver k who received from both s and r:

```julia
# Count of k where s→k and r→k
SharedReceiver()
```

**Interpretation**: Actors who contacted the same third party are more likely to interact.

### Common Neighbors (Undirected)

```julia
# Any common neighbor regardless of direction
CommonNeighbors()
```

### Geometrically Weighted Triads

Down-weights additional shared partners (similar to GWESP in ERGM):

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

Various four-cycle configurations:

```julia
# Different cycle types
FourCycle(cycle_type=:out_out)  # s→j←k→r (shared out-neighbor pattern)
FourCycle(cycle_type=:in_in)    # s←j→k←r (shared in-neighbor pattern)
FourCycle(cycle_type=:out_in)   # s→j→k→r (chain pattern)
FourCycle(cycle_type=:in_out)   # s←j←k←r (reverse chain)
FourCycle(cycle_type=:mixed)    # All patterns combined
```

Visual representation (out_out):

```text
s → j
    ↑
    k → r
```

**Interpretation**: Captures higher-order clustering beyond triangles.

### Geometrically Weighted Four-Cycles

```julia
GeometricWeightedFourCycles(cycle_type=:out_out, alpha=0.5)
```

## Node Attribute Statistics

These incorporate actor-level attributes for homophily and covariate effects.

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
NodeMix(gender, "M", "F")
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
# Create network state
state = EventNetworkState(seq)

# Process some events to build history
for i in 1:5
    update!(state, seq[i])
end

# Compute a single statistic for a potential event
rep = Repetition()
value = compute(rep, state, sender_id, receiver_id)

# Compute all statistics
values = compute_all(stats, state, sender_id, receiver_id)
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

# Compute all
values = compute_all(ss, state, sender, receiver)

# In-place variant for hot loops
dest = Vector{Float64}(undef, length(ss))
compute_all!(dest, ss, state, sender, receiver)
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
