# Events and Data

This guide covers how to work with relational event data in REM.jl.

## Events

An `Event` represents a single directed interaction between actors:

```julia
using REM

# Basic event: sender, receiver, time
e = Event(1, 2, 1.0)

# With optional event type and weight
e = Event(1, 2, 1.0; eventtype=:email, weight=2.0)
```

### Event Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `sender` | `Int` | ID of the event sender | Required |
| `receiver` | `Int` | ID of the event receiver | Required |
| `time` | `T` | Timestamp of the event | Required |
| `eventtype` | `Symbol` | Category of the event | `:event` |
| `weight` | `Float64` | Weight/magnitude | `1.0` |

### Accessing Event Data

```julia
e = Event(1, 2, 3.5; eventtype=:phone, weight=2.0)

e.sender      # 1
e.receiver    # 2
e.time        # 3.5
e.eventtype   # :phone
e.weight      # 2.0
```

### Timestamp Types

REM.jl supports various timestamp types:

```julia
using Dates

# Numeric timestamps
Event(1, 2, 1.0)                              # Float64
Event(1, 2, 1)                                # Int

# Calendar timestamps
Event(1, 2, DateTime(2024, 1, 15, 10, 30))    # DateTime
Event(1, 2, Date(2024, 1, 15))                # Date
```

All events in a sequence must have the same timestamp type.

## Event Sequences

An `EventSequence` is a time-sorted collection of events:

```julia
events = [
    Event(1, 2, 3.0),  # Not in chronological order...
    Event(2, 1, 1.0),
    Event(1, 3, 2.0),
]
seq = EventSequence(events)  # Automatically sorted by time

# After sorting: times are [1.0, 2.0, 3.0]
```

### Accessing Sequence Data

```julia
# Basic access
seq[1]              # First event (earliest time)
seq[end]            # Last event (latest time)
length(seq)         # Number of events

# Metadata
seq.n_actors        # Number of unique actors
seq.actors          # Set of actor IDs
seq.eventtypes      # Set of event types

# Iteration
for event in seq
    println(event.sender, " → ", event.receiver)
end

# Collect times
times = [e.time for e in seq]
```

### Adding Events

Events are inserted maintaining time order:

```julia
# Insert a new event
push!(seq, Event(3, 1, 1.5))

# The sequence remains sorted by time
```

### Creating Empty Sequences

```julia
# Empty sequence for Float64 timestamps
seq = EventSequence{Float64}()

# Add events incrementally
push!(seq, Event(1, 2, 1.0))
push!(seq, Event(2, 1, 2.0))
```

## Loading Data

### From DataFrame

The most common way to load events:

```julia
using DataFrames

df = DataFrame(
    sender = [1, 2, 1],
    receiver = [2, 1, 3],
    time = [1.0, 2.0, 3.0]
)

seq = load_events(df)
```

### Custom Column Names

When your DataFrame has different column names:

```julia
df = DataFrame(
    from = [1, 2, 1],
    to = [2, 1, 3],
    timestamp = [1.0, 2.0, 3.0],
    type = [:email, :email, :meeting],
    importance = [1.0, 2.0, 1.5]
)

seq = load_events(df;
    sender_col = :from,
    receiver_col = :to,
    time_col = :timestamp,
    type_col = :type,
    weight_col = :importance
)
```

### String Actor Names

When actors are identified by names rather than numeric IDs:

```julia
df = DataFrame(
    sender = ["Alice", "Bob", "Alice", "Carol"],
    receiver = ["Bob", "Alice", "Carol", "Bob"],
    time = [1.0, 2.0, 3.0, 4.0]
)

seq = load_events(df; actor_names=true)

# Actors are assigned numeric IDs internally
# Access the mapping through the returned sequence
println(seq.n_actors)  # 3
```

### From CSV File

Load directly from a CSV file:

```julia
# Write a small demo file first
using CSV, DataFrames
CSV.write("events.csv", DataFrame(sender=[1, 2, 1], receiver=[2, 1, 3],
                                  time=[1.0, 2.0, 3.0]))

# Basic usage
seq = load_events("events.csv")
```

Column names and actor-name handling are configurable:

<!-- skip-check -->
```julia
seq = load_events("events.csv";
    sender_col = :source,
    receiver_col = :target,
    time_col = :timestamp,
    actor_names = true
)
```

### DateTime Parsing

For string timestamps that need parsing:

```julia
df = DataFrame(
    sender = [1, 2, 1],
    receiver = [2, 1, 3],
    time = ["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"]
)

seq = load_events(df; time_type=DateTime)
```

### From a DynamicNetwork (NetworkDynamic.jl)

When NetworkDynamic.jl is loaded alongside REM.jl, a package extension
provides `EventSequence(::DynamicNetwork)`: each edge activation spell
becomes one event whose time is the spell's onset. This lets temporal
network data flow straight into `generate_observations`/`fit_rem`:

```julia
using REM, NetworkDynamic   # loading both activates the extension

dnet = DynamicNetwork(10; observation_start=0.0, observation_end=100.0)
activate!(dnet, 1.0, 3.0; edge=(1, 2))
activate!(dnet, 2.0, 5.0; edge=(2, 3))

seq = EventSequence(dnet)                       # events at t = 1.0, 2.0
result = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=20)
```

Options:

- `eventtype=:onset`, `weight=1.0` — set on every generated event.
- `include_onset_censored=false` — onset-censored spells are skipped by
  default (their recorded onset is the observation-window start, not an
  observed event).

- `actors=nothing` — by default the actor universe is declared from the
  network's vertex set, so vertices with no edge spell stay in the risk set
  as isolates. Pass `actors` to override it.

For undirected networks the smaller vertex ID becomes the sender (edges
are stored with `(min, max)` ordering).

REM.jl keeps zero hard dependencies on the network stack: the method
lives in the `REMNetworkDynamicExt` extension and is compiled only when
NetworkDynamic.jl is present in the environment.

## Node Attributes

Node attributes store actor-level covariates for use with attribute statistics.

### Creating Attributes

```julia
# Categorical attribute with default value
gender = NodeAttribute(:gender,
    Dict(1 => "M", 2 => "F", 3 => "M"),  # Actor ID → value
    "Unknown"                             # Default for unspecified actors
)

# Numeric attribute
age = NodeAttribute(:age,
    Dict(1 => 25.0, 2 => 30.0, 3 => 28.0),
    0.0  # Default
)

# Boolean attribute
is_manager = NodeAttribute(:manager,
    Dict(1 => true, 2 => false, 3 => true),
    false
)
```

### Accessing Attribute Values

```julia
gender[1]  # "M"
gender[2]  # "F"
gender[4]  # "Unknown" (default - actor 4 not in dict)

age[1]     # 25.0
age[99]    # 0.0 (default)
```

### Modifying Attributes

```julia
# Set a value
age[4] = 35.0

# Update existing
age[1] = 26.0
```

### Using Attributes in Statistics

```julia
# Homophily: same gender
AttributeMatch(gender)

# Difference: age difference
NodeDifference(age)

# Main effects
SenderAttribute(age)
ReceiverAttribute(age)

# Specific combinations
ActorMix(gender, "M", "F")  # Male sender, female receiver
```

## Actor Sets

For specifying custom sets of actors:

```julia
# From numeric IDs
actors = ActorSet([1, 2, 3, 4, 5])

# From names (creates ID mapping)
actors = ActorSet(["Alice", "Bob", "Carol", "David"])

# Access mappings
actors.name_to_id["Alice"]  # 1
actors.id_to_name[1]        # "Alice"
actors.ids                   # [1, 2, 3, 4]

# Check membership
2 in actors   # true
10 in actors  # false
```

## Risk Sets

Risk sets define which dyads could potentially experience an event. This is used internally for case-control sampling.

```julia
rs = RiskSet(
    5,                        # Index of focal event
    [1, 2, 3],                # Potential senders
    [1, 2, 3, 4];             # Potential receivers
    exclude_self_loops = true # Exclude s == r (default: true)
)

# Number of dyads in risk set
REM.n_dyads(rs)  # 3*4 - 3 = 9 (excluding self-loops)
```

## Working with Different Time Scales

### Numeric Time

For abstract time units:

```julia
events = [
    Event(1, 2, 0.0),
    Event(2, 1, 1.0),
    Event(1, 2, 2.5),
]
seq = EventSequence(events)

# Decay with numeric halflife
decay = halflife_to_decay(10.0)  # Half weight after 10 time units
```

### DateTime

For real calendar time:

```julia
using Dates

events = [
    Event(1, 2, DateTime(2024, 1, 1, 9, 0)),   # 9:00 AM
    Event(2, 1, DateTime(2024, 1, 1, 10, 30)), # 10:30 AM
    Event(1, 3, DateTime(2024, 1, 1, 14, 0)),  # 2:00 PM
]
seq = EventSequence(events)

# Decay: halflife of 1 hour = 3600 seconds
decay = halflife_to_decay(3600.0)
state = EventNetworkState(seq; decay=decay)
```

### Date

For daily granularity:

```julia
using Dates

events = [
    Event(1, 2, Date(2024, 1, 1)),
    Event(2, 1, Date(2024, 1, 8)),   # One week later
    Event(1, 3, Date(2024, 1, 15)),  # Two weeks later
]
seq = EventSequence(events)

# Decay: halflife of 7 days = 7 * 86400 seconds
decay = halflife_to_decay(7.0 * 86400)
```

## Data Validation

### Self-Loops

Events where sender equals receiver generate a warning:

```julia
e = Event(1, 1, 1.0)  # Warning: Self-loop detected
```

To filter self-loops:

```julia
raw_events = [Event(1, 2, 1.0), Event(2, 2, 2.0), Event(2, 3, 3.0)]
events = [e for e in raw_events if e.sender != e.receiver]
seq = EventSequence(events)
```

### Missing Data

For DataFrames with missing values:

```julia
df = DataFrame(sender=[1, 2, missing], receiver=[2, 1, 3],
               time=[1.0, 2.0, 3.0])

# Filter rows with missing values before loading
df_clean = dropmissing(df, [:sender, :receiver, :time])
seq = load_events(df_clean)
```

### Duplicate Events

Events at the exact same time between the same actors are allowed but may affect some statistics:

```julia
events = [
    Event(1, 2, 1.0),
    Event(1, 2, 1.0),  # Duplicate - both are included
]
seq = EventSequence(events)
length(seq)  # 2
```

## Utility Functions

### Time Conversions

```julia
# Convert halflife to decay rate
decay = halflife_to_decay(10.0)  # λ such that weight = 0.5 at t = 10

# Convert back
halflife = decay_to_halflife(decay)

# Compute decay weight for an elapsed time of 5 units
weight = compute_decay_weight(decay, 5.0)
```

### Sequence Statistics

```julia
# Time span
first_time = seq[1].time
last_time = seq[end].time
duration = last_time - first_time

# Event counts by actor
using StatsBase
senders = [e.sender for e in seq]
sender_counts = countmap(senders)

# Unique dyads
dyads = Set((e.sender, e.receiver) for e in seq)
n_unique_dyads = length(dyads)
```
