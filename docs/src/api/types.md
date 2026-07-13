# Types API Reference

This page documents the core data types in REM.jl.

## Events

### Event

```@docs
Event
```

### EventSequence

```@docs
EventSequence
```

## Actor Data

### ActorSet

```@docs
ActorSet
```

### NodeAttribute

```@docs
NodeAttribute
```

### RiskSet

```@docs
RiskSet
```

## Network State

The `EventNetworkState` type maintains the cumulative state of the network as events are processed. It tracks dyad counts, degrees, and neighbor sets, optionally with exponential decay.

### EventNetworkState

```@docs
EventNetworkState
```

### State Updates

```@docs
update!
reset!
```

## State Query Functions

These functions query the current network state for various quantities.

### Dyad Queries

```@docs
get_dyad_count
get_undirected_count
has_edge
```

### Degree Queries

```@docs
get_out_degree
get_in_degree
```

### Neighbor Queries

```@docs
get_out_neighbors
get_in_neighbors
```

## Time Utilities

The decay helpers are documented with the other utilities on the
[Estimation page](estimation.md#Time-Decay).
