# Statistics API Reference

This page documents all statistics available in REM.jl.

## Base Types and Interface

### Abstract Types

```@docs
AbstractStatistic
DyadStatistic
DegreeStatistic
TriangleStatistic
FourCycleStatistic
NodeStatistic
InteractionStatistic
```

### Interface Functions

```@docs
compute
name
```

### StatisticSet

```@docs
StatisticSet
compute_all
compute_all!
```

## Dyad Statistics

Statistics based on the history of events between the focal sender-receiver pair.

```@docs
Repetition
Reciprocity
InertiaStatistic
RecencyStatistic
DyadCovariate
```

## Degree Statistics

Statistics based on actor activity (out-degree) and popularity (in-degree).

```@docs
SenderActivity
ReceiverActivity
SenderPopularity
ReceiverPopularity
TotalDegree
DegreeDifference
LogDegree
```

## Triangle Statistics

Statistics capturing triadic closure effects in directed networks.

```@docs
TransitiveClosure
CyclicClosure
SharedSender
SharedReceiver
CommonNeighbors
GeometricWeightedTriads
```

## Four-Cycle Statistics

Statistics capturing higher-order clustering through four-node structures.

```@docs
FourCycle
GeometricWeightedFourCycles
```

## Node Attribute Statistics

Statistics incorporating actor-level attributes for homophily and covariate effects.

```@docs
AttributeMatch
NodeMix
NodeDifference
NodeSum
NodeProduct
SenderAttribute
ReceiverAttribute
SenderCategorical
ReceiverCategorical
```
