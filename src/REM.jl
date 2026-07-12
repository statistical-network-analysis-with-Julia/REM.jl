"""
    REM.jl - Relational Event Models for Julia

A Julia implementation for statistical analysis of relational event networks.
Port of eventnet (https://github.com/juergenlerner/eventnet).

Relational Event Models (REM) are statistical models for analyzing sequences of
time-stamped relational events to uncover factors explaining why some actors
interact at higher rates than others.
"""
module REM

using CSV
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics
using StatsAPI
using StatsBase

# StatsAPI generics extended for REMResult (shared with StatsBase, GLM, ...)
import StatsAPI: coef, stderror, coeftable

# Shared result-presentation infrastructure (Network.jl): the R-style
# coefficient table used by every model package in the ecosystem
using Network: print_coeftable

# Core types
export Event, EventSequence, RiskSet
export ActorSet, NodeAttribute

# Data loading
export load_events, load_events!

# Statistics types and computation
export AbstractStatistic, compute, name, StatisticSet, compute_all, compute_all!
export DyadStatistic, DegreeStatistic, TriangleStatistic, FourCycleStatistic
export NodeStatistic, InteractionStatistic

# Specific statistics - Dyad
export Repetition, Reciprocity, InertiaStatistic, RecencyStatistic, DyadCovariate

# Specific statistics - Degree
export SenderActivity, ReceiverActivity, SenderPopularity, ReceiverPopularity
export TotalDegree, DegreeDifference, LogDegree

# Specific statistics - Triangle
export TransitiveClosure, CyclicClosure, SharedSender, SharedReceiver
export CommonNeighbors, GeometricWeightedTriads

# Specific statistics - Four-cycle
export FourCycle, GeometricWeightedFourCycles

# Specific statistics - Node attributes
export AttributeMatch, NodeMix, NodeDifference, NodeSum, NodeProduct
export SenderAttribute, ReceiverAttribute
export SenderCategorical, ReceiverCategorical

# Network state
export EventNetworkState, update!, reset!
export get_dyad_count, get_undirected_count, get_out_degree, get_in_degree
# `has_edge` is intentionally not exported (it would collide with
# Graphs.has_edge); call it as `REM.has_edge`
export get_out_neighbors, get_in_neighbors

# Observation and estimation
export Observation, CaseControlSampler
export compute_statistics, generate_observations
export fit_rem, REMResult
export coef, stderror, coeftable

# Utility functions
export halflife_to_decay, decay_to_halflife, compute_decay_weight

# Include source files
include("types.jl")
include("events.jl")
include("network.jl")
include("statistics/base.jl")
include("statistics/dyad.jl")
include("statistics/degree.jl")
include("statistics/triangle.jl")
include("statistics/fourcycle.jl")
include("statistics/node.jl")
include("observation.jl")
include("estimation.jl")

# `EventSequence(::NetworkDynamic.DynamicNetwork)` is provided by the
# REMNetworkDynamicExt package extension, loaded automatically when
# NetworkDynamic.jl is present in the environment (see ext/).

end # module
