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

# Shared result-presentation infrastructure (Networks.jl): the R-style
# coefficient table used by every model package in the ecosystem
using Networks: print_coeftable

# The ONE shared resampling loop (Networks.jl `src/bootstrap.jl`): resample,
# refit, empirical covariance. `fit_rem(...; se=:bootstrap)` supplies the two
# callbacks — here what is resampled is the case-control risk set, not the model
# — and does not reimplement the loop, the threading or the rng discipline. It
# lives in Networks.jl (not ERGM.jl, where `newton_fit` lives) precisely because
# REM.jl does not depend on ERGM.jl.
using Networks: bootstrap_cov

# The shared statistic protocol (Networks.jl `src/statistics.jl`): `compute`,
# `name` and `compute_all` are ONE set of generics that every model package
# extends for its own statistic types. REM's methods take relational-event
# statistics (`compute(stat, state, sender, receiver)`), ERGM's take terms
# (`compute(term, net)`) — different signatures, same function, so
# `using ERGM, REM` (cross-sections + dynamics) leaves the verbs usable
# unqualified instead of undefined by Julia's conflicting-export rule.
# Imported by name because we add methods to them.
import Networks: compute, name, compute_all

# Same principle for `has_edge`: Networks.jl re-exports the `Graphs.has_edge`
# generic, and "is there a tie from i to j" is the same question of an
# accumulated event network as of a graph. REM adds a method for its own
# `EventNetworkState` rather than defining a rival function, which is what makes
# the name safe to export again.
import Networks: has_edge

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the seven
# generic accessors that say what a fit actually did — which estimand, which
# objective, whether it is exact FOR THIS FIT, how the standard errors were
# obtained, and how tied event times were treated. Imported by name because REM
# adds methods for `REMResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# The shared TIED-EVENT vocabulary (Networks.jl `src/results.jl`): one `ties=`
# keyword, one set of symbols, one meaning per symbol, across REM.jl and
# Relevent.jl. `check_tie_policy` refuses a policy this model cannot honour
# (`:batch`) instead of letting it silently no-op.
using Networks: TIE_POLICIES, check_tie_policy

# Core types
export Event, EventSequence, RiskSet, n_dyads
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
# `ActorMix` was called `NodeMix` before v0.2; it was renamed because ERGM.jl
# exports a distinct `NodeMix` term and the two collided. `REM.NodeMix` remains
# as a deprecated, non-exported alias (see `statistics/node.jl`).
export AttributeMatch, ActorMix, NodeDifference, NodeSum, NodeProduct
export SenderAttribute, ReceiverAttribute
export SenderCategorical, ReceiverCategorical

# Network state
export EventNetworkState, update!, reset!
export get_dyad_count, get_undirected_count, get_out_degree, get_in_degree
# `has_edge` is a METHOD of the shared Graphs/Networks generic (see above), not
# a rival function, so exporting it cannot collide
export has_edge
export get_out_neighbors, get_in_neighbors

# Observation and estimation
export Observation, CaseControlSampler
export compute_statistics, generate_observations
export fit_rem, REMResult
export coef, stderror, coeftable
# The control inclusion probabilities (and risk-set sizes) the fit conditioned
# on: part of the estimand, not an implementation detail (see `fit_rem`)
export sampling_probs, risk_set_sizes

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
