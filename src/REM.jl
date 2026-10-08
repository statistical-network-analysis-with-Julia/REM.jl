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
using PrecompileTools
using Printf
using Random
using Statistics
using StatsAPI
using StatsBase

# The StatsAPI surface: every verb is a METHOD on the
# StatsAPI generic (the binding StatsBase, GLM and NetworkCore all share), never a
# REM-local function of the same name. `NetworkCore.check_statsapi(fit; strict=true)`
# pins the full surface, `coefnames` included, in the testset.
import StatsAPI: coef, stderror, vcov, confint, loglikelihood, nobs, dof, aic, bic,
                 coeftable, coefnames

# Shared result-presentation infrastructure (NetworkCore.jl): the R-style
# coefficient table used by every model package in the ecosystem, its
# inspectable form `CoefficientTable` (what `coeftable(fit)` returns), the ONE
# z → two-sided-p helper (floored, NaN-aware) and the ONE `se=`
# validator
using NetworkCore: print_coeftable, CoefficientTable, z_pvalues, check_se
# The module itself, for the qualified separation verdict and policy helpers
# (`NetworkCore.clogit_separation`, `warn_separation`, `separation_caveat`),
# which are `public`, not exported
import NetworkCore

# The ONE Newton–Raphson optimizer of the ecosystem (NetworkCore.jl `src/newton.jl`,
# `public`, not exported): step halving, the combined
# |Δll|/gradient stopping rule and the Cholesky-based covariance live there.
# REM supplies only its per-stratum softmax kernel as the `(ll, grad, hess)`
# closure — it hosts no Newton loop of its own.
import NetworkCore: newton_fit

# The ONE shared resampling loop (NetworkCore.jl `src/bootstrap.jl`): resample,
# refit, empirical covariance. `control_draw_cov` supplies the two callbacks —
# here what is resampled is the case-control risk set, not the model, and the
# result is a draw-sensitivity diagnostic, not a standard error — and does not
# reimplement the loop, the threading or the rng discipline. It lives in
# NetworkCore.jl (not ERGM.jl, where `newton_fit` lives) precisely because REM.jl
# does not depend on ERGM.jl.
using NetworkCore: bootstrap_cov

# The shared statistic protocol (NetworkCore.jl `src/statistics.jl`): `compute`,
# `name` and `compute_all` are ONE set of generics that every model package
# extends for its own statistic types. REM's methods take relational-event
# statistics (`compute(stat, state, sender, receiver)`), ERGM's take terms
# (`compute(term, net)`) — different signatures, same function, so
# `using ERGM, REM` (cross-sections + dynamics) leaves the verbs usable
# unqualified instead of undefined by Julia's conflicting-export rule.
# Imported by name because we add methods to them.
import NetworkCore: compute, name, compute_all

# Same principle for `has_edge`: NetworkCore.jl re-exports the `Graphs.has_edge`
# generic, and "is there a tie from i to j" is the same question of an
# accumulated event network as of a graph. REM adds a method for its own
# `EventNetworkState` rather than defining a rival function, which is what makes
# the name safe to export again.
import NetworkCore: has_edge

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): the seven
# generic accessors that say what a fit actually did — which estimand, which
# objective, whether it is exact FOR THIS FIT, how the standard errors were
# obtained, and how tied event times were treated. Imported by name because REM
# adds methods for `REMResult`; `fit_metadata(fit)` collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# The shared TIED-EVENT vocabulary (NetworkCore.jl `src/results.jl`): one `ties=`
# keyword, one set of symbols, one meaning per symbol, across REM.jl and
# Revel.jl. `check_tie_policy` refuses a policy this model cannot honour
# (`:batch`) instead of letting it silently no-op.
using NetworkCore: TIE_POLICIES, check_tie_policy

# Core types
export Event, EventSequence, RiskSet, n_dyads
export ActorSet, NodeAttribute, has_default

# Data loading
export load_events, load_events!

# Statistics types and computation
export AbstractStatistic, compute, name, StatisticSet, compute_all, compute_all!
# Trait: does a statistic read the state's event log? (REM's own never do;
# foreign statistics default to `true` so nothing is silently dropped)
export needs_history
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
# `ActorMix` is not called `NodeMix` because ERGM.jl exports a distinct
# `NodeMix` term and the two would collide.
export AttributeMatch, ActorMix, NodeDifference, NodeSum, NodeProduct
export SenderAttribute, ReceiverAttribute
export SenderCategorical, ReceiverCategorical
export OfType, TimeVaryingCovariate

# Network state
export EventNetworkState, update!, reset!
export get_dyad_count, get_undirected_count, get_out_degree, get_in_degree
# `has_edge` is a METHOD of the shared Graphs/NetworkCore generic (see above), not
# a rival function, so exporting it cannot collide
export has_edge
export get_out_neighbors, get_in_neighbors

# Observation and estimation
export CaseControlSampler
export compute_statistics, generate_observations
export fit_rem, REMResult
# The control-draw sensitivity diagnostic of a sampled fit (NOT a standard
# error: see its docstring for why there is no `se=:bootstrap`)
export control_draw_cov
# The StatsAPI verbs REM implements (all ten): the StatsAPI bindings themselves
export coef, stderror, vcov, confint, loglikelihood, nobs, dof, aic, bic, coeftable,
       coefnames
# There is deliberately NO `rem` alias of `fit_rem` (the statnet-style verb the
# other model packages carry): `rem` is `Base.rem`, the remainder function, and
# exporting a rival binding would break `rem(7, 2)` in every `using REM` session
# (an ambiguity error, not a shadowing). relevent's `rem.dyad` likelihoods are
# Revel.jl's `fit_revel`. The "No export collides with Base" testset pins it.
# The control inclusion probabilities (and risk-set sizes) the fit conditioned
# on: part of the estimand, not an implementation detail (see `fit_rem`)
export sampling_probs, risk_set_sizes

# Utility functions
export halflife_to_decay, decay_to_halflife, compute_decay_weight

# Not exported (the name is generic), but part of the API: the guard that
# refuses the Efron tie correction on a sampled risk set, which Revel.jl's
# design calls so that both packages refuse the same combination
public check_tie_sampling

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
include("statistics/typed.jl")
include("observation.jl")
include("estimation.jl")

# `EventSequence(::DynamicNetworks.DynamicNetwork)` is provided by the
# REMDynamicNetworksExt package extension, loaded automatically when
# DynamicNetworks.jl is present in the environment (see ext/).

# ---------------------------------------------------------------------------
# Precompile workload
# ---------------------------------------------------------------------------
# Time-to-first-fit was 3.7 s of pure compilation on top of a 0.8 s `using REM`
# (measured 2026-09-09; CHANGELOG "Performance"). The workload below runs the
# whole pipeline once at precompile time — sequence construction, the
# case-control design (`generate_observations`), `fit_rem` with both
# single-draw standard-error estimators, the Breslow (sampled) and Efron (full
# risk set) tie corrections on a stream that carries one tie, `compute_statistics`, every StatsAPI verb and
# `show` — so that the native code for those call paths is cached in the
# package image instead of compiled in every session. The toy data is 6 actors
# and 20 events; nothing about it (seed, statistics, controls) leaks into any
# user-visible state: the draw is a local `Xoshiro`, never the global RNG.
@setup_workload begin
    _pc_rng = Random.Xoshiro(20260909)
    _pc_events = Event{Float64}[]
    for k in 1:20
        s = rand(_pc_rng, 1:6)
        r = rand(_pc_rng, 1:5)
        r >= s && (r += 1)
        # events 10 and 11 share a timestamp: the one tie the Efron path needs
        push!(_pc_events, Event(s, r, k == 11 ? 10.0 : Float64(k)))
    end
    @compile_workload begin
        seq = EventSequence(_pc_events; actors=ActorSet(1:6))
        stats = [Repetition(), Reciprocity(), SenderActivity()]
        sampler = CaseControlSampler(n_controls=5)
        # sampled controls with Breslow (the tie correction valid under
        # sampling), and Efron on the full risk set (29 = 6·5 − 1)
        obs = generate_observations(seq, stats, sampler; ties=:breslow,
                                    rng=Random.Xoshiro(1))
        fit = fit_rem(seq, stats; n_controls=5, ties=:breslow, rng=Random.Xoshiro(1))
        fit_rem(seq, stats; n_controls=29, ties=:efron)
        fit_rem(seq, stats; n_controls=5, ties=:breslow, se=:sandwich,
                rng=Random.Xoshiro(1))
        fit_rem(obs, [name(s) for s in stats])
        compute_statistics(seq, stats; ties=:breslow)
        coef(fit); stderror(fit); vcov(fit); confint(fit); loglikelihood(fit)
        nobs(fit); dof(fit); aic(fit); bic(fit); coeftable(fit)
        sprint(show, fit)
        sprint(show, seq)
        sprint(show, StatisticSet(stats))
    end
end

end # module
