# REM.jl

[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/REM.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/REM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/REM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/REM.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="REM.jl icon" width="160">
</p>

A Julia implementation of **Relational Event Models** for statistical analysis of time-stamped relational events in networks.

## Overview

Relational Event Models (REM) are statistical models for analyzing sequences of time-stamped relational events. They uncover factors explaining why certain actors interact at higher rates than others, accounting for:

- **Dyadic effects**: Repetition, reciprocity, and inertia between actor pairs
- **Actor effects**: Activity and popularity patterns at the node level
- **Structural effects**: Triadic closure, four-cycles, and local clustering
- **Attribute effects**: Homophily and covariate-based selection

REM.jl is a port of [eventnet](https://github.com/juergenlerner/eventnet), providing efficient tools for modeling sequences of directed interactions between actors.

**Modeling assumptions.** REM.jl estimates the *ordinal* relational event model: only the order of events enters the likelihood (each event is one stratum of a conditional logistic regression against the other dyads at risk — all of them, or a sample of controls). Exact inter-event waiting times are used only for the optional memory models, not as a hazard term — the interval-timing likelihood of R `relevent::rem.dyad` is not implemented here: it is in [Revel.jl](https://statistical-network-analysis-with-julia.github.io/Revel.jl/dev/) (`fit_revel(...; model=:timing)`), together with the participation-shift effects (`PShift`), effect × covariate interactions and the goodness-of-fit diagnostics (see [Not implemented](#not-implemented)). The statistics follow **eventnet's definitions**: the triadic and four-cycle families default to eventnet's weighted form (the two dyad weights of every closing two-path combined by `aggregation=:min` and the two-paths added; `weighted=false` is the count of closing third parties), and history can be discounted either by eventnet's **halflife decay** (`decay=halflife_to_decay(h)`) or by a **sliding window** (`window=`), never both. **Tied timestamps are refused by default** (`ties=:error` names the tie and throws): choose `ties=:breslow` or `ties=:efron` — the classical Cox tie corrections, pinned against `survival::coxph` — or `ties=:ordered` to keep sequence order without a correction. With **sampled controls** only `:breslow` is offered: Efron's correction on a sampled risk set is biased toward zero, so that combination is refused with an `ArgumentError` (Efron is available with the full risk set). A sequence with **more than one event type** is refused until you say how the types enter (`eventtypes=:pool`, or `cases_of=` to model one type); `OfType(stat, type)` gives any statistic the history of one type. **Undirected** events are fitted against the unordered pairs with `directed=false`, and a **dyad-level risk set** is a list of `(sender, receiver)` pairs passed as `at_risk=`. Controls are sampled without replacement; when a small actor set makes fewer distinct dyads available than `n_controls`, the full risk set is used instead. Estimation runs on the ecosystem's shared `NetworkCore.newton_fit`; an unconverged fit warns and is flagged (`converged == false`, `NetworkCore.is_exact == false`).

## Installation

Requires Julia 1.12+. REM.jl depends on the unregistered
[NetworkCore.jl](https://github.com/statistical-network-analysis-with-Julia/NetworkCore.jl)
(the shared contracts, optimizer, presentation layer and the bundled
datasets), which must be added first:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/NetworkCore.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/REM.jl")
```

The examples below also load DataFrames.jl and CSV.jl (the "Loading Data"
example) and the standard library `Random`, and the documentation's guides
also use StatsBase.jl (`countmap`, in the event-data guide) and
Distributions.jl (the likelihood-ratio test, in the estimation guide). Add
the registered packages in the same environment:

```julia
Pkg.add(["DataFrames", "CSV", "StatsBase", "Distributions"])
```

[DynamicNetworks.jl](https://github.com/statistical-network-analysis-with-Julia/DynamicNetworks.jl)
is **not** a dependency: add it only if you want the
`EventSequence(::DynamicNetwork)` adapter, which lives in a package extension
compiled when both are loaded.

For development of the whole ecosystem, the website's
[workspace guide](https://statistical-network-analysis-with-julia.github.io/getting-started/)
prepares side-by-side clones with `tools/prepare_workspace.jl`.

## Statistics Implemented

### 1. Dyad Statistics

Statistics based on the history of events between specific actor pairs.

<!-- skip-check -->
```julia
Repetition(; directed=true)      # Past events from sender to receiver
Reciprocity()                    # Past events from receiver to sender
InertiaStatistic()               # Combined repetition and reciprocity
RecencyStatistic()               # 1/Δ since the last event on the dyad (:inverse_log, :exp_decay too)
DyadCovariate(attr)              # Dyad-level covariate value
```

### 2. Degree Statistics

Statistics measuring actor activity and popularity levels.

<!-- skip-check -->
```julia
SenderActivity()                 # Sender's out-degree (past sending activity)
ReceiverActivity()               # Receiver's out-degree (receiver's sending history)
SenderPopularity()               # Sender's in-degree (how often sender receives)
ReceiverPopularity()             # Receiver's in-degree (how often receiver receives)
TotalDegree(; role=:sender)      # Combined in-degree and out-degree
DegreeDifference()               # Difference between sender and receiver degrees
LogDegree(; role=:sender)        # Log-transformed degree statistic
```

### 3. Triangle Statistics

Statistics capturing triadic closure patterns in directed networks — **eventnet's
definition by default**: for a candidate `s→r`, the two (decayed, event-weighted)
dyad counts of every two-path through a third party `k` are combined by
`aggregation` (`:min` by default; `:max`, `:sum`, `:product`) and the parallel
two-paths are added. `weighted=false` gives the pre-0.2 count of distinct third
parties instead.

<!-- skip-check -->
```julia
TransitiveClosure()              # Σ_k min(w(s,k), w(k,r))   — s→k→r (transitive triads)
CyclicClosure()                  # Σ_k min(w(r,k), w(k,s))   — r→k→s (cyclic triads)
SharedSender()                   # Σ_k min(w(k,s), w(k,r))   — common sender
SharedReceiver()                 # Σ_k min(w(s,k), w(r,k))   — common receiver
CommonNeighbors()                # Σ_k min(u(s,k), u(k,r))   — undirected counts u
TransitiveClosure(aggregation=:sum)     # another combining function
TransitiveClosure(weighted=false)       # number of closing third parties (never decays)
GeometricWeightedTriads(alpha=0.5)      # e^α(1 − (1 − e^{−α})^n), n = number of third parties
```

### 4. Four-Cycle Statistics

Statistics measuring local clustering through four-cycle configurations (same
`weighted`/`aggregation` keywords, over the three weights of a three-path).

<!-- skip-check -->
```julia
FourCycle(; cycle_type=:out_out)          # :out_out, :in_in, :out_in, :in_out or :mixed
GeometricWeightedFourCycles(alpha=0.5)    # Geometrically weighted four-cycles
```

Participation shifts (`PSAB-BA`, …) are relevent's statistics, not eventnet's:
they are in [Revel.jl](https://statistical-network-analysis-with-julia.github.io/Revel.jl/dev/)
(`PShift`), and plug
straight into `fit_rem` through the shared `compute` generic.

### 5. Typed statistics and time-varying covariates

<!-- skip-check -->
```julia
OfType(Repetition(), :email)          # any state-reading statistic, on one event type's history
TimeVaryingCovariate((s, r, t) -> x)  # exogenous covariate f(sender, receiver, time)
```

### 6. Node Attribute Statistics

Statistics based on actor-level attributes for homophily and covariate effects.

<!-- skip-check -->
```julia
AttributeMatch(attr)                  # Binary: 1 if sender and receiver match
ActorMix(attr, sender_val, receiver_val)   # Specific attribute combinations
NodeDifference(attr; absolute=false)     # Numeric difference between actors
NodeSum(attr)                    # Sum of sender and receiver attributes
NodeProduct(attr)                # Product of sender and receiver attributes
SenderAttribute(attr)            # Effect of sender's attribute value
ReceiverAttribute(attr)          # Effect of receiver's attribute value
SenderCategorical(attr, val)     # Sender has specific categorical value
ReceiverCategorical(attr, val)   # Receiver has specific categorical value
```

## Usage

### Basic Example

The running example is the one R's `relevent` uses: the World Trade Center
police radio calls (Butts, Petrescu-Prahova & Cross 2007), bundled with
NetworkCore.jl as `load_dataset(:wtc_police_calls)` — 481 calls among 37
officers, in observed order (an ordinal clock, so there are no tied times).

```julia
using NetworkCore, REM

wtc = load_dataset(:wtc_police_calls)      # (events, n_actors, is_icr)
events = [Event(wtc.events[k, 2], wtc.events[k, 3], Float64(wtc.events[k, 1]))
          for k in 1:size(wtc.events, 1)]

# Declare the actor universe: all 37 officers, including the two who never
# call or are called. The likelihood is conditional on this risk set, so
# inferring it from the events (omitting `actors`) would change the estimand
seq = EventSequence(events; actors=ActorSet(1:37))

# A covariate: the three institutionalised coordinator roles (ICR)
icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:37))

stats = [
    Repetition(),            # calling the same officer again
    Reciprocity(),           # returning a call
    SenderActivity(),        # the caller's past activity (out-degree)
    ReceiverPopularity(),    # the callee's past popularity (in-degree)
    TransitiveClosure(),     # eventnet's weighted two-path closure
    SenderAttribute(icr),    # coordinators call more ...
    ReceiverAttribute(icr),  # ... and are called more
]

# 37 actors give only 1332 ordered dyads: enumerate the full risk set, which
# makes the partial likelihood the exact ordinal likelihood (relevent's)
result = fit_rem(seq, stats; n_controls=37 * 36 - 1)
println(result)
```

Output:

```text
Relational Event Model Results
==============================
Events: 481, Observations: 640692
Risk-set size: 1332 dyads, control sampling probability: 1.0
Log-likelihood: -2455.1321
Converged: true (11 iterations)
Std. errors: inverse Hessian (full risk set)

                     Estimate  Std.Error   z value  Pr(>|z|)
repetition            -0.3393     0.0329  -10.3052    <1e-16 ***
reciprocity            0.3256     0.0311   10.4592    <1e-16 ***
sender_activity        0.0340     0.0027   12.5001    <1e-16 ***
receiver_popularity    0.0288     0.0021   13.5086    <1e-16 ***
transitive_closure     0.1295     0.0185    7.0045   2.5e-12 ***
sender_icr             0.6119     0.1714    3.5704    0.0004 ***
receiver_icr           1.3122     0.1562    8.4024    <1e-16 ***
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The same call reproduces relevent's tutorial model (`wtcfit1`, the ICR
covariate as `CovInt`, i.e. `x_i + x_j`), coefficient 2.104 on both sides —
pinned by a golden fixture generated from R (see [Validation Against R](#validation-against-r)):

```julia
tutorial = fit_rem(seq, [NodeSum(icr)]; n_controls=37 * 36 - 1)
coef(tutorial)                   # [2.1045] — relevent's wtcfit1
NetworkCore.is_exact(tutorial)      # true: full risk set, no ties, converged
```

For a large actor set the full risk set is out of reach, and case-control
sampling is what makes REMs tractable: each event is compared with
`n_controls` dyads drawn at random from its risk set. The fit is then an
approximation to the full-risk-set likelihood (recorded in
`NetworkCore.approximations(fit)` and noted under the table). Its standard
errors are the observed information of the sampled likelihood — a
consistent variance estimator under nested case-control sampling, so nothing
is "understated" — with `se=:sandwich` as the misspecification-robust
alternative; what the sampling *does* affect is the point estimate of a
misspecified model, which depends on the control draw, and
`control_draw_cov` measures that:

```julia
using Random
sampled = fit_rem(seq, stats; n_controls=100, rng=Xoshiro(42))               # 100 of 1331 controls
robust  = fit_rem(seq, stats; n_controls=100, rng=Xoshiro(42), se=:sandwich)  # event-clustered
draws   = control_draw_cov(seq, stats; n_controls=100, n_draws=10,    # refit over 10 control draws
                           rng=Xoshiro(7))
draws.sd                          # draw-to-draw spread of each coefficient — a diagnostic, not an SE
```

How close a sampled fit comes to the full-risk-set one depends on
`n_controls` *and* on how well specified the model is — with a deliberately
crude model the two differ visibly on these data. The
[estimation guide](https://statistical-network-analysis-with-Julia.github.io/REM.jl/dev/guide/estimation/)
shows the comparison and how to choose.

### Result Structure

`fit_rem` returns a `REMResult`; `println(result)` prints the R-style block
above — events, observations, risk-set size and control sampling probability,
log-likelihood, convergence with the iteration count, the standard-error
estimator, the tie policy when one bit, the coefficient table, and under it
whatever needs saying: a note that the risk set was sampled (none with the
full risk set), or a warning that the fit did not converge, that the
information matrix is singular (collinear statistics — standard errors
`NaN`), or that a coefficient separated (`survival::coxph`'s "may be
infinite" rule), each naming the statistics concerned. The fields:

- `coefficients::Vector{Float64}`: Estimated coefficients (log hazard ratios)
- `std_errors::Vector{Float64}`: Standard errors of coefficients
- `z_values::Vector{Float64}`: Z-statistics for hypothesis testing
- `p_values::Vector{Float64}`: Two-sided p-values (`NetworkCore.z_pvalues`, floored at `floatmin`)
- `stat_names::Vector{String}`: Names of statistics in the model
- `n_events::Int`, `n_observations::Int`: Events (strata) and case-control rows
- `log_likelihood::Float64`: Log partial likelihood at convergence
- `converged::Bool`, `iterations::Int`: `NetworkCore.newton_fit`'s verdict (an unconverged fit warns)
- `var_cov::Matrix{Float64}`: Covariance matrix matching the `se` estimator used
- `se_type::Symbol`, `tie_type::Symbol`: what was actually done (`:hessian`/`:sandwich`; `:none` or the tie correction that bit)
- `singular::Bool`, `singular_suspects::Vector{String}`, `separated::Vector{String}`: identification failures and the statistics responsible; singular information also sets `converged=false`, while separation can still satisfy the objective stopping rule (`NetworkCore.is_exact` is `false` for either)
- `strata`, `risk_set_sizes`, `sampling_probs`: the risk-set bookkeeping per stratum

`REMResult` implements the full StatsAPI surface (each verb a method of the
`StatsAPI` generic, so it composes with StatsBase/GLM):

```julia
coef(result)          # Coefficient estimates
coefnames(result)     # Their names, the statistics' (R's names(coef(fit)))
stderror(result)      # Standard errors  (== sqrt.(diag(vcov(result))))
vcov(result)          # Covariance matrix
confint(result)       # Wald confidence intervals (level=0.95)
loglikelihood(result) # Log partial likelihood
nobs(result)          # Number of events (one stratum per event)
dof(result)           # Number of coefficients
aic(result); bic(result)
coeftable(result)     # NetworkCore.CoefficientTable — the table `show` prints; tbl["repetition"] is a row
```

`coeftable` returned a `DataFrame` before 0.2.0; to build one now, take the
fields: `tbl = coeftable(result); DataFrame(statistic=tbl.names, coefficient=tbl.estimates, std_error=tbl.std_errors, z_value=tbl.z_values, p_value=tbl.p_values)`.

What the fit actually did is machine-readable through the shared
result-metadata protocol: `NetworkCore.is_exact(result)`, `se_method`,
`tie_method`, `approximations` and `fit_metadata(result)`.

### Case-Control Sampling

REM.jl uses case-control sampling with stratified conditional logistic regression for efficient estimation:

```julia
# Configure sampler
sampler = CaseControlSampler(
    n_controls=100,           # Controls per case
    exclude_self_loops=true,  # Exclude i→i from risk set
)

# Generate observations (one row per case and per control); the control draw
# comes from `rng`, so a fixed rng reproduces it
obs = generate_observations(seq, stats, sampler; rng=Xoshiro(42))

# Fit model from observations
result = fit_rem(obs, [name(s) for s in stats])
```

### Temporal Decay and Sliding Windows

Support for exponential decay of network effects, where older events contribute less (eventnet's halflife model):

```julia
# Convert halflife to decay rate (here 50 events, on the ordinal clock)
decay = halflife_to_decay(50.0)

# Create network state with decay
state = EventNetworkState(seq; decay=decay)

# Fit model with decay
result = fit_rem(seq, stats; n_controls=100, decay=decay, rng=Xoshiro(42))

# Utility functions
decay_to_halflife(decay)              # Convert decay rate back to halflife
compute_decay_weight(5.0, decay)      # Weight of an event 5 time units old (elapsed time first)
```

Decayed counts are stored as `(value, last_time)` pairs and decayed
**lazily on read**: advancing the state clock is O(1), each event is
absorbed exactly once, and nothing is rescanned or rescaled per
evaluation — so statistic computation cost is independent of how long the
event history is.

The alternative memory model — eventnet does not offer it; it is the
`remstats`-style **sliding window** — makes an event older than
`current_time − window` stop counting altogether (in every count, degree
and adjacency set), amortized O(1) per event:

```julia
# Only the events of the last 50 time units count (seconds for Date/DateTime clocks)
result_w = fit_rem(seq, stats; n_controls=100, window=50.0, rng=Xoshiro(42))
state_w = EventNetworkState(seq; window=50.0)
```

`decay` and `window` are mutually exclusive (`ArgumentError`); `window=Inf`
is no window. Both are pinned against `survival::clogit`
(`test/fixtures/rem_eventnet.toml`).

### Node Attributes

Define and use actor-level attributes:

```julia
# Without a default an actor the attribute lacks is an error when a statistic
# reads it (no silent zero-fill); pass a third argument to fill deliberately
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "M", 3 => "F"), "Unknown")
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0, 3 => 28.0))

# Use in statistics
attribute_stats = [
    AttributeMatch(gender),       # Homophily: same gender
    NodeDifference(age),          # Age difference effect
    SenderAttribute(age),         # Sender's age effect
    ActorMix(gender, "M", "F"),   # M→F mixing pattern
]
```

### Loading Data

```julia
using DataFrames

# From DataFrame
df = DataFrame(
    sender = [1, 2, 1, 3],
    receiver = [2, 1, 3, 2],
    time = [1.0, 2.0, 3.0, 4.0]
)
small = load_events(df)

# From CSV file (write one for the demo)
using CSV
CSV.write("events.csv", df)
small = load_events("events.csv")

# With string actor names (automatically converted to integer IDs)
df_names = DataFrame(
    sender = ["Alice", "Bob", "Alice"],
    receiver = ["Bob", "Alice", "Carol"],
    time = [1.0, 2.0, 3.0]
)
named = load_events(df_names; actor_names=true)

# With event types and weights
df_typed = DataFrame(
    sender = [1, 2, 1],
    receiver = [2, 1, 3],
    time = [1.0, 2.0, 3.0],
    type = [:email, :email, :phone],
    weight = [1.0, 2.0, 1.5]
)
typed = load_events(df_typed; type_col=:type, weight_col=:weight)
```

`load_events` infers the actor universe from the event endpoints; declare it
before fitting (`EventSequence(collect(small); actors=ActorSet(1:4))`) or
`fit_rem` warns that the risk set was inferred. Event weights must be finite
and non-negative (they are added into counts).

### Event types

A sequence with more than one event type is refused until you say how the
types enter the model — an untyped statistic would silently count a `:phone`
call as the repetition of an `:email`:

```julia
mixed = EventSequence(collect(typed); actors=ActorSet(1:3))
type_stats = [OfType(Repetition(), :email),   # past e-mails on the dyad
              Reciprocity()]                  # past events of any type, the other way
compute_statistics(mixed, type_stats; eventtypes=:pool)
pooled = generate_observations(mixed, type_stats, CaseControlSampler(n_controls=5);
                               eventtypes=:pool)   # every event is a case
emails = generate_observations(mixed, type_stats, CaseControlSampler(n_controls=5);
                               cases_of=:email)    # only e-mails are cases
```

`OfType(stat, type)` wraps any state-reading statistic; `cases_of=` (on
`fit_rem`, `generate_observations`, `control_draw_cov`) models the events of
some types while every event still builds the history.

### From a DynamicNetwork (DynamicNetworks.jl extension)

```julia
using REM, DynamicNetworks   # loading both activates the REMDynamicNetworksExt extension

# Six vertices, 24 edge-activation spells (a reciprocated, repeated pattern)
dnet = DynamicNetwork(6; observation_start=0.0, observation_end=100.0)
spells = [(1, 2, 1.0), (2, 3, 2.0), (2, 1, 3.0), (3, 1, 4.0), (1, 2, 6.0), (4, 2, 7.0),
          (2, 4, 8.0), (1, 3, 9.0), (3, 2, 10.0), (2, 3, 11.0), (5, 1, 12.0), (1, 5, 13.0),
          (2, 1, 14.0), (1, 2, 15.0), (4, 5, 16.0), (5, 4, 17.0), (3, 1, 18.0), (1, 3, 19.0),
          (2, 4, 20.0), (6, 2, 21.0), (2, 6, 22.0), (1, 2, 23.0), (3, 2, 24.0), (2, 3, 25.0)]
for (s, r, onset) in spells
    activate!(dnet, onset, onset + 1.5; edge=(s, r))
end

# Each edge activation spell becomes one event at its onset time
dseq = EventSequence(dnet)        # EventSequence{Float64}(24 events, 6 actors, declared universe)
dresult = fit_rem(dseq, [Repetition(), Reciprocity()]; n_controls=29)   # full risk set
```

Onset-censored spells (flagged, open-left, or beginning before the observation
window) are skipped by default (pass `include_onset_censored=true` to keep
them). REM.jl depends on
NetworkCore.jl (the shared contracts, optimizer and presentation layer) but
not on DynamicNetworks.jl: the `EventSequence(::DynamicNetwork)` method
lives in a package extension compiled only when DynamicNetworks.jl is
present in the environment.

### Computing Statistics Without Fitting

```julia
# Compute statistics for all events (without case-control sampling):
# one row per event, read off the state as it stands BEFORE the event
stats_df = compute_statistics(seq, stats)

# Access EventNetworkState for custom computation
state = EventNetworkState(seq)
for event in seq
    # Compute statistics before updating state
    values = compute_all(stats, state, event.sender, event.receiver)

    # Update state with event
    update!(state, event)
end
state      # EventNetworkState{Float64}(37 actors, 481 events absorbed, no memory decay, current_time = 481.0)
```

### Undirected events

`directed=false` (on `generate_observations`, `fit_rem` and
`control_draw_cov`) rewrites each event as the pair `(min, max)` and puts the
`n(n−1)/2` unordered pairs of a symmetric actor set at risk — not the `n(n−1)`
ordered dyads, which would put every non-case pair in the risk set twice. Use
undirected statistics (`Repetition(directed=false)`, `CommonNeighbors()`,
`RecencyStatistic(directed=false)`, `NodeSum`, …) with it:

```julia
calls = EventSequence([Event(1, 2, 1.0), Event(3, 2, 2.0), Event(2, 1, 3.0),
                       Event(1, 3, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)];
                      actors=ActorSet(1:4))
und = fit_rem(calls, [Repetition(directed=false)]; n_controls=5, directed=false)
risk_set_sizes(und)[1]                # 6 = 4·3/2 unordered pairs
```

## Utility Functions

<!-- skip-check -->
```julia
# Time decay utilities
halflife_to_decay(halflife)          # Convert halflife to decay parameter
decay_to_halflife(decay)             # Convert decay to halflife
compute_decay_weight(elapsed, decay) # exp(−decay · elapsed)

# EventNetworkState accessors
get_dyad_count(state, s, r)          # Events from s to r
get_undirected_count(state, i, j)    # Events between i and j (either direction)
get_out_degree(state, actor)         # Actor's out-degree
get_in_degree(state, actor)          # Actor's in-degree
get_out_neighbors(state, actor)      # Actors receiving from actor (read-only AbstractSet{Int})
get_in_neighbors(state, actor)       # Actors sending to actor (read-only AbstractSet{Int})
has_edge(state, s, r)                # Whether s→r exists

# Risk set utilities
n_dyads(risk_set)                    # Number of dyads in risk set
```

## R and other software

Where each relevent / rem / eventnet / goldfish / remstats feature lives in
this ecosystem. Revel.jl's `effect_catalogue()` gives the effect-by-effect
concordance (including each package's measurement choices).

| Feature | R / other package | Julia |
|---|---|---|
| Ordinal (order-only) REM, stratified conditional logit | `relevent::rem.dyad(ordinal=TRUE)`, eventnet, `survival::clogit` | REM.jl `fit_rem` (also Revel.jl `fit_revel`) |
| Case-control sampling of the risk set | eventnet, Lerner & Lomi (2020) | REM.jl `n_controls`; Revel.jl `n_controls` |
| Interval (waiting-time) likelihood, exponential baseline | `relevent::rem.dyad(ordinal=FALSE)` | Revel.jl `fit_revel(...; model=:timing)` |
| Endogenous statistics with rem's normalised half-life decay | rem (Brandenberger) | Revel.jl (`effect_catalogue()` lists rem's definitions) |
| Participation shifts (`PSAB-BA`, …), relevent's normalised degree, `FrPSndSnd`/`FrRecSnd`, recency ranks | relevent | Revel.jl (`PShift`, `effect_catalogue()`) |
| eventnet's weighted triadic / four-cycle family, halflife decay | eventnet | REM.jl |
| Sliding-window memory, typed statistics (`consider_type`), `scaling = "prop"`/`"std"` | remstats | REM.jl `window=`, `OfType`; Revel.jl layers, `Standardized` |
| Type-filtered attributes, dyad-level risk sets | eventnet | REM.jl `OfType`, `at_risk=` a list of dyads |
| Receiver-choice step of DyNAM | goldfish `subModel = "choice"` | Revel.jl `fit_receiver_choice` |
| Time-varying covariates | goldfish, remstats | REM.jl `TimeVaryingCovariate(f)`; Revel.jl (change tables) |
| Effect × covariate interactions, goodness of fit, hyperevents | goldfish, remstats, eventnet | Revel.jl |
| DyNAM sender-rate step, DyNAM-i | goldfish | not implemented |

## Not implemented

Each is refused with an `ArgumentError` where there is an entry point for it
(kept in sync with the CHANGELOG's "Known limitations"):

- **The interval (waiting-time) likelihood** and a baseline event rate:
  REM.jl's likelihood is ordinal. Use Revel.jl, `fit_revel(...; model=:timing)`.
- **Efron's tie correction with sampled controls** (`ties=:efron` together with
  `n_controls` below the risk set): refused, because the design is biased
  toward zero; use `ties=:breslow` with sampled controls, or Efron on the full
  risk set.
- **`ties=:batch`**: in an ordinal likelihood a simultaneous batch is the
  Breslow correction; refused with a pointer to `:breslow`.
- **A dyad × event-type risk set** (each dyad at risk once per type, with
  type-specific baseline rates; remstats `consider_type = "interact"`): the
  risk set is over dyads. Event types enter through `OfType` statistics and
  `cases_of=` (a likelihood conditional on the event's type).
- **Covariate tables**: a time-varying covariate is a function
  (`TimeVaryingCovariate(f)`), not a spell table; `NodeAttribute` and
  `DyadCovariate` are static. Revel.jl reads change tables.
- **relevent's normalised degree, `FrPSnd`/`FrRecSnd`, recency ranks and
  participation shifts; remstats' `scaling`**: REM.jl's degree statistics are
  eventnet's event counts. All are in Revel.jl.
- **The DyNAM rate model** (the sender-rate step of actor-oriented models) and
  DyNAM-i: not in this ecosystem.
- **Mixed directed/undirected statistics under `directed=false`**: the pair is
  stored as `(min, max)`, so a directed statistic would read it in ID order;
  use the undirected forms (not refused — documented here).
- **`se=:bootstrap`**: the draw-to-draw spread of a sampled fit is a
  diagnostic (`control_draw_cov`), not a standard error; refused with that
  pointer.
- **A `missing=` policy**: an event stream has no unobserved dyads to mask;
  declare the risk set instead.

## Validation Against R

Four golden fixtures (`test/fixtures/*.toml`, each regenerated by a checked-in
`test/fixtures/r/*.R` script and loaded with `NetworkCore.load_golden`, which
refuses a fixture without provenance) pin REM.jl against R:

- `rem_clogit.toml` — `survival::clogit`: the count statistics and the estimator (agreement < 1e-13);
- `rem_ties.toml` — `survival::coxph(ties="breslow"/"efron")`: the tie corrections (< 1e-11);
- `rem_eventnet.toml` — `survival::clogit`: eventnet's weighted triadic family under each aggregation, undirected repetition, halflife-decayed and windowed counts, every column rebuilt from the raw edgelist in plain R (tolerance 1e-8);
- `rem_relevent_wtc.toml` — `relevent::rem.dyad` (1.2.1) on the **bundled** WTC police calls (`NetworkCore.load_dataset(:wtc_police_calls)`; R reads the same TSV files): with the full risk set enumerated (`n_controls = 37·36 − 1`) REM's conditional logit *is* rem.dyad's ordinal likelihood, and the tutorial's ICR effect comes out at 2.104 on both sides (tolerance 1e-6, optimizer termination slack).

Every statistic also has a hand-computed testset with the expected numbers
written out, and every exported docstring carries a runnable example that the
test suite executes.

## Running Tests

<!-- skip-check -->
```julia
include("test/runtests.jl")
```

The allocation-regression gates and the benchmark suite live in `benchmark/`
(own environment; sources NetworkCore.jl from the sibling checkout):

<!-- skip-check -->
```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/regression_tests.jl   # 0-byte kernels, O(events) streaming
julia --project=benchmark benchmark/benchmarks.jl         # timings + SCALING assertions
```

Time to first fit is kept low by a PrecompileTools workload (a 20-event toy
fit runs at precompile time): `using REM` takes under a second and the first
`fit_rem` about half a second more.

## Documentation

For more detailed documentation, see:

- [Documentation](https://statistical-network-analysis-with-Julia.github.io/REM.jl/dev/)

## References

1. Butts, C.T. (2008). A relational event framework for social action. *Sociological Methodology*, 38(1), 155-200.

2. Lerner, J., Lomi, A. (2020). Reliability of relational event model estimates under sampling: How to fit a relational event model to 360 million dyadic events. *Network Science*, 8(1), 97-135.

3. Lerner, J., Bussmann, M., Snijders, T.A.B., Brandes, U. (2013). Modeling frequency and type of interaction in event networks. *Corvinus Journal of Sociology and Social Policy*, 4(1), 3-32.

4. Brandes, U., Lerner, J., Snijders, T.A.B. (2009). Networks evolving step by step: Statistical analysis of dyadic event data. *2009 International Conference on Advances in Social Network Analysis and Mining*, 200-205.

5. Perry, P.O., Wolfe, P.J. (2013). Point process modelling for directed interaction networks. *Journal of the Royal Statistical Society: Series B*, 75(5), 821-849.

6. Butts, C.T., Petrescu-Prahova, M., Cross, B.R. (2007). Responder communication networks in the World Trade Center disaster: Implications for modeling of communication within emergency settings. *Journal of Mathematical Sociology*, 31(2), 121-147.

## Citation

If you use REM.jl in your work, please cite it using the entry in
[`CITATION.bib`](CITATION.bib), and please also cite the methods it
implements: the relational event framework (Butts 2008, reference 1) and the
case-control sampling of eventnet (Lerner and Lomi 2020, reference 2). The
per-package list is on the ecosystem's
[How to cite](https://statistical-network-analysis-with-julia.github.io/citing/)
page.

```biblatex
@misc{SNWJREMJL,
  author = {Santoni, Simone},
  title = {REM.jl: Relational Event Models for Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/REM.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/REM.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
