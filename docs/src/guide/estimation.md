# Model Estimation

REM.jl estimates relational event models by stratified conditional logistic regression — the Cox partial likelihood with one stratum per event, the case dyad against the other dyads at risk. When the risk set is small the strata hold every dyad and the likelihood is exact; when it is not, case-control sampling makes estimation tractable.

## Overview

The estimation process follows three steps:

1. **Risk set**: For each observed event, the dyads that could have occurred instead — all of them, or a sample of controls
2. **Statistic Computation**: Calculate statistics for cases and controls, off the network state as it stands *before* each event
3. **Maximum Partial Likelihood**: Fit the stratified conditional logit on the shared `Networks.newton_fit`

The running example is the bundled WTC police-calls stream (see [Getting Started](../getting_started.md)):

```julia
using Networks, REM
using Random

wtc = load_dataset(:wtc_police_calls)
events = [Event(wtc.events[k, 2], wtc.events[k, 3], Float64(wtc.events[k, 1]))
          for k in 1:size(wtc.events, 1)]
seq = EventSequence(events; actors=ActorSet(1:37))
icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:37))
stats = [Repetition(), Reciprocity(), SenderActivity(), ReceiverPopularity(),
         TransitiveClosure(), SenderAttribute(icr), ReceiverAttribute(icr)]
full = 37 * 36 - 1                  # n_controls that enumerates all 1332 dyads
```

## Why Case-Control Sampling?

For a network with $n$ actors, there are $n(n-1)$ possible directed dyads at each time point. Computing statistics for all dyads at all time points is often computationally infeasible.

Case-control sampling solves this by:

- Treating observed events as "cases"
- Sampling a subset of non-events as "controls"
- Using stratified estimation to obtain consistent parameter estimates

This approach is statistically valid and dramatically reduces computation time. It is also unnecessary when $n(n-1)$ is small: 37 actors are 1332 dyads, and enumerating them all costs well under a second here.

## Configuring the Sampler

```julia
sampler = CaseControlSampler(
    n_controls = 100,         # Controls per case
    exclude_self_loops = true, # Exclude s→s from risk set
    seed = 42                  # Random seed for reproducibility
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_controls` | Number of controls sampled per case | Required |
| `exclude_self_loops` | Whether to exclude self-events | `true` |
| `seed` | Random seed (nothing = draw from the `rng` keyword) | `nothing` |

### [Choosing Number of Controls](@id choosing-n-controls)

Controls are sampled **without replacement**; if fewer distinct dyads exist than `n_controls`, the sampler enumerates the full risk set (all non-case dyads) and warns once. Passing `n_controls = n(n−1) − 1` therefore enumerates it on purpose, silently.

What sampling costs is not only variance. For a correctly specified model the sampled estimates centre on the truth for any `n_controls` (nested case-control sampling is consistent, and the standard errors of the sampled fit are calibrated — see [Standard errors](@ref "Standard errors: two estimators, and a diagnostic that is not one")); for a model that is *not* the true one — every real model — the sampled fit converges to a *pseudo-true value that depends on the control draw*, and both the draw-to-draw spread and the distance from the full-risk-set fit shrink as `n_controls` grows. One seed shows one draw; `control_draw_cov` refits over independent draws and reports their mean and spread. On the WTC calls, ten draws each:

```julia
exact = fit_rem(seq, stats; n_controls=full)
println(rpad("n_controls", 12), rpad("repetition (mean ± sd over 10 draws)", 40), "per-draw SE")
for n_controls in (20, 100, 400)
    d = control_draw_cov(seq, stats; n_controls=n_controls, n_draws=10, rng=Xoshiro(1))
    one = fit_rem(seq, stats; n_controls=n_controls, seed=1)
    println(rpad(n_controls, 12),
            rpad(string(round(d.mean[1], digits=2), " ± ", round(d.sd[1], digits=2)), 40),
            round(stderror(one)[1], digits=2))
end
println(rpad("1331 (full)", 12), rpad(round(coef(exact)[1], digits=2), 40), round(stderror(exact)[1], digits=2))
```

| n_controls | repetition (mean ± sd over 10 draws) | per-draw SE | reciprocity (mean ± sd) | exact |
|---|---|---|---|---|
| 20 | −0.50 ± 0.07 | 0.08 | 0.59 ± 0.08 | false |
| 100 | −0.41 ± 0.04 | 0.05 | 0.43 ± 0.04 | false |
| 400 | −0.33 ± 0.02 | 0.04 | 0.32 ± 0.02 | false |
| 1331 (full) | −0.34 | 0.03 | 0.33 | true |

Two things are going on, and they are different. The **mean over draws moves** toward the full-risk-set value as `n_controls` grows (−0.50 → −0.41 → −0.33 → −0.34): that is the pseudo-true shift of a misspecified model, a bias that only more controls cure. And **individual draws scatter** around that mean — by about a standard error at 20 controls (seed 42 alone gives −0.72, nearly two draw-sds out), by half of one at 400. Neither is a deficiency of the standard errors, which are calibrated for the sampled likelihood they belong to; they are the practical reason to use the full risk set when you can, and to check a sampled fit against a larger `n_controls` (or `control_draw_cov`) when you cannot.

So: **enumerate the full risk set whenever you can afford it** (tens of thousands of dyads is fine). When you cannot, use as many controls as you can afford, check the estimates against a larger `n_controls`, and read the sampling probability the fit reports (`Networks.approximations(fit)` records that the likelihood was sampled). The rough guidance:

| n_controls | Use Case |
|------------|----------|
| full risk set | Whenever `n(n−1)` is affordable — the exact ordinal likelihood |
| 100-200 | Exploratory analysis on large actor sets |
| 400+ | Final results on large actor sets; verify stability across `n_controls` |

!!! note "Ordinal likelihood"
    REM.jl fits the *ordinal* REM: only event order enters the likelihood; the exact
    inter-event waiting times are not part of the hazard (unlike `relevent::rem.dyad`'s
    interval likelihood, which lives in Relevent.jl). The default standard errors are
    the inverse observed information on the risk set that was used — sampled or full;
    `se=:sandwich` is the misspecification-robust alternative (below).

## Tied event times

The likelihood is a likelihood over the **order** of the events, and the statistics of
an event are read off the network state as it stands *before* it. So two events sharing
a timestamp are not a sorting nuisance: whichever is placed first enters the *statistics*
of the one placed second (its `Repetition`, its `Reciprocity`, its degrees). Sorting a tie
invents the very thing the model is about.

`fit_rem` therefore **refuses tied data by default** and the policy is explicit
(`Networks.TIE_POLICIES`, the same vocabulary as `Relevent.fit_obpm`/`fit_timing`):

| `ties=` | what it does |
|---|---|
| `:error` (default) | names the tie and throws |
| `:ordered` | sequence order, no correction (the pre-0.2 behaviour) |
| `:breslow` | Breslow correction: the state is frozen across the tie block, each tied event is a stratum, all share one denominator |
| `:efron` | Efron correction: as Breslow, plus the `1 − (j−1)/d` denominator weights on the tied cases — the better approximation, and R's default |
| `:batch` | refused here: with the state frozen, a simultaneous batch in an ordinal likelihood *is* Breslow |

The WTC clock is ordinal (one event per tick), so it has no ties and the policy never bites; on a coarse clock it does:

```julia
# Two events share the timestamp 3.0: refused by default, corrected on request
tied = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0), Event(3, 2, 3.0),
        Event(2, 3, 4.0), Event(1, 2, 5.0), Event(3, 1, 6.0), Event(2, 1, 7.0)]
tied_seq = EventSequence(tied; actors=ActorSet(1:4))
tied_stats = [Repetition(), Reciprocity()]

try
    fit_rem(tied_seq, tied_stats; n_controls=10, seed=1)          # ties=:error — throws
catch err
    println(err.msg[1:60], "…")
end
fit = fit_rem(tied_seq, tied_stats; n_controls=10, seed=1, ties=:efron)   # corrected the way coxph would
Networks.tie_method(fit)             # :efron
```

This is a Cox partial likelihood, so `:breslow` and `:efron` are the classical
corrections in the classical sense — `test/fixtures/rem_ties.toml` pins them against
`survival::coxph(..., ties="breslow"/"efron")` on tied data (agreement < 1e-11).

On tie-free data all four policies produce the identical design and the identical fit.
`Networks.tie_method(fit)` reports what actually happened (`:none` when the data had no
ties), and `Networks.approximations(fit)` carries the caveat when a correction was
applied. A fit with ties in it is never `is_exact`.

## Generating Observations

```julia
# The case-control design as a DataFrame: one row per case and per control
obs = generate_observations(seq, stats, sampler)
size(obs)                       # (48581, 15): 481 cases + 481 × 100 controls
```

The resulting DataFrame contains:

| Column | Description |
|--------|-------------|
| `event_index` | Index of the focal event in the sequence |
| `sender` | Sender ID |
| `receiver` | Receiver ID |
| `is_event` | `true` for cases, `false` for controls |
| `stratum` | Stratum ID (groups each case with its controls) |
| `risk_set_size` | Number of dyads in the stratum's risk set (case included) |
| `sampling_prob` | Probability each non-case dyad entered the sample as a control |
| `tie_weight` | Denominator weight of the row (`1.0` except under `ties=:efron`) |
| `<stat_name>` | One column per statistic |

### Options

```julia
obs = generate_observations(seq, stats, sampler;
    start_index = 1,           # First event to include
    end_index = length(seq),   # Last event to include
    decay = 0.0,               # Exponential decay rate (eventnet's halflife memory)
    window = nothing,          # Or a sliding window: events older than this stop
                               # counting (mutually exclusive with decay > 0)
    at_risk = nothing,         # Risk set: actor universe, RiskSet,
                               # per-event vector, or callback (see below)
    ties = :error,             # Tied-timestamp policy (above)
    rng = Random.default_rng() # Source of the control draw when the sampler has no seed
)
```

### Excluding Early Events

The first few events may have unreliable statistics (no history):

```julia
# Skip the first 20 events
obs = generate_observations(seq, stats, sampler; start_index=21)
```

### The Actor Universe and Risk Sets

The REM likelihood is **conditional on the risk set**: each event competes
against the other dyads that could have occurred instead. Getting the risk set
wrong changes the estimand — eligible actors who never happen to send or
receive an event (isolates, or actors only observed as receivers) still belong
in the denominator, and dropping them biases activity, popularity and covariate
effects. Two of the 37 WTC officers never call or are called; they are in the
risk set because `ActorSet(1:37)` was declared.

Declare the actor universe on the sequence:

```julia
# A small demonstration sequence: actors 1-3 are observed; 7 and 42 are
# eligible isolates
toy_events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
              Event(3, 2, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
toy = EventSequence(toy_events; actors=ActorSet([1, 2, 3, 7, 42]))
toy_stats = [Repetition(), Reciprocity()]
```

If `actors` is omitted, the universe falls back to the observed event endpoints
("participants only") and `fit_rem` warns, because that is a silent change of
estimand.

Alternatively, hand the risk set to `generate_observations`/`fit_rem` directly.
`at_risk` (alias `riskset` in `fit_rem`) accepts:

```julia
toy_sampler = CaseControlSampler(n_controls=10, seed=1)

# 1. A static actor universe (ActorSet, Set{Int} or Vector{Int})
obs = generate_observations(toy, toy_stats, toy_sampler; at_risk=Set(1:10))

# 2. A static RiskSet — senders and receivers may differ (every case must
#    still belong to it: here 7 only ever sends and 42 only ever receives)
rs = RiskSet(0, [1, 2, 3, 7], [1, 2, 3, 42])
obs = generate_observations(toy, toy_stats, toy_sampler; at_risk=rs)

# 3. Per-event risk sets (time-varying membership): one entry per event —
#    here actor k+3 joins the universe at event k
joined = [Set(1:(k + 3)) for k in eachindex(toy)]
obs = generate_observations(toy, toy_stats, toy_sampler; at_risk=joined)

# 4. A callback (event_index, state) -> RiskSet, evaluated against the
#    current network state
obs = generate_observations(toy, toy_stats, toy_sampler;
    at_risk = (i, state) -> RiskSet(i, sort!(collect(state.actors)),
                                    sort!(collect(state.actors))))

toy_result = fit_rem(toy, toy_stats; n_controls=89, at_risk=Set(1:10))   # 10 actors: full risk set
```

Every case is validated against **its own** risk set before any fitting happens:
if the event's sender or receiver is not in the risk set (the case and its
controls would then come from different actor universes), or if the risk set
admits no valid control, an `ArgumentError` is thrown.

Each stratum records its risk-set size and the probability with which each
non-case dyad entered the sample as a control, both on the observation DataFrame
(`risk_set_size`, `sampling_prob`) and on the fitted model
(`result.risk_set_sizes`, `result.sampling_probs`, aligned with `result.strata`).

## Fitting Models

### Direct Fitting (Recommended)

The simplest approach combines sampling and fitting:

```julia
result = fit_rem(seq, stats; n_controls=full)              # exact: every dyad at risk
sampled = fit_rem(seq, stats; n_controls=100, seed=42)     # 100 controls per event
```

### Two-Stage Fitting

For more control over the process:

```julia
using DataFrames   # for nrow

# Stage 1: Generate observations
sampler = CaseControlSampler(n_controls=100, seed=42)
obs = generate_observations(seq, stats, sampler)

# Inspect observations if needed
println("Observations: ", nrow(obs))
println("Cases: ", sum(obs.is_event))
println("Controls: ", sum(.!obs.is_event))

# Stage 2: Fit model
stat_names = [name(s) for s in stats]
two_stage = fit_rem(obs, stat_names)
coef(two_stage) == coef(sampled)          # true: same draw, same design
```

### Fit Options

```julia
two_stage = fit_rem(obs, stat_names;
    maxiter = 100,   # Maximum Newton-Raphson iterations (Networks.newton_fit)
    tol = 1e-8       # Convergence: |Δ log-likelihood| < tol and ‖gradient‖ < √tol
)
```

The optimizer is the ecosystem's one `Networks.newton_fit` (Newton–Raphson
with step halving); REM supplies only its per-stratum conditional-logit kernel.
The keyword vocabulary is the family's: `maxiter`, `tol`, `se`, `rng` (and
`seed`, which pins the control draw — see [Reproducibility](#Reproducibility));
`control_draw_cov` adds `n_draws` and `threaded`.

## Understanding Results

The `REMResult` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Vector{Float64}` | Estimated coefficients |
| `std_errors` | `Vector{Float64}` | Standard errors |
| `z_values` | `Vector{Float64}` | Z-statistics (coef/se) |
| `p_values` | `Vector{Float64}` | Two-sided p-values |
| `stat_names` | `Vector{String}` | Names of statistics |
| `n_events` | `Int` | Number of events (cases) |
| `n_observations` | `Int` | Total observations |
| `log_likelihood` | `Float64` | Log-likelihood at convergence |
| `converged` | `Bool` | Whether optimization converged (an unconverged fit warns, see below) |
| `iterations` | `Int` | Newton–Raphson iterations taken |
| `var_cov` | `Matrix{Float64}` | Covariance matching `se_type` (`vcov(result)`) |
| `se_type` | `Symbol` | The standard-error estimator that was used |
| `tie_type` | `Symbol` | `:none`, or the tie correction that bit |
| `strata` | `Vector{Int}` | Stratum IDs (sorted), indexing the two fields below |
| `risk_set_sizes` | `Vector{Int}` | Risk-set size of each stratum (case included) |
| `sampling_probs` | `Vector{Float64}` | Control sampling probability per stratum |
| `singular` | `Bool` | The observed information at the solution is singular (standard errors `NaN`) — see [Convergence Issues](@ref) |
| `singular_suspects` | `Vector{String}` | The statistics loading on the null direction (collinear, or constant within every stratum) — see [Convergence Issues](@ref) |
| `separated` | `Vector{String}` | Coefficients that may be infinite (`survival::coxph`'s rule) — see [Convergence Issues](@ref) |

### Accessor Functions

`REMResult` implements the full StatsAPI surface — the same verbs a GLM or an
ERGM answers, each a method of the `StatsAPI` generic:

```julia
coef(result)          # Coefficient vector
stderror(result)      # Standard errors vector (== sqrt.(diag(vcov(result))))
vcov(result)          # Covariance matrix, matching se_method(result)
confint(result)       # Wald 95% intervals (p × 2 matrix); confint(result; level=0.9)
loglikelihood(result) # Log partial likelihood
nobs(result)          # Number of events (one stratum each — survival's `nevent`)
dof(result)           # Number of coefficients
aic(result)           # −2ℓ + 2k
bic(result)           # −2ℓ + k·log(n_events)
coeftable(result)     # Networks.CoefficientTable: index by position or by name
```

`coeftable` returns the ecosystem's inspectable `Networks.CoefficientTable`
(`tbl.names`, `tbl.estimates`, `tbl.std_errors`, `tbl.z_values`, `tbl.p_values`;
`tbl["repetition"]` is a row) — the same table `show(result)` prints. Until
0.2.0 it returned a `DataFrame`; build one from the fields if you need it:

```julia
using DataFrames
tbl = coeftable(result)
DataFrame(statistic=tbl.names, coefficient=tbl.estimates, std_error=tbl.std_errors,
          z_value=tbl.z_values, p_value=tbl.p_values)
```

The result-metadata accessors say what the fit *did*, so a script can act on
it instead of parsing `show` output:

```julia
Networks.is_exact(result)            # true — full risk set, no ties, converged
Networks.is_exact(sampled)           # false
Networks.se_method(sampled)          # :hessian
Networks.tie_method(sampled)         # :none — the data had no ties
Networks.approximations(sampled)     # ["case-control sampling of the risk set …", …]
Networks.fit_metadata(sampled)       # all of the above, printed
```

### Displaying Results

```julia
println(sampled)
```

Output:

```text
Relational Event Model Results
==============================
Events: 481, Observations: 48581
Risk-set size: 1332 dyads, control sampling probability: 0.0751
Log-likelihood: -1230.3128
Converged: true (9 iterations)
Std. errors: inverse Hessian (one control draw)

                     Estimate  Std.Error   z value  Pr(>|z|)
repetition            -0.5086     0.0494  -10.3000    <1e-16 ***
reciprocity            0.5488     0.0520   10.5571    <1e-16 ***
sender_activity        0.0287     0.0027   10.8081    <1e-16 ***
receiver_popularity    0.0254     0.0021   11.9042    <1e-16 ***
transitive_closure     0.1503     0.0227    6.6307   3.3e-11 ***
sender_icr             0.7203     0.1824    3.9497   7.8e-05 ***
receiver_icr           1.3571     0.1681    8.0725   6.9e-16 ***
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Note: the risk set was sampled (100 of 1331 controls per event), so this
partial likelihood approximates the full-risk-set one. If the model is
misspecified the estimates depend on the control draw: refit with a
larger `n_controls` or the full risk set, or measure the draw-to-draw
spread with `control_draw_cov`.
```

The block is the family convention (ERGM.jl and Relevent.jl print the same
shape): the header states the design and what was done, the table comes from
the shared `Networks.print_coeftable`, and the note matches the design —
with the full risk set there is none. A fit that is *wrong* rather than
approximate — unconverged, singular, separated — prints a `Warning:` instead
(see [Convergence Issues](@ref)).

## Standard errors: two estimators, and a diagnostic that is not one

```julia
hess = fit_rem(seq, stats; n_controls=100, seed=42)                      # default
sand = fit_rem(seq, stats; n_controls=100, seed=42, se=:sandwich)        # Godambe H⁻¹BH⁻¹
coef(hess) == coef(sand)                       # true — only the covariance differs
round.(stderror(hess), digits=3)               # [0.049, 0.052, 0.003, 0.002, 0.023, 0.182, 0.168]
round.(stderror(sand), digits=3)               # [0.069, 0.077, 0.003, 0.002, 0.027, 0.173, 0.149]
```

| `se=` | what it is |
|---|---|
| `:hessian` (default) | the inverse observed information of the partial likelihood on the risk set that was used. With a **sampled** risk set this is the information of the *sampled* likelihood, which nested case-control theory shows to be a consistent estimator of the estimator's variance (Goldstein & Langholz 1992; Borgan, Goldstein & Langholz 1995): the information lost by sampling fewer controls is what makes it larger than the full-risk-set one, and nothing further needs adding. The package pins this by simulation — 95 % Wald coverage over replicate sequences with events *and* controls redrawn |
| `:sandwich` | the event-clustered Godambe sandwich `H⁻¹BH⁻¹`, robust to misspecification of the within-stratum conditional model. Equals `survival::coxph(..., robust=TRUE)` with the stratum as the cluster (pinned against R) |

Under both, `stderror(fit) == sqrt.(diag(vcov(fit)))` and
`Networks.se_method(fit)` reports which one ran.

There is deliberately **no `se=:bootstrap`**. Redrawing the controls and
refitting does not expose a variance component the Hessian misses — combining
the between-draw covariance of the refits with the Hessian *double-counts*
(97–99 % coverage in the calibration simulation). What the redraws expose is
how far the **point estimate** of a *misspecified* model depends on the draw:
a pseudo-true value that moves with the controls, which a larger `n_controls`
cures and a wider interval does not. That is a sensitivity diagnostic, and it
is offered as one:

```julia
draws = control_draw_cov(seq, stats; n_controls=100, n_draws=10, rng=Xoshiro(7))
round.(draws.sd[1:2], digits=3)                # draw-to-draw sd of repetition, reciprocity
round.(draws.mean[1:2], digits=3)              # their mean over the draws
round.(stderror(hess)[1:2], digits=3)          # [0.049, 0.052] — compare, do not add
```

Read `draws.mean` against a fit at a larger `n_controls` (or the full risk
set): if it moves, the model's pseudo-true value depends on the sampling. Read
`draws.sd` against `stderror(hess)`: a spread of the order of the standard
error says a single draw is not a precise summary of the data. Both point at
the same remedy — more controls. The section on
[choosing the number of controls](@ref choosing-n-controls)
tabulates it on these data.

## Interpreting Coefficients

### Log-Hazard Ratios

Coefficients are log-hazard ratios. A coefficient β means:

- **exp(β)** is the multiplicative effect on the event rate
- **β > 0** increases the rate
- **β < 0** decreases the rate

### Example Interpretations

From the full-risk-set fit above:

| Statistic | Coefficient | exp(β) | Interpretation |
|-----------|-------------|--------|----------------|
| ReceiverAttribute(icr) | 1.31 | 3.7 | A coordinator is called at 3.7× the rate of another officer |
| SenderAttribute(icr) | 0.61 | 1.8 | Coordinators call at 1.8× the rate |
| Reciprocity | 0.33 | 1.4 | Each past call from the callee raises the rate of calling back by 40% |
| TransitiveClosure | 0.13 | 1.14 | Each unit of weighted two-path closure raises the rate by 14% |
| Repetition | −0.34 | 0.71 | Given activity and popularity, each earlier call on the dyad lowers the rate of another |

### Confidence Intervals

Wald intervals come from `confint` (normal reference, `level=0.95` by
default); exponentiate for hazard-ratio intervals:

```julia
ci = confint(result)                 # 7 × 2: lower, upper
ci90 = confint(result; level=0.90)
hr = exp.(confint(result))           # hazard-ratio intervals
tbl = coeftable(result)
[(n, round(exp(lo), digits=2), round(exp(hi), digits=2))
 for (n, lo, hi) in zip(tbl.names, ci[:, 1], ci[:, 2])]
```

## Computing Statistics Without Sampling

To compute statistics for all events (without controls):

```julia
stats_df = compute_statistics(seq, stats; decay=0.0)
```

Returns a DataFrame with one row per event:

| Column | Description |
|--------|-------------|
| `sender` | Sender ID |
| `receiver` | Receiver ID |
| `time` | Event time |
| `<stat_name>` | Statistic values |

**Use case**: Exploratory analysis, visualization, or exporting statistics.

## Model Comparison

### Comparing Log-Likelihoods

Nested models fitted on the **same risk set** (the full one, or the same
`seed`) can be compared by likelihood ratio and by information criteria:

```julia
# Nested models
stats1 = [Repetition(), Reciprocity()]
stats2 = [Repetition(), Reciprocity(), TransitiveClosure()]

result1 = fit_rem(seq, stats1; n_controls=full)
result2 = fit_rem(seq, stats2; n_controls=full)

println("Model 1 LL: ", loglikelihood(result1), "  AIC: ", aic(result1), "  BIC: ", bic(result1))
println("Model 2 LL: ", loglikelihood(result2), "  AIC: ", aic(result2), "  BIC: ", bic(result2))

# Likelihood ratio test
using Distributions
LR = 2 * (loglikelihood(result2) - loglikelihood(result1))
df = dof(result2) - dof(result1)
p_value = ccdf(Chisq(df), LR)
println("LR test p-value: ", p_value)
```

### Multiple Seeds

For a sampled fit, compare results across control draws — by hand, or with
`control_draw_cov`, which refits over independent draws and reports their mean
and spread:

```julia
results = [fit_rem(seq, stats; n_controls=100, seed=s) for s in 1:5]

# Check coefficient stability
for (i, r) in enumerate(results)
    println("Seed $i: ", round.(coef(r), digits=3))
end

draws = control_draw_cov(seq, stats; n_controls=100, n_draws=10, rng=Xoshiro(1))
round.(draws.sd, digits=3)           # the same spread, as a number per coefficient
```

### Reproducibility

Every random draw in `fit_rem` flows through one of two keywords, and never
through the global RNG behind your back:

- `seed` pins the **control draw** to a local `Xoshiro(seed)`: the same seed
  gives the same controls, whatever else is going on.
- `rng` (an `AbstractRNG`, default `Random.default_rng()`) is the source of
  everything `seed` does not pin — the control draw when `seed === nothing`,
  and the per-draw seeds of `control_draw_cov` always. `seed` takes precedence
  over `rng` for the control draw.

```julia
a = fit_rem(seq, stats; n_controls=100, rng=Xoshiro(9))
b = fit_rem(seq, stats; n_controls=100, rng=Xoshiro(9))
coef(a) == coef(b)                   # true — no seed needed

c = control_draw_cov(seq, stats; n_controls=100, n_draws=10, rng=Xoshiro(7))
d = control_draw_cov(seq, stats; n_controls=100, n_draws=10, rng=Xoshiro(7))
c.cov == d.cov                       # true — every draw's seed comes from `rng`
```

The refits of `control_draw_cov` run on all available threads, and the result
is independent of the thread count: `Networks.bootstrap_cov` draws one seed per
replicate from `rng` up front (`threaded=false` gives the same bits serially —
the test suite asserts it).

## Convergence Issues

### An unconverged fit is loud

`fit_rem` runs the shared `Networks.newton_fit` (Newton–Raphson with step
halving) and stops when the log-likelihood change falls below `tol` **and** the
gradient norm below `sqrt(tol)`. If `maxiter` is exhausted first — or no step
improves the likelihood — the fit is returned, but never silently:

- a **warning** is emitted naming `maxiter`, the final gradient norm and `tol`;
- `result.converged` is `false` and `result.iterations` says how far it got;
- `Networks.is_exact(result)` is `false`, and `Networks.approximations(result)`
  carries the entry "the optimizer did not converge … estimates and standard
  errors are not maximum-partial-likelihood values";
- `show(result)` prints `Converged: false (N iterations; see approximations)`
  and a caveat under the coefficient table.

```julia
short = fit_rem(seq, stats; n_controls=full, maxiter=2)   # warns
short.converged                     # false
short.iterations                    # 2
Networks.is_exact(short)            # false
Networks.approximations(short)[1]   # "the optimizer did not converge in 2 iterations — …"
```

So checking the flag is a matter of *acting* on it, not of *discovering* it:

```julia
if !short.converged
    short = fit_rem(seq, stats; n_controls=full, maxiter=500)
end
short.converged                     # true
```

### Common Causes and Solutions

| Issue | Symptom | What the package does | Solution |
|-------|---------|-----------------------|----------|
| Perfect separation | A coefficient of −20 with a standard error of 18 000 — and `Converged: true` | warns naming the statistic, lists it in `fit.separated`, prints a caveat under the table, `is_exact == false` | Remove or transform the statistic (on the WTC calls, `NodeProduct(icr)` separates: no coordinator ever calls another) |
| Multicollinearity | `NaN` standard errors, `Converged: false` | warns naming the suspect statistics, `fit.singular == true`, `fit.singular_suspects`, caveat, `is_exact == false` | Remove one of the collinear statistics (`NodeSum(icr)` with both `SenderAttribute(icr)` and `ReceiverAttribute(icr)` is collinear) |
| Sparse data | Non-convergence | warns, `converged == false` | Increase n_controls, simplify model |
| Too many parameters | Slow convergence | — | Reduce model complexity |

A statistic that is constant across all candidate dyads within every event's
risk set cannot be estimated, even if it changes between events. It contributes
zero information and is reported in `singular_suspects`; remove it from the model.
Adding a common covariate offset within an event's risk set leaves the conditional
likelihood and its standard errors unchanged.
The numerical rank check accounts for the number of rows in the largest risk
set, so rounding in long covariance sums cannot identify a collinear statistic.

Singular information makes the shared optimizer return `converged=false` and
undefined uncertainty, even when the objective has stopped changing. Separation
can still satisfy the objective stopping rule with `converged=true`. The package
names the responsible statistics and records both cases in its diagnostics;
check `singular` and `separated` before interpreting coefficients:

```julia
sep = fit_rem(seq, [Repetition(), NodeProduct(icr)]; n_controls=full)   # warns: may be infinite
sep.separated                       # ["product_icr"]
Networks.is_exact(sep)              # false
col = fit_rem(seq, [NodeSum(icr), SenderAttribute(icr), ReceiverAttribute(icr)]; n_controls=full)
col.singular, col.singular_suspects # (true, ["sum_icr", "sender_icr", "receiver_icr"])
```

Separation is detected as `survival::coxph` detects it ("coefficient may be
infinite"): at a converged solution the Newton increment on a healthy
coordinate is ~1e-15, on a separated one it is O(1) however far the
coefficient has run. The check is scale-free: it is applied to the
standardised coefficient and increment (both multiplied by the column's
standard deviation, which rescale together with the covariate), so
`NodeProduct(icr)` is flagged whether the attribute is coded 0/1, 0/1000 or
0/0.001 — a covariate in natural units (metres, message lengths, counts) never
defeats it.

### Handling Non-Convergence

```julia
# Increase iterations
result = fit_rem(seq, stats; n_controls=full, maxiter=500)

# Belt and braces: the package already warns on separation; a manual check
# on the magnitude costs nothing
for (i, name) in enumerate(result.stat_names)
    if abs(result.coefficients[i]) > 10
        @warn "Possible separation for $name"
    end
end
```

## Advanced Topics

### Is the estimate a property of the draw?

A hand-rolled bootstrap over *events* is the wrong tool for a sampled fit —
resampling events with replacement changes the order and repetition structure
the statistics are built from — and a bootstrap over *controls* is not a
standard error either (the sampled likelihood's own information already
accounts for the sampling; see above). What redrawing the controls answers is
a different question: does the point estimate move with the draw?

```julia
draws = control_draw_cov(seq, stats; n_controls=100, n_draws=20, rng=Xoshiro(7))
round.(draws.mean[1:2], digits=2)                 # ≈ [-0.39, 0.40] — vs the full risk set's [-0.34, 0.33]
draws.sd[1:2] .< 2 .* stderror(hess)[1:2]         # true — the draws scatter by about one SE here
draws.n_unconverged                               # 0
```

`n_draws` replicates cost `n_draws` refits (threaded,
thread-count-independent). With the full risk set there is nothing to redraw
and the spread is exactly zero.

### Time-Varying Effects

Two different questions hide under "effects that change over time":

**Do the coefficients differ between periods?** Split the sequence and fit
each period on its own design (`start_index`/`end_index` live on
`generate_observations`; fit from the observation DataFrames). The network
state still accumulates from the first event, so the late period sees the
full history:

```julia
mid_point = length(seq) ÷ 2
period_sampler = CaseControlSampler(n_controls=100, seed=1)

obs_early = generate_observations(seq, stats, period_sampler;
                                  start_index=1, end_index=mid_point)
obs_late = generate_observations(seq, stats, period_sampler;
                                 start_index=mid_point + 1)

result_early = fit_rem(obs_early, [name(s) for s in stats])
result_late = fit_rem(obs_late, [name(s) for s in stats])

# Compare coefficients
println("Early period: ", round.(coef(result_early), digits=3))
println("Late period:  ", round.(coef(result_late), digits=3))
```

**Does only the recent past matter?** That is a memory model, not a period
split: `decay=` (eventnet's halflife) or `window=` (events older than the
window stop counting) on `fit_rem`, applied to every statistic — see
[Temporal Decay](decay.md):

```julia
recent = fit_rem(seq, stats; n_controls=full, window=50.0)   # the last 50 calls count
```

## Best Practices

1. **Declare the actor universe** (`actors=`): the risk set is the estimand
2. **Enumerate the risk set when affordable**, and check a sampled fit against a larger `n_controls`
3. **Set the seed or the rng** for a sampled fit
4. **Check convergence**: act on `result.converged` (the fit has already warned)
5. **Skip early events**: Consider excluding first few events with `start_index`
6. **Avoid multicollinearity**: Don't include highly correlated statistics
7. **Scale large counts**: Use `LogDegree` for networks with high-degree hubs
8. **Sufficient events**: Aim for at least 10 events per parameter
9. **Compare seeds**: Verify stability across control draws (`control_draw_cov`), and against a larger `n_controls`
