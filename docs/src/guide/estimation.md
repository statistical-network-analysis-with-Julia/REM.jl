# Model Estimation

REM.jl estimates relational event models using case-control sampling with stratified conditional logistic regression. This approach is equivalent to Cox proportional hazards for survival data and enables efficient estimation even for large networks.

## Overview

The estimation process follows three steps:

1. **Case-Control Sampling**: For each observed event, sample non-events from the risk set
2. **Statistic Computation**: Calculate statistics for cases and controls
3. **Maximum Likelihood Estimation**: Fit stratified conditional logistic regression

## Why Case-Control Sampling?

For a network with $n$ actors, there are $n(n-1)$ possible directed dyads at each time point. Computing statistics for all dyads at all time points is often computationally infeasible.

Case-control sampling solves this by:

- Treating observed events as "cases"
- Sampling a subset of non-events as "controls"
- Using stratified estimation to obtain consistent parameter estimates

This approach is statistically valid and dramatically reduces computation time.

## Configuring the Sampler

```julia
using REM

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
| `seed` | Random seed (nothing = random each time) | `nothing` |

### Choosing Number of Controls

The number of controls affects estimation accuracy and computation time:

| n_controls | Use Case |
|------------|----------|
| 20-50 | Quick exploratory analysis |
| 50-100 | Standard analysis |
| 100-200 | Final results, publication |
| 200+ | Very precise estimates needed |

More controls = more accurate standard errors, but diminishing returns beyond ~100-200.

Controls are sampled **without replacement**. If the actor set is small enough that fewer distinct dyads exist than `n_controls`, the sampler switches to the full risk set (all distinct non-event dyads) and warns once.

!!! note "Ordinal likelihood"
    REM.jl fits the *ordinal* REM: only event order enters the likelihood; the exact
    inter-event waiting times are not part of the hazard (unlike `relevent::rem.dyad`'s
    interval likelihood). Standard errors are model-based (inverse information) from the
    sampled risk set — the standard nested case-control variance.

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

```julia
fit_rem(seq, stats; ties=:efron)     # tied data, corrected the way coxph would
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
# A demo sequence and statistics
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
          Event(3, 2, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
seq = EventSequence(events; actors=ActorSet([1, 2, 3]))
stats = [Repetition(), Reciprocity()]

# Create observations DataFrame
obs = generate_observations(seq, stats, sampler)
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
| `<stat_name>` | One column per statistic |

### Options

```julia
obs = generate_observations(seq, stats, sampler;
    start_index = 1,           # First event to include
    end_index = length(seq),   # Last event to include
    decay = 0.0,               # Exponential decay rate
    at_risk = nothing          # Risk set: actor universe, RiskSet,
                               # per-event vector, or callback (see below)
)
```

### Excluding Early Events

The first few events may have unreliable statistics (no history):

```julia
# Skip first 5 events
obs = generate_observations(seq, stats, sampler; start_index=6)
```

### The Actor Universe and Risk Sets

The REM likelihood is **conditional on the risk set**: each event competes
against the other dyads that could have occurred instead. Getting the risk set
wrong changes the estimand — eligible actors who never happen to send or
receive an event (isolates, or actors only observed as receivers) still belong
in the denominator, and dropping them biases activity, popularity and covariate
effects.

Declare the actor universe on the sequence:

```julia
# Actors 1-3 are observed in the events; 7 and 42 are eligible isolates
seq = EventSequence(events; actors=ActorSet([1, 2, 3, 7, 42]))
```

If `actors` is omitted, the universe falls back to the observed event endpoints
("participants only") and `fit_rem` warns, because that is a silent change of
estimand.

Alternatively, hand the risk set to `generate_observations`/`fit_rem` directly.
`at_risk` (alias `riskset` in `fit_rem`) accepts:

```julia
# 1. A static actor universe (ActorSet, Set{Int} or Vector{Int})
obs = generate_observations(seq, stats, sampler; at_risk=Set(1:10))

# 2. A static RiskSet — senders and receivers may differ
rs = RiskSet(0, [1, 2, 3], [4, 5, 6])          # e.g. a two-mode network
obs = generate_observations(seq, stats, sampler; at_risk=rs)

# 3. Per-event risk sets (time-varying membership): one entry per event
joined = [Set(1:k) for k in eachindex(seq)]
obs = generate_observations(seq, stats, sampler; at_risk=joined)

# 4. A callback (event_index, state) -> RiskSet, evaluated against the
#    current network state
obs = generate_observations(seq, stats, sampler;
    at_risk = (i, state) -> RiskSet(i, sort!(collect(state.actors)),
                                    sort!(collect(state.actors))))

result = fit_rem(seq, stats; n_controls=50, at_risk=Set(1:10))
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
result = fit_rem(seq, stats;
    n_controls = 100,
    seed = 42
)
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
result = fit_rem(obs, stat_names)
```

### Fit Options

```julia
result = fit_rem(obs, stat_names;
    maxiter = 100,   # Maximum Newton-Raphson iterations
    tol = 1e-8       # Convergence tolerance for log-likelihood
)
```

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
| `converged` | `Bool` | Whether optimization converged |
| `strata` | `Vector{Int}` | Stratum IDs (sorted), indexing the two fields below |
| `risk_set_sizes` | `Vector{Int}` | Risk-set size of each stratum (case included) |
| `sampling_probs` | `Vector{Float64}` | Control sampling probability per stratum |

### Accessor Functions

```julia
coef(result)        # Coefficient vector
stderror(result)    # Standard errors vector
coeftable(result)   # Full results as DataFrame
```

### Displaying Results

```julia
println(result)
```

Output:

```text
Relational Event Model Results
==============================
Events: 100, Observations: 10100
Log-likelihood: -234.5678
Converged: true

Coefficients:
------------------------------------------------------------
Statistic                  Coef    Std.Err          z      P>|z|
------------------------------------------------------------
repetition               0.4523     0.0812     5.5700     0.0000 ***
reciprocity              0.3156     0.0923     3.4200     0.0006 ***
sender_activity          0.0234     0.0156     1.5000     0.1336
receiver_popularity      0.0567     0.0189     3.0000     0.0027 **
------------------------------------------------------------
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Interpreting Coefficients

### Log-Hazard Ratios

Coefficients are log-hazard ratios. A coefficient β means:

- **exp(β)** is the multiplicative effect on the event rate
- **β > 0** increases the rate
- **β < 0** decreases the rate

### Example Interpretations

| Statistic | Coefficient | exp(β) | Interpretation |
|-----------|-------------|--------|----------------|
| Repetition | 0.5 | 1.65 | Each past s→r event increases rate by 65% |
| Reciprocity | 0.8 | 2.23 | Past r→s events double the rate |
| SenderActivity | -0.1 | 0.90 | High-activity senders have 10% lower per-dyad rates |
| TransitiveClosure | 0.3 | 1.35 | Each shared partner increases rate by 35% |
| AttributeMatch | 0.4 | 1.49 | Same-attribute dyads are 49% more likely |

### Confidence Intervals

Approximate 95% confidence intervals:

```julia
using Distributions

alpha = 0.05
z = quantile(Normal(), 1 - alpha/2)

lower = coef(result) .- z .* stderror(result)
upper = coef(result) .+ z .* stderror(result)

# Hazard ratio confidence intervals
hr_lower = exp.(lower)
hr_upper = exp.(upper)
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

```julia
# Nested models
stats1 = [Repetition(), Reciprocity()]
stats2 = [Repetition(), Reciprocity(), TransitiveClosure()]

result1 = fit_rem(seq, stats1; n_controls=100, seed=42)
result2 = fit_rem(seq, stats2; n_controls=100, seed=42)

println("Model 1 LL: ", result1.log_likelihood)
println("Model 2 LL: ", result2.log_likelihood)

# Likelihood ratio test (approximate)
using Distributions
LR = 2 * (result2.log_likelihood - result1.log_likelihood)
df = length(stats2) - length(stats1)
p_value = 1 - cdf(Chisq(df), LR)
println("LR test p-value: ", p_value)
```

### Multiple Seeds

For robustness, compare results across different random seeds:

```julia
results = [fit_rem(seq, stats; n_controls=100, seed=s) for s in 1:5]

# Check coefficient stability
for (i, r) in enumerate(results)
    println("Seed $i: ", round.(coef(r), digits=3))
end
```

## Convergence Issues

### Checking Convergence

```julia
if !result.converged
    @warn "Model did not converge - results may be unreliable"
end
```

### Common Causes and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Perfect separation | Very large coefficients | Remove or transform problematic statistic |
| Multicollinearity | Large standard errors | Remove correlated statistics |
| Sparse data | Non-convergence | Increase n_controls, simplify model |
| Too many parameters | Slow convergence | Reduce model complexity |

### Handling Non-Convergence

```julia
# Increase iterations
result = fit_rem(seq, stats; n_controls=100, maxiter=500)

# Check for separation
for (i, name) in enumerate(result.stat_names)
    if abs(result.coefficients[i]) > 10
        @warn "Possible separation for $name"
    end
end
```

## Advanced Topics

### Bootstrapping Standard Errors

For more robust standard errors:

```julia
function bootstrap_rem(seq, stats; n_bootstrap=100, n_controls=100)
    n = length(seq)
    boot_coefs = zeros(n_bootstrap, length(stats))

    for b in 1:n_bootstrap
        # Resample events with replacement
        indices = rand(1:n, n)
        boot_events = [seq[i] for i in sort(indices)]
        boot_seq = EventSequence(boot_events)

        result = fit_rem(boot_seq, stats; n_controls=n_controls, seed=b)
        boot_coefs[b, :] = coef(result)
    end

    # Bootstrap standard errors
    boot_se = vec(std(boot_coefs, dims=1))
    return boot_se
end
```

### Time-Varying Effects

Model effects that change over time using event windows:

```julia
# Split sequence into periods (start_index/end_index live on
# generate_observations; fit from the observation DataFrames)
mid_point = length(seq) ÷ 2
sampler = CaseControlSampler(n_controls=100, seed=1)

obs_early = generate_observations(seq, stats, sampler;
                                  start_index=1, end_index=mid_point)
obs_late = generate_observations(seq, stats, sampler;
                                 start_index=mid_point + 1)

result_early = fit_rem(obs_early, [name(s) for s in stats])
result_late = fit_rem(obs_late, [name(s) for s in stats])

# Compare coefficients
println("Early period: ", coef(result_early))
println("Late period: ", coef(result_late))
```

## Best Practices

1. **Set random seed**: Always use a seed for reproducibility
2. **Check convergence**: Verify `result.converged == true`
3. **Adequate controls**: Use at least 50-100 controls
4. **Skip early events**: Consider excluding first few events with `start_index`
5. **Avoid multicollinearity**: Don't include highly correlated statistics
6. **Scale large counts**: Use `LogDegree` for networks with high-degree hubs
7. **Sufficient events**: Aim for at least 10 events per parameter
8. **Compare seeds**: Verify stability across different random seeds
