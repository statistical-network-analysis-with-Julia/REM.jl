# Estimation API Reference

This page documents the functions for data loading, observation generation, and model fitting.

## Data Loading

### load_events

```@docs
load_events
load_events!
```

## Observation Generation

The design is the `DataFrame` that `generate_observations` returns (one row
per case or control: `event_index`, `sender`, `receiver`, `is_event`,
`stratum`, `risk_set_size`, `sampling_prob`, `tie_weight` and one column per
statistic); a hand-built design is a `DataFrame` with at least `is_event`,
`stratum` and the statistic columns.

### CaseControlSampler

```@docs
CaseControlSampler
```

### generate_observations

```@docs
generate_observations
```

### compute_statistics

```@docs
compute_statistics
```

### Tie corrections and sampled controls

`ties=:efron` is offered with the full risk set only; with sampled controls
the design is biased toward zero and is refused, pointing at `:breslow`. The
guard is `public` (not exported) so that Revel.jl's design refuses the same
combination with the same words.

```@docs
REM.check_tie_sampling
```

## Model Fitting

### fit_rem

```@docs
fit_rem
```

### REMResult

```@docs
REMResult
```

### Result Accessors (the StatsAPI surface)

`REMResult` answers all eleven verbs of the ecosystem's StatsAPI surface — each a
method on the `StatsAPI` generic that StatsBase, GLM and NetworkCore share, so
`using REM, StatsBase` dispatches one `coef`. `NetworkCore.check_statsapi(fit;
strict=true)` passes. `coeftable` returns the shared, inspectable
`NetworkCore.CoefficientTable` (not a `DataFrame`), and `vcov` is the covariance
that matches `se_method(fit)`: `stderror(fit) == sqrt.(diag(vcov(fit)))` under
`:hessian` and `:sandwich` alike.

```@docs
coef
coefnames
stderror
vcov
confint
loglikelihood
nobs
dof
aic
bic
coeftable
```

## Utility Functions

### Time Decay

```@docs
halflife_to_decay
decay_to_halflife
compute_decay_weight
```

### Risk Set Utilities

```@docs
n_dyads
```

### Sampling Design

The case-control sampling design that a fit actually ran on — part of the
estimand, since the partial likelihood is conditional on the risk set.

```@docs
risk_set_sizes
sampling_probs
```

### Control-draw sensitivity

How far the point estimate of a sampled fit depends on which controls were
drawn. A diagnostic, not a standard error (the docstring says why there is no
`se=:bootstrap`).

```@docs
control_draw_cov
```

## Result Metadata

REM.jl implements the ecosystem's
[result-metadata protocol](https://Statistical-network-analysis-with-Julia.github.io/NetworkCore.jl/dev/api/metadata/),
so what a fit actually did is programmatically inspectable via
`NetworkCore.fit_metadata(result)` rather than buried in a `show` method.

`NetworkCore.is_exact` is the one to read first: the case-control partial likelihood
is exact only when the sampled risk set is the full risk set. `tie_method`
reports the tie policy that actually **ran** — never `:error`, because a fit
that would have had to break a tie under `ties=:error` threw instead of
returning.

The remaining accessors (`estimand`, `missing_method`, `approximations`) take
their documentation from the generics in the NetworkCore.jl manual.

```@docs
NetworkCore.objective(::REMResult)
NetworkCore.is_exact(::REMResult)
NetworkCore.se_method(::REMResult)
NetworkCore.tie_method(::REMResult)
```
