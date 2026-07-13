# Estimation API Reference

This page documents the functions for data loading, observation generation, and model fitting.

## Data Loading

### load_events

```@docs
load_events
load_events!
```

## Observation Generation

### Observation

```@docs
Observation
```

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

## Model Fitting

### fit_rem

```@docs
fit_rem
```

### REMResult

```@docs
REMResult
```

### Result Accessors

```@docs
coef
stderror
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

The case-control sampling design that a fit actually ran on. These are what the
standard errors condition away when `se=:hessian`: `se=:bootstrap` redraws the
risk set and exposes the sampling variability they hide.

```@docs
risk_set_sizes
sampling_probs
```

## Result Metadata

REM.jl implements the ecosystem's
[result-metadata protocol](https://Statistical-network-analysis-with-Julia.github.io/Networks.jl/dev/api/metadata/),
so what a fit actually did is programmatically inspectable via
`Networks.fit_metadata(result)` rather than buried in a `show` method.

`Networks.is_exact` is the one to read first: the case-control partial likelihood
is exact only when the sampled risk set is the full risk set. `tie_method`
reports the tie policy that actually **ran** — never `:error`, because a fit
that would have had to break a tie under `ties=:error` threw instead of
returning.

The remaining accessors (`estimand`, `missing_method`, `approximations`) take
their documentation from the generics in the Networks.jl manual.

```@docs
Networks.objective(::REMResult)
Networks.is_exact(::REMResult)
Networks.se_method(::REMResult)
Networks.tie_method(::REMResult)
```
