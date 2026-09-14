# Getting Started

Fit an ordinal relational event model to the bundled World Trade Center police
radio calls, inspect its diagnostics, and extend it with history effects.
The data contain 481 calls among 37 eligible officers. Their timestamps are
**event numbers**, so this analysis concerns event choice given the past.

## Prepare the environment

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for this <strong>unreleased 0.2.0 development version</strong>. From the prepared workspace:</p>
```

```bash
julia --project=REM.jl
```

```@raw html
<p><a href="/Networks.jl/dev/">Networks.jl</a> supplies the data. NetworkDynamic.jl is optional and only needed for the dynamic-network conversion extension. No installation commands are part of the analysis below.</p>
```

## Declare the actors and events

```julia
using Networks, REM, Random

wtc = load_dataset(:wtc_police_calls)
n = wtc.n_actors
calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
seq = EventSequence(calls; actors=ActorSet(1:n))
(length(seq), seq.n_actors)                # (481, 37)
```

Declare **all eligible actors**, including those absent from the log. Two officers
in this dataset have no observed calls but remain eligible as alternative senders
or receivers. Inferring eligibility from outcomes changes the model's risk set.

For your own data, construct `Event(sender, receiver, time)` values or use
`load_events` for tables/CSV files, then declare the actor universe. See
[events and data](guide/events.md) for named actors, calendar timestamps and
restricted or changing risk sets. Times tied at the same recorded value require
an explicit `ties` policy; the default throws.

## Start with one covariate

The institutionalized coordinator-role indicator is known for every officer.
A `NodeAttribute` without a default rejects a lookup for a missing actor.

```julia
icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:n))
full_controls = n * (n - 1) - 1
fit = fit_rem(seq, [NodeSum(icr)]; n_controls=full_controls)
@assert fit.converged && !fit.singular && isempty(fit.separated)
coeftable(fit)
exp.(coef(fit))
```

Each call is compared with all 1,331 other directed non-self dyads.
`NodeSum(icr)` is `icr[sender] + icr[receiver]`. A one-unit increase multiplies
the relative event rate by `exp(beta)`, holding other modeled features fixed.
The coefficient is about 2.10 in this model. This association is not a causal
estimate of the effect of becoming a coordinator.

## Add history effects

```julia
stats = [Repetition(), Reciprocity(), SenderActivity(), ReceiverPopularity(),
         TransitiveClosure(), SenderAttribute(icr), ReceiverAttribute(icr)]
extended = fit_rem(seq, stats; n_controls=full_controls)
@assert extended.converged && !extended.singular && isempty(extended.separated)
coeftable(extended)
```

`Repetition` counts past same-direction calls; `Reciprocity` counts calls in the
opposite direction. Activity/popularity terms control for accumulated actor
activity. `TransitiveClosure()` uses eventnet's weighted two-path definition;
`weighted=false` instead counts third parties. Inspect the
[statistic definitions](guide/statistics.md) before interpreting a coefficient.

Do not include `NodeSum(icr)` together with both sender and receiver versions:
they are exactly collinear. On these particular data, `NodeProduct(icr)` also
separates because no coordinator calls another coordinator.

## Read diagnostics before uncertainty

```julia
Networks.fit_metadata(extended)
Networks.approximations(extended)
coef(extended)
stderror(extended)
vcov(extended)
confint(extended)
```

Check `converged`, `singular` and `separated`. Singular information gives NaN
uncertainty and `converged=false`. Separation may coexist with numerical objective
convergence; affected estimates remain unsuitable for inference. Warnings and
metadata describe these conditions. `Networks.is_exact` describes the fitted
objective and recorded caveats, not model adequacy or causality.

The default `se=:hessian` uses inverse observed information. `se=:sandwich` uses
an event-clustered sandwich covariance, leaving coefficients unchanged. Both are
asymptotic. See [model estimation](guide/estimation.md) for assumptions and
comparison of nested models on the same risk set.

## Sample controls when enumeration is too costly

```julia
sampled = fit_rem(seq, stats; n_controls=100, rng=Xoshiro(42))
Networks.fit_metadata(sampled)
Networks.approximations(sampled)
```

A fresh seeded RNG reproduces the draw. Under misspecification, the sampled
estimate can vary with the draw and control count. Compare against more controls;
`control_draw_cov` summarizes variation over repeated draws. It is a sensitivity
diagnostic, **not an additional standard-error estimator**. Do not add its
covariance to the Hessian covariance. Sampled likelihood values from different
control draws are not directly comparable.

## Give memory a meaningful time scale

REM supports either half-life decay (`decay=halflife_to_decay(h)`) or a sliding
window (`window=w`), never both. Numeric values use the clock's units; calendar
clocks also accept `Dates.Period` values. On the WTC ordinal clock, a half-life
of 50 means **50 calls**, not 50 minutes. These settings change the history
statistics, while the likelihood remains conditional on event choice.
See [temporal decay](guide/decay.md).

```@raw html
<p>For actual duration likelihoods or participation shifts, continue with <a href="/Relevent.jl/dev/">Relevent.jl</a>.</p>
```

For REM function signatures and result fields,
use the [estimation API](api/estimation.md).
