# REM.jl

Model **which dyad acts next**, given past interactions and the actors eligible
to participate. REM.jl fits ordinal relational event models to sender–receiver
event sequences, using eventnet-style history statistics and stratified
conditional logistic regression.

**Start here:** [Getting started](getting_started.md) ·
[Events and risk sets](guide/events.md) · [Statistics](guide/statistics.md) ·
[Estimation API](api/estimation.md)

## Fit the bundled radio calls

The World Trade Center police radio dataset contains 481 calls and a declared
universe of 37 officers, including two who never appear in the calls. Its clock
is event order. The model below relates event choice to whether the sender or
receiver holds an institutionalized coordinator role.

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for the current <strong>0.2.0 development version, unreleased</strong>. The examples assume that environment is already prepared.</p>
```

```julia
using Networks, REM

wtc = load_dataset(:wtc_police_calls)
n = wtc.n_actors
calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
seq = EventSequence(calls; actors=ActorSet(1:n))
icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:n))

# Compare each call with every other eligible directed dyad.
fit = fit_rem(seq, [NodeSum(icr)]; n_controls=n * (n - 1) - 1)
@assert fit.converged && !fit.singular && isempty(fit.separated)
coeftable(fit)
Networks.fit_metadata(fit)
```

`NodeSum(icr)` is the sender-plus-receiver role covariate. Its coefficient is
about 2.10 on these data. The [tutorial](getting_started.md) adds repetition,
reciprocity and actor activity, then explains sampled controls and uncertainty.

## Choose the right likelihood and risk set

| Need | Current support |
|---|---|
| Next-event choice | Ordinal partial likelihood; enumerate the full risk set when affordable or sample controls with a seeded RNG. |
| Changing eligibility | Declared actor sets, asymmetric sender/receiver sets, per-event risk sets and callbacks; every observed event must be eligible. |
| History and time | Dyadic, degree, triadic, four-cycle and attribute statistics; optional half-life decay or a sliding window. |
| Uncertainty and diagnostics | Hessian or event-clustered sandwich covariance, convergence/identification warnings, fit metadata and a separate control-draw sensitivity diagnostic. |

```@raw html
<p>REM.jl does <strong>not fit inter-event durations or a baseline event rate</strong>. Use <a href="/Relevent.jl/dev/">Relevent.jl</a> for its supported exponential timing model and participation-shift effects. Use <a href="/Siena.jl/dev/">Siena.jl</a> when your observations are network waves rather than individual events.</p>
```

Tied timestamps require an explicit policy; the default is refusal. With sampled
controls, a misspecified model's estimate can depend on the draw and control count.
Check `converged`, `singular`, `separated` and `Networks.approximations(fit)` before
inference: numerical convergence alone does not establish identification.
Reference fixtures validate selected designs and fits against R, not every model
an analyst can specify. See [model estimation](guide/estimation.md).

Package citation: [CITATION.bib](https://github.com/statistical-network-analysis-with-Julia/REM.jl/blob/main/CITATION.bib).

## Module

```@docs
REM
```
