# Changelog

All notable changes to REM.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Types are renamed to compose with ERGM.jl and Graphs.jl, decay is lazy (O(1) event
updates), control sampling is corrected, and results adopt the ecosystem's
StatsAPI/presentation conventions.

**Dependencies renamed:** the foundation package is now `NetworkCore` (developed as `Networks`) and the optional dynamic-network package is now `DynamicNetworks` (developed as `NetworkDynamic`); update `using` lines accordingly, and the package extension is now `REMDynamicNetworksExt`. Types and functions keep their names.

### Breaking

- **`ties=:efron` with sampled controls is refused** (`ArgumentError` pointing at
  `:breslow`): the forced tied cases dominated the sampled denominator and biased
  estimates toward zero (truth 1.0: 0.63 at 5 controls, 0.88 at 100, 0 % coverage).
  Efron stays available with the full risk set; `:breslow` is unbiased with sampled
  controls (simulation testset). `fit_rem(::DataFrame)` refuses an Efron design with
  sampled strata. The guard is the `public` `REM.check_tie_sampling`, which Revel.jl calls.
- **A tie block straddling `start_index` is a tie**: `ties=:error` refuses it and
  `:breslow`/`:efron` no longer absorb its earlier members before the window opens
  (they leaked into the statistics of their simultaneous partners).
- **`compute_statistics` follows the tied-event contract**: new `ties=:error` default;
  `:breslow`/`:efron` freeze the state across a tie, `:ordered` is the old behaviour.
- **A sequence with more than one event type is refused** by `fit_rem`,
  `generate_observations`, `compute_statistics` and `control_draw_cov` until the
  caller chooses `eventtypes=:pool` or `cases_of=` (untyped statistics used to
  pool the types silently).
- **Event weights must be finite and non-negative** in `EventSequence` and
  `update!` (a `NaN` propagated; a negative weight zeroed two-paths). `Event` itself
  stays permissive.
- **Two statistics with one name are refused** by `StatisticSet` (one column used to
  overwrite the other). *Migration:* give variants distinct `name=`s.
- **`seed` is replaced by `rng`** (`fit_rem`, `CaseControlSampler`); there is no
  alias, and `CaseControlSampler` has no `seed` field. `seed` used to override
  `rng` silently. *Migration:* `seed=k` → `rng=Xoshiro(k)` (the same controls).
- **The fallback `compute(::AbstractStatistic, state, sender, receiver)` is untyped**,
  so a user's `compute(::MyStat, state, i, j)` is no longer ambiguous with it; a
  statistic without a method gets an `ArgumentError` naming the fix.
- **`NodeMatch` renamed to `AttributeMatch`** (collided with `ERGM.NodeMatch`); no alias.
  *Migration:* replace `NodeMatch(` with `AttributeMatch(`.
- **`NetworkState` renamed to `EventNetworkState`** (collided with Siena); no alias.
  *Migration:* replace `NetworkState` with `EventNetworkState`, including `{T}` forms.
- **`NodeMix` renamed to `ActorMix`** (collided with `ERGM.NodeMix`); no alias.
  *Migration:* use `ActorMix(`.
- **`has_edge` is a method of the shared `Graphs.has_edge`** (`REM.has_edge ===
  NetworkCore.has_edge`), so it is exported again and `using Graphs, REM` no longer clashes.
- **`compute`, `name` and `compute_all` are the shared NetworkCore.jl generics**, so
  `using ERGM, REM` no longer leaves them undefined. *Migration:* extensions must
  `import NetworkCore: compute, name` (or `import REM: ...`, the same bindings).
- **`generate_observations` samples controls without replacement**: exactly
  `min(n_controls, available)` distinct controls per event; asking for more than the
  risk set warns and uses all of it. Estimates change versus 0.1.0.
- **The sampler seed no longer reseeds the global RNG** (a local `Xoshiro` is used).
  *Migration:* seed the global RNG yourself if you relied on that side effect.
- **`fit_rem(seq, stats)` draws controls from the `rng` keyword**
  (default `Random.default_rng()`). Calls that passed `rng=` now get different,
  reproducible controls.
- **`StatisticSet` is tuple-backed** (`StatisticSet{T<:Tuple}`), still constructible
  from a vector. *Migration:* treat it as an iterable, or pass plain vectors to `fit_rem`.
- **`se=:bootstrap` and `n_boot` are removed.** Under case-control sampling the
  inverse Hessian is already consistent (Wald coverage 0.953/0.967); the bootstrap
  double-counted. `se=:bootstrap` is refused with a pointer to `control_draw_cov`;
  `se_type` is `:hessian` or `:sandwich`. *Migration:* use the default fit plus
  `control_draw_cov(seq, stats; n_draws=B, rng=r)`, or `se=:sandwich`.
- **`Observation` and `observations_to_dataframe` are removed.** *Migration:* build
  the design `DataFrame` (`is_event`, `stratum`, statistic columns) directly.
- **`RecencyStatistic(transform=:exp_decay)` returns `1.0` at zero elapsed time**
  when the dyad has a prior event (it returned `0.0`, i.e. "never happened").
  `:inverse`/`:inverse_log` keep the documented `0.0` cap at Δ = 0.
- **`RecencyStatistic(transform=:log)` is refused; use `:inverse_log`** (same formula,
  `1/log(1+Δ)`, name `recency_inverse_log`). Unknown transforms throw at construction.
- **`RecencyStatistic(directed=false)` reads the last event in either direction**
  (it looked up only the `(min, max)` dyad); values change where the latest event
  was the other direction.
- **The triadic and four-cycle statistics default to eventnet's weighted
  definition** (`weighted=true, aggregation=:min`) for `TransitiveClosure`,
  `CyclicClosure`, `SharedSender`, `SharedReceiver`, `CommonNeighbors` (weighted on
  undirected counts) and `FourCycle`. *Migration:* `weighted=false` restores the
  0.1 distinct-third-party count.
- **`NodeAttribute(name, values)` without a default throws on a missing actor**
  (`ArgumentError` naming attribute and actor); `NodeAttribute(name, values, default)`
  remains the explicit fill.
- **Singular or unidentified observed information gives `converged=false`** and NaN
  uncertainty (shared `NetworkCore.newton_fit` rank guard).
- **A separated fit reports `converged=false` and withholds inference**, following the
  ecosystem's separation policy: a warning, the separated statistics in `fit.separated`,
  and `NaN` z values, p-values and confidence intervals. Separation is decided on the
  design by NetworkCore.jl's shared verdict (`NetworkCore.clogit_separation`), not by a
  rule on the Newton increment, so it does not depend on where the optimizer stopped.
- **A case must belong to its own risk set**: an event outside `at_risk`, or a risk
  set with no valid control, throws an `ArgumentError`. *Migration:* declare the
  universe (`EventSequence(events; actors=...)`) or widen `at_risk`.
- **`EventSequence(::DynamicNetwork)` takes the actor universe from the vertex set**,
  so isolates stay in the risk set. *Migration:* pass `actors=` for the old behaviour.
- **`REMResult` gained fields** (`strata`, `risk_set_sizes`, `sampling_probs`,
  `var_cov`, `iterations`, `singular`, `singular_suspects`, `separated`); the padding
  9/12/13/14-argument constructors are gone. *Migration:* a hand-built result names
  all nineteen fields.
- **`coeftable(fit)` returns a `NetworkCore.CoefficientTable`, not a `DataFrame`**
  (fields `names`, `estimates`, `std_errors`, `z_values`, `p_values`; rows by index or
  name). *Migration:* read the fields or rebuild a `DataFrame` from them.
- **`EventNetworkState.event_history` is kept only when a statistic needs it**
  (`keep_history = any(needs_history, stats)`; unknown statistics default to `true`).
  Hand-built states keep it by default. *Migration:* `REM.needs_history(::MyStat) = true`.
- **The far-tail p-value floor is `floatmin(Float64)`** (shared `NetworkCore.z_pvalues`),
  not a subnormal or `0.0`; `NaN` where the SE is not positive.
- **`get_out_neighbors`/`get_in_neighbors` return `AbstractSet{Int}`**; an isolated
  actor gets a shared immutable empty set (`copy` it for a `Set{Int}`).
- **One-line `show` for `EventSequence`, `EventNetworkState`, `ActorSet`, `RiskSet`,
  `CaseControlSampler` and `StatisticSet`** instead of a field dump. `show(::REMResult)`
  is unchanged except the SE line reads `inverse Hessian (full risk set)`/`(one control draw)`.
- **`EventNetworkState` gained an `n_events::Int` field** (events absorbed since
  `reset!`); the keyword constructor is unchanged.
- **Minimum Julia is 1.12; package UUID regenerated.** *Migration:* re-resolve
  environments pinning the old UUID.
- The unexported `events_before`/`events_in_window` helpers are removed.

### Added

- **Event types are usable**: `OfType(stat, type)` computes any state-reading
  statistic on one type's history (typed sub-states, `EventNetworkState(seq;
  types=)`), and `cases_of=` models the events of some types while all events
  build the history.
- **`TimeVaryingCovariate(f)`**: an exogenous covariate `f(sender, receiver, time)`
  read at the time of the event being explained.
- **Dyad-level risk sets**: `at_risk=` accepts a list of `(sender, receiver)` pairs
  (also `RiskSet(i, pairs)`, per event or from a callback); exactly the listed dyads
  are at risk. `RiskSet` gained a `dyads` field.
- **Undirected events: `directed=false`** on `generate_observations`, `fit_rem` and
  `control_draw_cov` rewrites each event as `(min, max)` and puts the `n(n−1)/2`
  unordered pairs of a symmetric actor set at risk (was: the `n(n−1)` ordered dyads).
- **`eventtypes=:error|:pool`** and **`ties=`** keywords on `compute_statistics`; its
  frame carries the `"tie_method"` metadata.
- **README "Not implemented" and "R and other software" sections**; the docs route
  interval timing, participation shifts and typed effects to Revel.jl.
- **Aqua.jl testset**; `[compat]` for the test-only extras.
- **Full StatsAPI surface on `REMResult`**: `coef`, `coefnames` (the statistic names,
  R's `names(coef(fit))`), `stderror`, `coeftable`, `vcov`, `confint` (Wald),
  `loglikelihood`, `nobs` (events), `dof`, `aic`, `bic`, all exported StatsAPI methods. `vcov` matches `se_method(fit)`, so `stderror == sqrt.(diag(vcov))`.
- **Robust standard errors: `se=:hessian|:sandwich`** on both `fit_rem` methods.
  `:sandwich` is the event-clustered Godambe sandwich `H⁻¹BH⁻¹`; point estimates are
  unchanged. `se_method(fit)` reports what was used.
- **`control_draw_cov(seq, stats; n_controls, n_draws=20, rng, threaded)`**: the
  between-draw spread of a sampled fit as a sensitivity diagnostic, returning
  `(cov, sd, replicates, mean, n_unconverged)`; never combined with `stderror`.
- **Collinearity and separation are loud.** A singular information sets `fit.singular`
  and names `fit.singular_suspects`; separated statistics (the shared verdict) are
  listed in `fit.separated`. Both warn, are recorded in `approximations`, make
  `is_exact` false and are captioned by `show`.
- **An unconverged fit is loud**: `fit_rem` warns (naming `maxiter`, gradient norm,
  `tol`), sets `converged == false`, records `iterations`, makes `is_exact` false and
  adds an `approximations` entry; `show` prints `Converged: … (N iterations)`.
- **`show` notes sampled risk sets and warns about wrong fits** (unconverged,
  singular, separated) under the table; a healthy full-risk-set fit prints nothing extra.
- **Tied event times are a policy: `ties=:error|:ordered|:breslow|:efron`**
  (shared `NetworkCore.TIE_POLICIES`). The default `:error` names the tie; `:ordered` is
  the pre-0.2 behaviour; `:efron` adds a `tie_weight` column and needs distinct tied
  dyads; `:batch` is refused. Tie-free data give identical designs under all four.
- **`tie_method(fit)` reports what actually happened** (`:none` on untied data);
  `is_exact` requires a full risk set and untied data. The policy travels as DataFrame
  metadata; an `:efron` frame missing `tie_weight` is refused.
- **Sliding-window memory: `window=`** on `EventNetworkState`, `generate_observations`,
  `compute_statistics`, `fit_rem` (and `control_draw_cov`). Expired events leave counts,
  degrees and adjacency (exact `0.0`). Mutually exclusive with `decay > 0`.
- **`Dates.Period` windows on `Date`/`DateTime` clocks** (`window=Day(2)`, also
  `halflife_to_decay(Day(7))`), converted to seconds; a `Period` window on a numeric
  clock throws an `ArgumentError`.
- **`aggregation=` (`:min`, `:max`, `:sum`, `:product`)** on the triadic and
  four-cycle statistics, validated at construction.
- **Explicit actor universe**: `EventSequence(events; actors=...)` (`ActorSet`,
  `Set{Int}`, `Vector{Int}` or range); out-of-universe endpoints throw, and fitting on
  an inferred universe warns.
- **Explicit risk sets** via `fit_rem(...; at_risk=)` (alias `riskset`): a static
  universe, a `RiskSet`, per-event risk sets, or a callback `(event_index, state) -> RiskSet`.
- **Risk-set bookkeeping**: per-stratum `risk_set_size`/`sampling_prob`, exposed by
  the exported `sampling_probs(fit)` and `risk_set_sizes(fit)` and shown by `show`.
- **DynamicNetworks extension** (`using DynamicNetworks`): `EventSequence(::DynamicNetwork;
  eventtype=:onset, weight=..., include_onset_censored=...)`, with `missing=:error|:face`
  and `report=true` returning a `NetworkCore.ConversionReport`.
- **Golden fixtures against R**: `rem_clogit.toml` (`survival::clogit`, < 1e-13, plus
  `robust_std_errors` pinning `se=:sandwich`), `rem_ties.toml` (`coxph` Breslow/Efron,
  < 1e-11), `rem_eventnet.toml` (weighted triadic, aggregations, decay, windows; 1e-8)
  and `rem_relevent_wtc.toml` (`relevent::rem.dyad` on WTC police calls; 1e-6).
- **Actionable errors**: missing actor on a default-less `NodeAttribute`; wrong value
  type for `ActorMix`/`*Categorical` (exactly convertible values are coerced); numeric
  statistics on categorical attributes; misspelt statistic columns; a `Vector{Event}`
  where an `EventSequence` is expected; `decay` combined with `window`.
- **More forgiving entry points**: `fit_rem(seq, Repetition())` accepts a single
  statistic; swapped arguments (`fit_rem(stats, seq)`) throw an `ArgumentError`; a 0/1
  integer `is_event` column is accepted, other `is_event`/`stratum` types are refused.
- **Clear refusals for an empty statistics list and self-loop events** (the latter
  names the loop and `exclude_self_loops=false`).
- **`fit_rem` input validation**: missing statistic/`is_event`/`stratum` columns,
  strata without exactly one case, and strata with no controls throw `ArgumentError`.
- **`NodeAttribute` prints one line** (`NodeAttribute{Float64}(:icr, 37 actors, no default)`).
- `has_default(attr)` and the `needs_history(stat)` trait (exported);
  `EventNetworkState(seq; keep_history=)`.
- `generate_observations(...; rng=)`; `eachindex`, `firstindex` and `eltype` on
  `EventSequence` (so `collect(seq)` is a `Vector{Event{T}}`).
- `compute_all!` for in-place statistic evaluation.
- **A runnable example in every exported docstring**, executed by a testset.
- **Docs teach on the bundled WTC police-calls stream** (`load_dataset(:wtc_police_calls)`),
  reproducing relevent's `wtcfit1` (2.1045) on the full risk set.
- **New testsets**: hand-computed values for every statistic, 0-byte allocation pins
  for every statistic, case-control consistency, calibrated sampled-risk-set SEs,
  compact `show`, no export colliding with `Base`, O(events) streaming, `rng` contract.
- **Benchmark suite with scaling assertions** and a standalone
  `benchmark/regression_tests.jl` allocation gate.
- **PrecompileTools workload** covering the whole fit pipeline (see Performance).
- **CI runs one cell with `JULIA_NUM_THREADS=4`** (threaded vs serial
  `control_draw_cov` bit-identical) and runs the benchmark regression gate.

### Changed

- `n_controls = n(n−1) − 1` no longer warns "only … available" when an Efron tie block
  holds other tied cases; the warning fires only when the risk set itself is too small.
- The risk-set provider closures no longer box their captured variables.
- `benchmark/benchmarks.jl` reads its two time-ratio bounds as the median of seven
  alternating re-timings (fastest sample each), so machine load no longer fails
  the gate; the 2.5x limit is unchanged.
- The `EventSequence(::DynamicNetwork)` extension reports base edges that have no
  spell record (`:default_active_edges`) instead of omitting them silently.
- **The estimation loop is the shared `NetworkCore.newton_fit`** (step halving, combined
  stopping rule, SEs from the post-update Hessian); NaN with a warning when the Hessian
  is not negative definite. Compensated summation brings the Breslow fixture to 3e-16.
- **Decay is lazy**: counts decay on read; the unexported eager `apply_decay!` is
  removed. *Migration:* drop any `REM.apply_decay!` call; accessors are unchanged.
- **`se=` is validated by the shared `NetworkCore.check_se`**; only the explanatory
  `se=:bootstrap` refusal stays in REM.
- **No `rem` alias of `fit_rem`** (it would collide with `Base.rem`); relevent's
  `rem.dyad` likelihoods are Revel.jl's `fit_revel`.
- `REMResult` prints through `NetworkCore.print_coeftable` (p-values floored at `<1e-16`).
- `EventNetworkState` is concretely typed; risk-set actors are sorted before sampling.
- Constructing an `Event` with `sender == receiver` no longer warns.
- **Docs and README describe current behaviour** (DynamicNetworks is a weak dependency,
  `coeftable`/`confint`/`control_draw_cov`, `window=`, weighted triadic defaults);
  every snippet runs cleanly. Documenter default themes with a new package icon.
- `docs/make.jl` keeps `"stable" => "dev"` until 0.2.0 is tagged.
- `benchmark/Project.toml` sources NetworkCore at `../../NetworkCore.jl`, so a fresh clone
  instantiates; CI/Documentation clone lists are derived from `[sources]`.
- `CLAUDE.md` rewritten to the current code.

### Fixed

- **Covariates are centred within each conditional-logit stratum**, so a statistic
  constant within every risk set has exactly zero information on every platform (no
  spurious separation or finite SE).
- **`EventSequence(::DynamicNetwork)` no longer converts masked dyads silently**; it
  refuses a masked network unless `missing=:face` is passed.
- **`EventSequence(::DynamicNetwork)` no longer emits an event at `-Inf`.** An open-left
  spell (onset `-Inf`, or `typemin` on an integer clock: a tie present before
  observation) is onset-censored: skipped by default and counted in the
  `:onset_censored_spells` report entry. With `include_onset_censored=true` it enters at
  the start of the observation period, and without a finite start it is an
  `ArgumentError`. On `DateTime` and `Date` axes the open bound is the axis minimum
  (`DynamicNetworks.unbounded_spell`), and it is recognised there too: it used to
  become an event in the year −146138511.
- **`EventSequence(::DynamicNetwork)` treats an onset before the observation window as
  onset-censored** (skipped and reported, or placed at the window start with
  `include_onset_censored=true`), as TSNA does; it used to be an event outside the
  window.
- **A fit whose information is singular at the starting value says so.** It used to
  warn "did not converge in 1 of maxiter = 100 iterations" and advise raising
  `maxiter`; the warning, `show` and `approximations` now name the reason
  `NetworkCore.newton_fit` reports, which `REMResult` keeps in its new `stop` field.
- **`EventSequence(::DynamicNetwork)` orders simultaneous onsets by sender, then
  receiver.** They came out in the order of an internal `Dict`, which mattered under
  `ties=:ordered`.
- `compute_decay_weight` was documented with swapped arguments; it is `(elapsed_time, decay)`.
- Fixture provenance citations corrected to survival 3.8.6.

### Performance

- **Time to first fit 3.7 s → 0.6 s** (first `show` 0.38 s → 0.002 s) via the
  precompile workload.
- **`fit_rem(::DataFrame)` no longer allocates per row**: 16.7 MB → 6.2 MB, 15.8 ms →
  7.6 ms for a 2000-event, 20-control fit; strata validation is single-pass.
- **`generate_observations` streams in O(events)**, independent of the actor universe
  (column-major design buffer, allocation-free `n_dyads`).
- **`update!` allocates 0 bytes on a warmed state** and is O(1) per event with lazy
  decay; neighbour sets are maintained incrementally (O(degree) queries).
- **Every exported statistic allocates 0 bytes**, including the triadic family.
- **Allocation-free conditional-logit kernel** with a single-pass strata index
  (deterministic summation order); BLAS rank-1 updates and tuple-backed `StatisticSet`
  give statically dispatched loops.

### Known limitations

Kept in sync with the README's "Not implemented" section; each is refused with an
`ArgumentError` where there is an entry point for it.

- No interval (waiting-time) likelihood or baseline rate: use Revel.jl
  (`fit_revel(...; model=:timing)`).
- `ties=:efron` with sampled controls (biased) and `ties=:batch` are refused.
- No dyad × event-type risk set (type-specific baselines): types enter through
  `OfType` statistics and `cases_of=`.
- A time-varying covariate is a function (`TimeVaryingCovariate`), not a spell
  table; Revel.jl reads change tables.
- relevent's normalised degree, `FrPSnd`/`FrRecSnd`, recency ranks and participation
  shifts, and remstats' `scaling`, are Revel.jl's.
- No DyNAM rate model or DyNAM-i.
- Under `directed=false` a directed statistic reads the pair in ID order (documented,
  not refused).
- No `se=:bootstrap` (by design) and no `missing=` policy.

## [0.1.0] - 2026-02-09

Initial release: relational event sequences, decaying network state,
event-history statistics, and stratified case-control (conditional logit)
estimation.
