# Changelog

All notable changes to REM.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: types are renamed to
compose cleanly with ERGM.jl and Graphs.jl, decay becomes lazy (O(1) event
updates), control sampling is corrected, and results adopt the ecosystem-wide
StatsAPI/presentation conventions.

### Breaking

- Singular or numerically unidentified observed information now gives
  `converged=false` as well as NaN uncertainty through the shared
  `Networks.newton_fit` rank guard. REM retains the named `singular` diagnostics.
  Separation remains distinct: an unbounded coefficient can satisfy the
  objective stopping rule, so callers must also inspect `separated` and
  `Networks.approximations`. The exactly confounded WTC regression and
  documentation now reflect this contract.

- **`NodeMatch` renamed to `AttributeMatch`** (it collided with
  `ERGM.NodeMatch`, breaking `using ERGM, REM`). The old name is gone — no
  alias. *Migration:* replace `NodeMatch(` with `AttributeMatch(`.
- **`NetworkState` renamed to `EventNetworkState`** (it collided with
  Siena's state type). The old name is gone — no alias; all statistic
  `compute` signatures use the new name. *Migration:* replace
  `NetworkState` with `EventNetworkState` (including the parametric
  `EventNetworkState{T}` form).
- **`has_edge` is now a method of the shared `Graphs.has_edge` generic**
  (re-exported by Networks.jl) rather than a REM-local function of the same
  name, which is what made it collide. Asking "is there a tie from i to j" of
  an accumulated event network is the same question Graphs asks of a graph, so
  REM adds a method for `EventNetworkState` instead of a rival function — and
  the name is therefore safe to export again (`REM.has_edge ===
  Networks.has_edge`). *Migration:* none; both `has_edge(state, s, r)` and
  `REM.has_edge(state, s, r)` work, and `using Graphs, REM` no longer clashes.
- **`NodeMix` renamed to `ActorMix`** (it collided with `ERGM.NodeMix`, a
  different thing in a different domain — a cross-sectional mixing-matrix
  term). Unlike the renames above this one keeps a **deprecated,
  non-exported** alias: `REM.NodeMix` still constructs an `ActorMix` and
  warns. It is not exported, because exporting it would recreate the
  collision. *Migration:* replace `NodeMix(` with `ActorMix(`.
- **`compute`, `name` and `compute_all` are no longer REM's own generics.**
  They are now the shared generics from Networks.jl, which REM extends with
  methods for its statistic types (`REM.compute === ERGM.compute ===
  Networks.compute`). Loading ERGM and REM together used to leave the
  unqualified verbs *undefined* (Julia's rule for conflicting exports), which
  broke the core statnet workflow of modelling cross-sections with ERGM and
  dynamics with REM in one session. *Migration:* code that extends the
  protocol must `import Networks: compute, name` (or `import REM: compute,
  name`, which resolves to the same bindings) instead of defining its own.
- **`generate_observations` samples controls without replacement.**
  Previously the rejection loop could draw duplicate control dyads and
  silently under-fill; now controls are distinct, and requesting more than
  the risk set warns and uses the full risk set. Estimates change versus
  0.1.0. *Migration:* expect exactly `min(n_controls, available)` distinct
  controls per event.
- **The sampler seed no longer reseeds the global RNG** (a local `Xoshiro`
  is used). *Migration:* seed the global RNG explicitly if you relied on
  that side effect.
- **`StatisticSet` is tuple-backed** (`StatisticSet{T<:Tuple}`); it is still
  constructible from a vector, but code annotating the old concrete type or
  reading `.statistics` as a `Vector` breaks. *Migration:* treat it as an
  iterable, or pass plain vectors to `fit_rem` (auto-converted).
- **`se=:bootstrap` is gone; the draw-to-draw spread of a sampled fit is
  `control_draw_cov`, a diagnostic** (2026-09 round 2). The first round's
  `se=:bootstrap` redrew the controls, refitted, and reported `W̄ + (1 +
  1/B)·B_between` — the mean of the per-draw inverse-information covariances
  plus the between-draw covariance of the refits — on the claim that the
  inverse-Hessian standard errors of a sampled fit "do not include the variance
  the risk-set sampling induces, so they are understated". That claim was
  wrong. Under nested case-control sampling the inverse observed information
  of the *sampled* partial likelihood is already a consistent estimator of the
  estimator's variance (Goldstein & Langholz 1992; Borgan, Goldstein & Langholz
  1995): the information lost by sampling fewer controls is what makes it
  larger than the full-risk-set one, and there is nothing left to add. The new
  "Sampled-risk-set standard errors are calibrated" testset pins it by
  simulation from the package's own correctly specified generator with events
  *and* controls redrawn on every replicate — 95 % Wald coverage for
  `se=:hessian` (measured 0.953 / 0.967 over 300 replicates); the withdrawn
  combination reached 97–99 %, i.e. it double-counted (and its `(1 + 1/B)`
  factor was Rubin's rule for the mean of `B` imputations, misapplied to a
  single draw). What redrawing the controls *does* measure is how far the
  **point estimate** of a *misspecified* model depends on the particular draw
  — a pseudo-true value that moves with `n_controls`, cured by more controls
  and not by a wider interval. That is a sensitivity diagnostic, and it is now
  offered as one: `control_draw_cov(seq, stats; n_controls, n_draws=20, rng,
  threaded)` returns `(cov, sd, replicates, mean, n_unconverged)` — the
  between-draw covariance of the refits, its square-rooted diagonal, the
  `n_draws × p` replicates, their mean — and is never combined with the
  Hessian nor reported by `stderror`. `se=:bootstrap` is refused on both
  `fit_rem` methods with the reason and the pointer; `n_boot` is gone;
  `REMResult.se_type` is `:hessian` or `:sandwich`; the sampled-fit caveat
  under `show` and in `approximations` now states the facts (the risk set was
  sampled; a misspecified model's estimate depends on the draw; refit with more
  controls or measure the spread) and the word "understated" appears nowhere.
  *Migration:* replace `fit_rem(seq, stats; se=:bootstrap, n_boot=B, rng=r)`
  with the default fit plus `control_draw_cov(seq, stats; n_draws=B, rng=r)`
  if you want the spread, or `se=:sandwich` if you want misspecification
  robustness.
- **`Observation` and `observations_to_dataframe` are removed.**
  `generate_observations` has built its `DataFrame` column-major since the
  first 2026-09 round; the row type was produced by nothing and consumed only
  by the unexported converter, which no test exercised — a second design
  builder that would drift. A hand-built design is a `DataFrame` with
  `is_event`, `stratum` and the statistic columns (`fit_rem(::DataFrame, …)`).
  *Migration:* build the `DataFrame` directly.
- **`RecencyStatistic(transform=:exp_decay)` returns `1.0` at zero elapsed
  time** when the dyad has a prior event (an event tied with the last one on
  the dyad under `ties=:ordered`, or a state read at its last event's time):
  `exp(−λ·0)` is well defined and means "just happened", whereas the old
  blanket `0.0` conflated it with "never happened" — the opposite of a recency
  effect (the statistic jumped from 0.9999999995 at Δ = 1e-9 to 0.0 at Δ = 0).
  `:inverse` and `:inverse_log` are genuinely singular at 0 and keep the
  `0.0`, now documented as a deliberate cap. *Migration:* only fits with
  same-dyad ties under `ties=:ordered` and the `:exp_decay` transform change.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.
- **A case must belong to its own risk set.** Previously an event whose
  sender or receiver lay outside `at_risk` was accepted while its controls
  were drawn from the (different) `at_risk` universe, silently changing the
  estimand; this now throws an `ArgumentError` before fitting, as does a risk
  set that admits no valid control. *Migration:* declare the actor universe
  (`EventSequence(events; actors=...)`) or widen `at_risk` so that it covers
  every case.
- **`EventSequence(::DynamicNetwork)` declares the actor universe from the
  network's vertex set**, so vertices with no edge spell remain in the risk
  set as isolates. *Migration:* pass `actors=` to restore the old
  participants-only universe.
- **`REMResult` gained fields** (`strata`, `risk_set_sizes`, `sampling_probs`;
  in the 2026-09 round `var_cov` and `iterations`; in its second round
  `singular`, `singular_suspects` and `separated` — see "Collinearity and
  separation are loud" under Added). The 9-, 12-, 13- and 14-argument
  positional `REMResult` constructors that padded the missing fields (an empty
  risk-set bookkeeping, `:hessian`/`:none`, a NaN covariance, `iterations ==
  0`) are **gone**: nothing in the ecosystem built a result by hand (Relevent.jl
  reads `REMResult`, it never constructs one), no test exercised them, and a
  result that reports `se_type = :hessian` while carrying no covariance is
  exactly the silent inconsistency `vcov` was added to rule out. *Migration:*
  a `REMResult` comes from `fit_rem`; a hand-built one names all nineteen
  fields.
- **`coeftable(fit)` returns a `Networks.CoefficientTable`, not a `DataFrame`**
  (panel 2026-09, item 15). It is the ecosystem's inspectable coefficient
  table — the vectors `show(fit)` prints (`names`, `estimates`, `std_errors`,
  `z_values`, `p_values`), rows by index or by name (`tbl["repetition"]`),
  `show` through `Networks.print_coeftable` — and what every model package's
  `coeftable` now returns, so a REM table and an ERGM table are the same
  type. *Migration:* read the fields directly, or rebuild the old frame:
  `tbl = coeftable(fit); DataFrame(statistic=tbl.names,
  coefficient=tbl.estimates, std_error=tbl.std_errors, z_value=tbl.z_values,
  p_value=tbl.p_values)`.
- **`EventNetworkState.event_history` is kept only when a statistic needs it.**
  `generate_observations`, `compute_statistics` and `fit_rem` build their state
  with `keep_history = any(needs_history, stats)`: the per-event log (O(events)
  memory retained by every state) is dropped for REM's own statistics, none of
  which reads it, and kept for any statistic that declares — or, by default,
  does not deny — reading it (Relevent.jl's `PShift`, `Momentum`,
  `PriorInteraction`, ...; the trait defaults to `true` for a statistic REM does
  not know, so nothing is dropped silently). The field is now documented as the
  public, read-only log. A hand-built `EventNetworkState` still keeps it by
  default (`keep_history=false` opts out). *Migration:* none for the
  constructors; a statistic outside REM that reads the log keeps working, and
  can say so explicitly with `REM.needs_history(::MyStat) = true`.
- **The far-tail p-value floor is `floatmin(Float64)`, not a subnormal.**
  z- and p-values now come from the ONE shared `Networks.z_pvalues` (item 13):
  `erfc`-based, `NaN` where the standard error is not positive, and floored at
  `floatmin` for a finite statistic (`≈ 2.2e-308`, never `0.0`). The old
  `2·ccdf(Normal(), |z|)` returned a subnormal there. *Migration:* none in
  practice (`p < 1e-300` either way); a test asserting `issubnormal(p)` should
  assert `p == floatmin(Float64)`.

- **The triadic and four-cycle statistics default to eventnet's weighted
  definition** (panel 2026-09, WP2). `TransitiveClosure`, `CyclicClosure`,
  `SharedSender`, `SharedReceiver`, `CommonNeighbors` and `FourCycle` now
  default to `weighted=true, aggregation=:min`: for a candidate `s → r` the
  (decayed, event-weighted) counts of the two dyads of each two-path through a
  third party `k ∉ {s, r}` are combined by the aggregation — eventnet's `min`
  by default; `:max`, `:sum`, `:product` are the other three — and the values
  of the parallel two-paths are added (`FourCycle` aggregates its three
  weights the same way). `CommonNeighbors` gained the weighted form on the
  **undirected** counts. The 0.1 definition — the *number of distinct* third
  parties, read off the adjacency, which never decays — is `weighted=false`.
  Coefficients on these statistics change wherever a dyad has repeated events.
  *Migration:* `TransitiveClosure(weighted=false)` (etc.) restores the count;
  the `rem_clogit` golden test now passes it explicitly, and the new
  `rem_eventnet.toml` fixture pins the weighted forms against R.
- **`RecencyStatistic(transform=:log)` is refused; the transform is
  `:inverse_log`.** What `:log` computed was `1 / log(1 + Δ)` — a decreasing
  recency transform, not a log transform — so it is now named for what it is
  (same formula, statistic name `recency_inverse_log`); `:log` throws an
  `ArgumentError` pointing at `:inverse_log`, and an unknown transform is
  refused at construction rather than at the first `compute`. *Migration:*
  replace `transform=:log` with `transform=:inverse_log`.
- **`NodeAttribute(name, values)` — constructed without a default — throws when
  a statistic reads an actor it has no value for** (`ArgumentError` naming the
  attribute and the actor, and how to opt into a fill). Filling a missing
  covariate silently with a default is the zero-fills-as-data trap;
  `NodeAttribute(name, values, default)` remains the explicit opt-in and
  behaves as before. `has_default(attr)` reports which kind it is.
  *Migration:* code that relied on the old three-argument form is unchanged;
  new code that wants a fill must say so.
- **`get_out_neighbors`/`get_in_neighbors` return `AbstractSet{Int}`**: an actor
  with no neighbours gets a shared **immutable** empty set (`copy` it for a
  `Set{Int}`) instead of a shared *mutable* `Set{Int}()` that one stray `push!`
  could corrupt for every caller. *Migration:* none unless code mutated the
  returned set (which the docstring forbade) or dispatched on `Set{Int}`.
- `RecencyStatistic(directed=false)` **reads the last event in either
  direction** (the later of the two dyads' last-event times). Before, it looked
  up the `(min, max)` dyad only — one direction — so the "undirected" recency
  of `(2, 1)` after a `2 → 1` event was `0.0` whenever no `1 → 2` event
  existed. Values change for dyads whose most recent event was the
  `(max, min)` direction.
- The unexported, unused `events_before`/`events_in_window` helpers are gone
  (a window is a property of the network state now, see Added).
- **`fit_rem(seq, stats)` without `seed` draws its controls from the `rng`
  keyword** (default `Random.default_rng()`), no longer from the global RNG
  behind the caller's back (see Changed, "All randomness flows through `rng`").
  A call that relied on `Random.seed!` for reproducibility keeps working (the
  default `rng` *is* the global one); a call that passed `rng=` and expected it
  to be ignored gets different — now reproducible — controls. *Migration:*
  pass `seed=` or `rng=`.
- **`show` of `EventSequence`, `EventNetworkState`, `ActorSet`, `RiskSet`,
  `CaseControlSampler` and `StatisticSet` is one informative line** —
  `EventSequence{Float64}(481 events, 37 actors, declared universe)`,
  `EventNetworkState{Float64}(37 actors, 481 events absorbed, no memory decay,
  current_time = 481.0)`, … — instead of the default dump of the fields (an
  `EventSequence` printed its whole event vector). `show(::REMResult)` is
  unchanged: the R-style block every tutorial relies on. The header's
  standard-error line now reads `inverse Hessian (full risk set)` when nothing
  was sampled (`(one control draw)` otherwise). *Migration:* code parsing the
  old `repr` of those types (unlikely) must read the fields.
- `EventNetworkState` gained an `n_events::Int` field — the events absorbed
  since the last `reset!`, kept whether or not the event log is. The keyword
  constructor is unchanged.

### Added

- **The full StatsAPI surface on `REMResult`** (panel 2026-09, item 15):
  `vcov`, `confint(fit; level=0.95)` (Wald, normal reference), `loglikelihood`
  (the `log_likelihood` field, whose name is unchanged), `nobs` (the number of
  **events** — one stratum each, `survival::clogit`'s `nevent`), `dof`
  (`length(coef)`), `aic` (`−2ℓ + 2k`) and `bic` (`−2ℓ + k·log(n_events)`), all
  methods of the `StatsAPI` generics alongside `coef`, `stderror` and
  `coeftable`, all exported. `Networks.check_statsapi(fit; strict=true)` passes
  for `:hessian`, `:sandwich` and DataFrame-method fits. The covariance on the
  result **matches `se_method(fit)`** — inverse observed information or
  Godambe sandwich `H⁻¹BH⁻¹` — so `stderror(fit) == sqrt.(diag(vcov(fit)))`
  under both (the sandwich path now keeps the full matrix instead of only its
  diagonal). Both golden testsets also compare `sqrt.(diag(vcov))` to R.
- **Collinearity and separation are loud** (2026-09 round 2, criterion 2).
  Both come back "converged" from Newton–Raphson with numbers a reader would
  paste into a report: on the bundled WTC calls `fit_rem(seq, [NodeSum(icr),
  SenderAttribute(icr), ReceiverAttribute(icr)])` printed three `NaN` standard
  errors under `Converged: true` with `is_exact == true` and an empty
  `approximations`, and `fit_rem(seq, [Repetition(), NodeProduct(icr)])`
  printed `product_icr −20.1 (17931.5) p = 0.999` as an ordinary row. Now a
  **singular** observed information (collinear statistics, a statistic
  constant within every stratum) sets `fit.singular`, names the statistics
  that load on its null direction in `fit.singular_suspects` (the eigenvector
  of the zero eigenvalue: `sum_icr`, `sender_icr`, `receiver_icr` there), warns
  with them, adds an `approximations` entry, makes `is_exact` `false` and
  prints a caveat under the table; and a **separated** coefficient is detected
  as `survival::coxph` detects it ("coefficient may be infinite"): at a
  converged solution the Newton increment `H⁻¹g` is ~1e-15 on a healthy
  coordinate and O(1) on a separated one (measured −1.0000 on `product_icr`),
  so a coefficient with `|δ| > 0.01·max(1, |β|)` is listed in `fit.separated`,
  warned about, recorded, made inexact and captioned. **The rule is
  scale-free** (round 3): it is applied to the standardised coefficient and
  increment, `β_j·s_j` and `δ_j·s_j` with `s_j` the column's standard
  deviation, which rescale together with the covariate exactly as
  `survival::coxph`'s test does. On the raw scale the same separated
  `NodeProduct` with the ICR attribute coded 0/1000 came back as β = −2.4e-5,
  SE 0.13, `separated = []`, `is_exact == true` — a z ≈ 0 row a reader would
  take for "no effect" when the maximum is at −∞; it is now flagged at ×1,
  ×1000 and ×0.001 alike (pinned), and the "rescale a covariate measured in
  thousands" limitation is withdrawn. The 3-event toy sequences the metadata
  and actor-universe testsets used turned out to be separated themselves
  (coefficients −42 / +25 with standard errors of 7e4 — they now use six
  events). The estimation guide's "Common Causes" table says what the package
  does for each.
- **An unconverged fit is loud.** When `Networks.newton_fit` exhausts `maxiter`
  (or no Newton step improves the likelihood) `fit_rem` **warns**, naming
  `maxiter`, the final gradient norm and `tol`; the result carries
  `converged == false` and the new `iterations` field; `Networks.is_exact(fit)`
  is `false`; `Networks.approximations(fit)` gains "the optimizer did not
  converge in N iterations — estimates and standard errors are not
  maximum-partial-likelihood values"; and `show` prints `Converged: false (N
  iterations; see approximations)` plus a caveat under the table. A converged
  fit prints `Converged: true (N iterations)`. Unconverged `control_draw_cov`
  refits are counted and warned about once.
- **`aggregation=` on the triadic and four-cycle statistics** (`:min`, `:max`,
  `:sum`, `:product`): eventnet's four combining functions for the weights of a
  two-path (three-path for `FourCycle`), applied only to two-paths whose both
  legs exist (a missing leg contributes nothing, which matters for `:sum` and
  `:max`); validated at construction.
- **Sliding-window memory: `window=` on `EventNetworkState`,
  `generate_observations`, `compute_statistics` and `fit_rem`** (WP2). An event
  older than `current_time − window` stops counting in the dyad counts, the
  undirected counts, the degrees *and* the adjacency sets (a dyad whose events
  have all expired has no edge and closes no triangle) — the fixed-memory
  alternative to eventnet's halflife decay, which is the only memory model
  eventnet itself offers. Implemented with a FIFO of the live events and an
  expiry cursor advanced when the clock moves (amortized O(1) per event; the
  FIFO holds O(live events)); a key whose events have all expired is deleted,
  so windowed counts return to an exact `0.0`, not a floating-point residue.
  `window` is in clock units for numeric time and in **seconds** for
  `Date`/`DateTime`; an event exactly `window` old still counts; `Inf` (or
  `nothing`, the default) is no window and produces the identical design row
  for row. **Mutually exclusive with `decay > 0`** (`ArgumentError`). The
  last-event times (`RecencyStatistic`) and the event log are not windowed.
  `control_draw_cov` threads the window through its redraws. **On a
  `Date`/`DateTime` clock `window=Day(2)` (any `Dates.Period` or
  `CompoundPeriod`) is accepted on every entry point and converted to the
  seconds the clock counts in** (round 2), and so is
  `halflife_to_decay(Day(7))`; before, the calendar idiom failed with a
  `TypeError`/`MethodError` that never mentioned seconds, while `window=2.0`
  silently meant two seconds. **On a numeric clock a `Period` window is
  refused** (round 3): `EventNetworkState(seq; window=Day(2))` on a `Float64`
  event-index clock used to convert to 172 800 clock units — effectively no
  window, and `fit_rem(seq, stats; window=Day(2))` returned the no-window fit
  with no warning; the `EventNetworkState{T}` constructor, which every entry
  point routes through, now throws an `ArgumentError` unless `T <:
  Dates.TimeType`, telling the user to pass a number in clock units or to
  build the sequence on `Date`/`DateTime` timestamps.
- **A second `survival::clogit` fixture, `test/fixtures/rem_eventnet.toml`**
  (`test/fixtures/r/rem_eventnet.R`), on the `rem_clogit` sequence: the
  min-weighted triadic family plus undirected repetition, `TransitiveClosure`
  under each of the four aggregations, halflife-decayed repetition / activity /
  closure, and windowed repetition / reciprocity / closure — every column
  rebuilt from the raw edgelist in plain R, full risk set on both sides,
  tolerance 1e-8 justified as `rem_clogit.toml`'s (same exact likelihood, no
  Monte Carlo).
- **A `relevent::rem.dyad` fixture on the bundled WTC police calls,
  `test/fixtures/rem_relevent_wtc.toml`** (`test/fixtures/r/rem_relevent_wtc.R`,
  relevent 1.2.1). R reads the same two TSV files
  `Networks.load_dataset(:wtc_police_calls)` reads, so the 481 events and the
  37-actor universe are provably identical; with `n_controls = 37·36 − 1` the
  full risk set is enumerated and REM's conditional logit IS `rem.dyad`'s
  ordinal likelihood. Three models: relevent's `CovInt` (`x_i + x_j`, the
  tutorial's `wtcfit1` — `NodeSum(icr)`, coefficient 2.104), `CovSnd + CovRec`
  (`SenderAttribute`, `ReceiverAttribute`) and `CovSnd + CovRec + CovEvent(D)`
  with a fixed dyadic matrix (`DyadCovariate`). Coefficients, standard errors
  (`sqrt.(diag(vcov))` against R's Hessian-based ones) and `loglikelihood`
  agree to 1e-6 (optimizer termination slack: BFGS vs Newton; log-likelihoods
  to 1e-8). The ICR-to-ICR product (`NodeProduct`) is *not* fitted: no ICR
  actor ever calls another in these data, so that coefficient separates.
  Relevent's `CovInt` is the sender-plus-receiver covariate, collinear with
  `CovSnd + CovRec` — the three together are a singular Hessian in R.
- **Hand-computed testsets for every exported statistic** — the triadic family
  × aggregation × `weighted` × {no decay, halflife = 1 where the count stays
  `1.0` while the weighted value is `0.5`}, `FourCycle` on all five cycle types
  with a 4-actor example, `GeometricWeightedTriads`/`GeometricWeightedFourCycles`
  at `e^α(1 − (1 − e^{−α})^n)` for `n = 0..3`, `InertiaStatistic`,
  `RecencyStatistic` (all three transforms, undirected), `DyadCovariate`, every
  degree statistic (all `degree_type`s, `absolute`), `LogDegree`, `ActorMix`,
  `NodeSum`, `NodeProduct`, `SenderAttribute`, `ReceiverAttribute`,
  `SenderCategorical`, `ReceiverCategorical` — with event weights ≠ 1 and
  `Int`/`Date`/`DateTime` clocks per family, the expected numbers written out.
- **Allocation pins for every statistic** (item 26): `@allocated compute(stat,
  state, i, j) == 0` for every exported statistic on a warmed 30-actor state
  under no memory, decay and a window, and for `compute_all!` over all of them.
- `has_default(attr::NodeAttribute)` (exported).
- **Actionable errors for the common mistakes** (criterion 5): a statistic
  reading an actor a default-less `NodeAttribute` lacks names the attribute and
  the actor; `ActorMix`/`SenderCategorical`/`ReceiverCategorical` with a value
  of the wrong type name the attribute's value type (an exactly convertible
  value, an `Int` for a `Float64` attribute, is coerced); `NodeDifference`/
  `NodeSum`/`NodeProduct`/`SenderAttribute`/`ReceiverAttribute` on a
  categorical attribute point at `AttributeMatch`/`ActorMix`/`*Categorical`;
  `fit_rem(obs, ["repetiton"])` lists the statistic columns the frame has;
  `fit_rem`/`generate_observations`/`compute_statistics`/`EventNetworkState`
  given a `Vector{Event}` point at `EventSequence(events; actors=ActorSet(ids))`;
  `decay > 0` with a `window` says the two memory models cannot be combined.
  Round 3 (panel item 31 for REM): `fit_rem(seq, Repetition())` — one
  statistic, not a vector, relevent's `effects="CovInt"` habit — is accepted
  as a one-element model, and `fit_rem(stats, seq)` / `control_draw_cov(stats,
  seq)` (the arguments in the other order) throw an `ArgumentError` spelling
  out the order, instead of a bare `MethodError` with a "Closest candidates"
  dump; a hand-built design whose `is_event` column is 0/1 integers is
  accepted (`fit_rem(DataFrame(is_event=[1,0,1,0], stratum=[1,1,2,2], x=…),
  ["x"])` used to fail with an internal `MethodError` on `_stratum_counts`),
  and any other `is_event` or a non-integer `stratum` is refused by name.
- **A runnable example in the docstring of every export** (criterion 5) — the
  statistics, the state and its accessors, the types (`Event`, `EventSequence`,
  `ActorSet`, `NodeAttribute`, `RiskSet`), data loading (`load_events`,
  `load_events!`), the estimation surface (`CaseControlSampler`,
  `generate_observations`, `compute_statistics`, `fit_rem`, `REMResult`,
  `sampling_probs`, `risk_set_sizes`, every StatsAPI verb) and the decay
  utilities — pinned by the "Every exported docstring carries a runnable
  example" testset, which walks `names(REM)` and **executes** each block in a
  fresh module (32 of 88 exports had no example before).
- **The docs teach on the bundled WTC police-calls event stream**
  (`Networks.load_dataset(:wtc_police_calls)`, relevent's running example; panel
  item 22): the README "Basic Example", the manual's quick start, the
  getting-started tutorial and the estimation and decay guides declare
  `ActorSet(1:37)`, build the ICR covariate as a `NodeAttribute`, fit the full
  risk set (`n_controls = 37·36 − 1`, exact, reproducing relevent's `wtcfit1` at
  2.1045) and then case-control samples with `se=:sandwich` and
  `control_draw_cov`. The estimation guide tabulates how a sampled fit of a
  misspecified model moves toward the full-risk-set fit as `n_controls` grows
  — as **mean ± sd over ten control draws** (round 2; the first round showed
  one seed, whose −0.72 at 20 controls was 1.8 draw-sds out): repetition
  −0.50 ± 0.07 at 20 controls, −0.41 ± 0.04 at 100, −0.33 ± 0.02 at 400,
  −0.34 in full, against per-draw standard errors of 0.08 / 0.05 / 0.04 /
  0.03 — the mean moves (the pseudo-true shift) and the draws scatter by about
  a standard error at 20 controls, neither of which is an SE deficiency — and
  recommends enumerating the risk set whenever affordable. Every printed
  `Output:` block is pasted from a real run. The toy 4-actor examples that
  used to print `Converged: false` and NaN standard errors are gone.
- **Actionable errors for an empty statistics list and a self-loop event**
  (round 2). `fit_rem(seq, AbstractStatistic[])` used to surface the internal
  `tie_weights has 36 entries for 0 observation rows`; every fit entry point
  now says "fit_rem needs at least one statistic (e.g. `[Repetition(),
  Reciprocity()]`)". A self-loop event under the default
  `exclude_self_loops=true` used to fail with the not-a-member-of-its-risk-set
  message whose two suggested fixes (declare the universe, fix `at_risk`) do
  not apply; it now names the loop and the keyword ("Event 1 is a self-loop (1
  → 1) … drop self-loop events … or pass `exclude_self_loops=false`"), and the
  events guide says a self-loop event is refused at fit time unless the loops
  are admitted (it claimed the risk set handled it silently).
- **`NodeAttribute` prints one line** (round 2): `NodeAttribute{Float64}(:icr,
  37 actors, no default)` / `default = 0.0`, and `_NoDefault` prints `no
  default`, so `NodeSum(icr)` and a `stats` vector no longer dump a 37-entry
  `Dict` and a private sentinel type into the REPL. Pinned in the `show`
  testset.
- A "Case-control sampling is consistent for a correctly specified model"
  testset: on the simulated 8-actor sequence, eight independent 20-control
  draws all lie within 2.5 standard errors of the truth and their mean within
  0.12 of the full-risk-set estimate, and 5-control draws are further away on
  average.
- A "show is compact and informative" testset (`sprint(show, x)` for every
  type above and the header lines of the fit).
- `generate_observations(...; rng=)`: the control draw's RNG when the sampler
  has no `seed` (see Changed).
- `eachindex(seq)` and `firstindex(seq)` on `EventSequence` (alongside the
  existing `getindex`/`lastindex`), so `[spec(k) for k in eachindex(seq)]`
  builds per-event risk sets as the guide shows; and `eltype`, so
  `collect(seq)` is a `Vector{Event{T}}` that `EventSequence(events;
  actors=...)` accepts (the way to declare a universe on a sequence
  `load_events` inferred).
- `needs_history(stat)` (exported trait) and `EventNetworkState(seq;
  keep_history=)`: whether a statistic reads the state's event log, and whether
  a state keeps one (see Breaking).
- A "No export collides with Base" testset, an "Allocation-free clogit
  kernel" testset (300 vs 3000 strata), a "Streams in O(events)" testset
  (allocation pins across 100 → 2000 actors and 2000 → 4000 events) and an
  "rng contract" testset.

- **Tied event times are now a policy, not a warning: `ties=:error|:ordered|:breslow|:efron`**
  (issue #2, review finding 12). The partial likelihood is a **Cox partial
  likelihood**, so tied timestamps are the classical Cox tie problem — and the
  tie does something specific here: the statistics are read off the network
  state *before* each event, so ordering two simultaneous events lets the one
  placed first enter the **statistics** of the one placed second. That is
  invented information, so `fit_rem`/`generate_observations` now **default to
  `:error`** and name the tie (which events, which timestamp, how many in all)
  instead of sorting it.
  - `:ordered` — the pre-0.2 behaviour (sequence order, no correction), now an
    explicit, recorded choice.
  - `:breslow` — the network state is frozen across the tie block (absorbed as a
    whole afterwards); each tied event is a stratum, all share one denominator.
  - `:efron` — as `:breslow`, plus the `1 − (j−1)/d` denominator weights on the
    tied cases, carried in a new **`tie_weight`** column of the observations
    frame. The better approximation, and `survival::coxph`'s own default.
    Requires the tied cases to be distinct dyads (a dyad competing with itself
    has no fractional weight — it throws rather than invent one).
  - `:batch` — refused, with a pointer: with the risk set held fixed, a
    "simultaneous batch" in an ordinal likelihood *is* Breslow.

  The vocabulary is `Networks.TIE_POLICIES`, defined once in Networks.jl and
  shared with Relevent.jl, and `Networks.check_tie_policy` makes a policy a model
  cannot honour **fail loudly rather than no-op**. On tie-free data all four
  policies produce the identical design, row for row.
- **`tie_method(fit)` now reports what ACTUALLY happened** (shared result-metadata
  protocol): `:none` when the data had no ties — it no longer reports the name of
  a correction that corrected nothing — and `:ordered`/`:breslow`/`:efron` when
  one bit. `:error` can never appear (under it a tie throws). `approximations(fit)`
  carries the matching caveat, `is_exact(fit)` now requires *both* a full risk set
  *and* untied data, and `show` prints the policy only when it bit. The policy
  travels with the design as `:note`-style DataFrame metadata, so
  `fit_rem(::DataFrame, ...)` reports the truth without being told twice — and a
  frame marked `:efron` that has lost its `tie_weight` column is refused rather
  than fitted unweighted while claiming Efron.
- **A golden fixture for the tie corrections against `survival::coxph`**
  (`test/fixtures/rem_ties.toml`, regenerated by `test/fixtures/r/rem_ties.R`):
  a sequence observed on a coarse clock (25 of 53 timestamps tied, up to 4 deep),
  fitted in R with `ties="breslow"` and `ties="efron"` on a counting-process
  design rebuilt from the raw edgelist in plain R. Julia agrees to **< 1e-11** on
  coefficients, standard errors and log-likelihood — which is what turns "we
  implemented Breslow" into "we implemented Breslow *and it is what R computes*".

- **A real R golden fixture: coefficients against `survival::clogit`** (issue
  #8). `test/fixtures/rem_clogit.toml` freezes an actual R run (survival
  3.8.6, R 4.6.1 — the version the `[provenance]` block records) and
  `test/fixtures/r/rem_clogit.R` regenerates it. The
  design matrix is rebuilt from the raw edgelist *in plain R*, so this checks
  the STATISTICS (`Repetition`, `Reciprocity`, `SenderActivity`,
  `ReceiverPopularity`, `TransitiveClosure`) as well as the estimator, and the
  risk set is enumerated in full on both sides so there are no sampled
  controls to reconcile across two RNGs. Tolerance **1e-8** — both sides
  maximize the same exact conditional-logit likelihood by Newton-Raphson, so
  nothing is allowed to differ but floating-point summation order. **Observed
  agreement: < 1e-13** on every coefficient and standard error. **Round 2 adds
  `robust_std_errors`** to the same fixture: `clogit(... + strata(stratum) +
  cluster(stratum), method = "breslow")` — the stratum-clustered robust
  variance, which is exactly REM's `se=:sandwich` (one event per stratum, so
  Breslow is the exact likelihood there; without `cluster(stratum)` R's robust
  variance is the per-row dfbeta sandwich, 1.2e-3 away) — pinned at 1e-8 with
  observed agreement 4e-16, so the sandwich estimator now has an R
  counterpart in the fixtures too. The `_clogit_sandwich_cov` docstring, which
  said `coxph(..., robust = TRUE)` without the cluster, names the R call
  exactly.
- **Robust standard errors: `fit_rem(seq, stats; se=:hessian|:sandwich)`**
  (issue REM#2).
  - `se=:hessian` (default) — the inverse observed information of the partial
    likelihood on the risk set that was used. With a sampled risk set it is the
    information of the sampled likelihood, a consistent variance estimator
    under nested case-control sampling (see Breaking, `se=:bootstrap`).
  - `se=:sandwich` — the **event-clustered Godambe sandwich** `H⁻¹ B H⁻¹`, with
    the meat `B = Σ_e u_e u_eᵀ` the outer product of the per-event score
    contributions (each event is exactly one stratum, so the event is the
    clustering unit). Robust to misspecification of the within-stratum
    conditional model. Available on both the `EventSequence` and the
    observations-`DataFrame` methods, and pinned against
    `survival::clogit(... + cluster(stratum))`.
  The point estimates are unchanged under both — only the covariance differs.
  The first round's `se=:bootstrap` was withdrawn in round 2 in favour of the
  `control_draw_cov` diagnostic (see Breaking).
- **`sampling_probs(fit)` and `risk_set_sizes(fit)`** (exported): the control
  inclusion probabilities each stratum conditioned on, and its risk-set size.
  These are part of the estimand, not an implementation detail — they are what
  `is_exact` reads and what the missing variance component is *about*.
- `se_method(fit)` now reports what was actually used
  (`:hessian`/`:sandwich`), read off the new `REMResult.se_type` field; `show`
  names the estimator, and under the table prints a *note* when the risk set
  was sampled (what that approximates, and that a misspecified model's
  estimate depends on the draw) and a *warning* when the fit is wrong rather
  than approximate (unconverged, singular, separated). With the full risk set
  and a healthy fit nothing is printed at all.

- **Conversion invariants for `EventSequence(::DynamicNetwork)`**: the adapter
  takes the ecosystem `missing=:error`/`:face` policy and a `report=true`
  keyword returning `(seq, ::Networks.ConversionReport)` that names the fields
  an event sequence cannot carry (spell termini, vertex spells / time-varying
  risk sets, attributes, the observation window). See the ecosystem table in
  Networks.jl `docs/src/guide/conversion_invariants.md`.

- **Explicit actor universe**: `EventSequence(events; actors=ActorSet(...))`
  (also `Set{Int}`, `Vector{Int}`, or any integer range) declares the actor
  universe including isolates and noncontiguous IDs, instead of inferring it
  from observed event endpoints. Endpoints outside a declared universe throw.
  Fitting against an inferred universe now warns: the risk set determines the
  estimand, and observed participants are not the eligible population.
- **Explicit risk sets in `fit_rem`** via `at_risk` (alias `riskset`), forwarded
  to `generate_observations`. Supported forms: a static actor universe
  (`ActorSet`/`Set{Int}`/`Vector{Int}`), a static `RiskSet` (asymmetric
  sender/receiver sets), a vector of per-event risk sets (time-varying
  membership), or a callback `(event_index, state) -> RiskSet` evaluated against
  the live network state.
- **Risk-set bookkeeping**: `generate_observations` records `risk_set_size` and
  `sampling_prob` (the probability with which each non-case dyad entered the
  sample as a control) per stratum; `REMResult` exposes `strata`,
  `risk_set_sizes` and `sampling_probs`, and `show` reports them.
- `fit_rem(observations, ...)` rejects strata with no controls (previously
  accepted and silently uninformative).

- NetworkDynamic bridge extension (`REMNetworkDynamicExt`, loads with
  `using NetworkDynamic`): `EventSequence(::DynamicNetwork; eventtype=:onset,
  weight=..., include_onset_censored=...)` converts edge-activation spells
  into a relational event sequence.
- StatsAPI integration: `coef`, `stderror`, and `coeftable` extend the
  StatsAPI generics instead of package-local functions.
- `compute_all!` in-place statistic evaluation for sampling loops.
- Input validation in `fit_rem` (missing statistic/`is_event`/`stratum`
  columns, strata without exactly one case throw `ArgumentError`).

- **Benchmark suite, complexity assertions and allocation-regression gates**
  (panel 2026-09, item 7 / criterion 4). `benchmark/benchmarks.jl` gained an
  `observations` group (`generate_observations` at 100 and 2000 actors, 2000
  and 4000 events) and a `fit` group (`fit_rem` on the 2000- and 4000-event
  streams) beside the state-update sweeps, and now asserts the streaming
  complexity after the timings, printing one `SCALING` line per bound in the
  format `tools/run_benchmarks.jl` consumes and exiting non-zero on a
  violation: events 2000 → 4000 must cost ≤ 2.5× the time and ≤ 2.2× the
  bytes (both for the design build and for the fit), actors 100 → 2000 ≤ 1.5×
  the bytes (measured 2.05 / 2.14 / 1.17 and 2.07 / 2.10). The new
  `benchmark/regression_tests.jl` is the standalone gate holding the
  `@allocated` pins the round established — the 0-byte `_clogit_derivatives!`
  kernel with a strata-independent closure/sandwich budget, 0 B for every
  exported statistic (enumerated from `names(REM)`, so a new statistic cannot
  ship unpinned) on plain, decaying and windowed states plus `compute_all!`,
  and O(events) `generate_observations` with an allocation-free `n_dyads`.
  `julia tools/run_benchmarks.jl REM` (site repo) reports PASS.
- **PrecompileTools workload** (panel 2026-09, item 18): `src/REM.jl` ends
  with a `@compile_workload` that runs the whole pipeline once at precompile
  time on a 6-actor / 20-event toy stream carrying one tie — `EventSequence`
  construction, `generate_observations`, `fit_rem` with `se=:hessian` and
  `se=:sandwich` under `ties=:efron`, the DataFrame method, `compute_statistics`,
  all ten StatsAPI verbs and `show`. The draw is a local `Xoshiro`; nothing
  leaks into user-visible state. Numbers under "Performance".
- **CI exercises the thread-independence contract and the regression gates**
  (items 7, 29): the ubuntu / Julia 1.12 cell runs with `JULIA_NUM_THREADS=4`
  and the "rng contract" testset asserts that `control_draw_cov`'s threaded
  loop and its serial loop (`threaded=false`) give bit-identical covariances —
  a real check at 4 threads, a tautology at 1, which is why that cell exists
  (round 2: the first round compared two threaded runs in one process, which
  proved reproducibility at one thread count, not independence of it) — and,
  after the test suite, instantiates `benchmark/` and runs
  `benchmark/regression_tests.jl`.

### Changed

- **`benchmark/Project.toml` sources Networks at `../../Networks.jl`** (panel
  2026-09, item 7). It said `../Networks.jl` — one `../` short, resolved from
  `benchmark/` — and survived only on a gitignored `Manifest.toml` that still
  pinned the right path; on a fresh clone `julia --project=benchmark -e 'using
  Pkg; Pkg.instantiate()'` failed with *expected package Networks to exist at
  path …/REM.jl/Networks.jl*. The stale Manifest is deleted; the environment
  now instantiates from the sibling layout (and in CI, from the sibling clone).
- **CI clone lists are derived from `[sources]`** (item 29): `CI.yml` clones
  Networks, NetworkDynamic and ERGM — exactly the `[sources]` section of
  `Project.toml`, which now carries a comment tying the two together (ERGM is
  test-only, for the `using ERGM, REM` co-load test) — and
  `Documentation.yml` clones only Networks, the one sibling `docs/Project.toml`
  sources (it cloned NetworkDynamic and ERGM too, which the docs never load).
- **`apply_decay!` is removed** (round 3, dead-code policy). The unexported
  eager materialiser of the raw count tables had no caller in `src/`, `ext/`,
  `docs/` or `benchmark/` — only one test invoked it, to check that it did not
  change what the accessors read. Counts decay lazily on read and every
  accessor and statistic reads them that way; there is no other read path to
  materialise for. *Migration:* none for any public API (it was never
  exported); a script that called `REM.apply_decay!(state, t)` can drop the
  call — the accessors return the same numbers without it.
- `docs/make.jl` still deploys `"stable" => "dev"`: until 0.2.0 is tagged
  (the ecosystem-wide release step, panel item 21) there is no stable
  version to serve, so `/stable` deliberately shows the development docs. The
  release commit drops that line.
- **The estimation loop is `Networks.newton_fit`** (panel 2026-09, item 14).
  `_fit_stratified_clogit` no longer carries a Newton–Raphson of its own: it
  hands the shared optimizer a closure around REM's per-stratum softmax kernel
  and takes `converged`, `iterations` and the Cholesky-based inverse
  information (`NaN`, with a warning, when the Hessian at the solution is not
  negative definite — the old `-inv(hess)` inside a `try` returned finite
  nonsense or NaN silently) from it. The kernel itself stays in REM (it is a
  conditional-logit softmax, not a logistic likelihood). The likelihood is now
  accumulated with compensated (Neumaier) summation, so a rounding-level
  "decrease" can no longer stop the optimizer one step short; the Breslow golden
  fixture agrees with R to 3e-16 (was 9e-9 at the same tolerance).
- **All randomness flows through `rng`** (panel 2026-09, item 16 / criterion 2).
  `fit_rem(seq, stats; rng=Xoshiro(9))` without `seed` used to draw the
  controls from the **global** RNG (two such calls gave different
  coefficients); now `generate_observations` and `fit_rem` draw them from the
  caller's `rng`, with `seed` — when given — pinning the draw to a local
  `Xoshiro(seed)` and taking precedence over `rng` (every existing `seed=`
  result is unchanged). The `control_draw_cov` redraws draw their per-draw
  seeds from the same `rng` up front (thread-count-independent through
  `Networks.bootstrap_cov`).
- The `se=` keyword is validated by the ONE shared `Networks.check_se` (item
  28) — the message names the fitter, the vocabulary and the offender — with
  only the bespoke `se=:bootstrap` refusal (which explains why and points at
  `control_draw_cov`) kept in REM.
- **No `rem` alias of `fit_rem`**, and a test that no exported name of REM is
  exported by `Base`. The other model packages carry a statnet-style short verb
  (`ergm`, `stergm`); REM's would be `rem`, which is `Base.rem` — exporting a
  rival binding would make `rem(7, 2)` an ambiguity error in every `using REM`
  session. relevent's `rem`/`rem.dyad` names live in Relevent.jl (`rem_dyad`).
- `REMResult` prints through the shared `Networks.print_coeftable` (R-style
  coefficient table, significance codes, p-values floored at `<1e-16`).
- `EventNetworkState` is fully concretely typed (no `Any` fields); risk-set
  actor ordering is deterministic (sorted) before sampling.
- Constructing an `Event` with `sender == receiver` no longer warns
  (exclude self-loops via `CaseControlSampler(exclude_self_loops=true)`).
- **Docs and README describe the current behaviour** (panel item 8): the
  installation blocks no longer tell users to add NetworkDynamic.jl (a weak
  dependency, needed only for the `EventSequence(::DynamicNetwork)` adapter)
  and mirror Networks.jl's wording; the "no hard dependency on the network
  stack" sentence is replaced by the truth (Networks.jl is a hard dependency,
  NetworkDynamic.jl the extension); the events guide no longer claims a
  self-loop warns; `coeftable` is documented as returning a
  `Networks.CoefficientTable` everywhere; the hand-rolled confidence-interval
  and event-bootstrap recipes are replaced by `confint` and `control_draw_cov`;
  "Time-Varying Effects" distinguishes period splitting from the memory models
  and points at `window=`; the decay guide's statistics table separates the
  weighted (decaying) triadic default from the `weighted=false` count that does
  not; `compute_decay_weight` is shown with its actual argument order
  (`elapsed_time, decay`); and the estimation guide's `seq`-less tie example is
  self-contained. `tools/check_snippets.jl` executes every block of every page
  with 0 failures and no warnings.
- `CLAUDE.md` rewritten to the current code: docs/snippet commands, the `show`
  conventions, the docstring-example testset, the teaching dataset, the
  four-fixture table, and cross-repo pointers.

### Fixed

- Center covariates within each conditional-logit stratum before evaluating
  the likelihood, derivatives and sandwich scores. A statistic constant within
  every risk set now has exactly zero information on every platform, avoiding
  roundoff that could falsely report separation or finite uncertainty. Form
  covariance from mean-centered rows and compensate its sum across strata to
  preserve collinear directions. The shared optimizer's information-rank
  tolerance accounts for the largest stratum's dot-product length. Regression
  tests cover large stratum offsets and Efron denominator weights; the kernel
  retains its zero-allocation contract.

- The full-risk-set covariance test checks identical replicate coefficients and
  permits only floating-point centering roundoff in their reported spread.
- **`EventSequence(::DynamicNetwork)` silently converted masked dyads.** An
  `Event` is an instant and cannot record that a dyad is *unobserved*, so an
  unobserved dyad became a never-happened non-event — which biases a likelihood
  that is *conditional on the risk set*. The adapter now rejects a masked
  dynamic network unless `missing=:face` is passed.

- `RecencyStatistic(directed=false)` looked up one direction only (see
  Breaking): the "either direction" recency now takes the later of the two
  last-event times.
- P-values no longer underflow to exactly `0.0` in the far tail (now the
  shared `Networks.z_pvalues`, floored at `floatmin`; see Breaking).
- Newton–Raphson hardened — step halving when the log-likelihood would
  decrease, a combined `|Δll|`/gradient-norm stopping rule, standard errors
  always from the post-update Hessian — now provided by `Networks.newton_fit`
  (see Changed), which REM no longer duplicates.
- `compute_decay_weight` was documented (README, events guide) with its
  arguments swapped; the signature is `(elapsed_time, decay)`.
- CLAUDE.md and this file cited survival 3.8-3 for `rem_clogit.toml`; every
  fixture's `[provenance]` block records 3.8.6 (round 2).

### Performance

- **`fit_rem(::DataFrame)` no longer allocates per row** (criterion 4). The
  stratum validation loop ("exactly one case, at least one control") iterated
  `zip(observations.stratum, observations.is_event)` in the same scope that
  pulled those columns out of the DataFrame, where they are typed
  `AbstractVector` — so every row went through dynamic dispatch: ≈10
  allocations and 250 B per row, 420 000 allocations and 10.8 MB of the
  16.7 MB a 2000-event / 20-control fit allocated in total, the benchmark
  suite's `fit` group made it visible. The loop now runs behind the function
  barrier `_stratum_counts(strata, y)`, which specialises on the concrete
  column types and allocates only the O(strata) Dict storage (47 allocations
  for 42 000 rows; the fit is 6.2 MB and, end to end from the sequence,
  7.6 ms instead of 15.8 ms — the benchmark suite's `fit/a100_e2000`).
  Pinned by `@allocations` in the
  "Streams in O(events)" testset and in `benchmark/regression_tests.jl`.
- **`update!` allocates 0 bytes on a warmed state** (round 2). The adjacency
  update `push!(get!(state.out_neighbors, s, Set{Int}()), r)` evaluated the
  empty set on every call, key present or not — 80 B per line, 160 B per
  absorbed event, 2.3 allocations/event in the `state/update_stream`
  benchmark, on the one path the streaming design runs once per event. The
  lazy `get!(() -> Set{Int}(), …)` form allocates only for a new key. Pinned
  at 0 B (no log, no window; plain and decayed) and ≤ 64 B with the event log
  or the window FIFO kept, in the "update! is allocation-free" testset and in
  `benchmark/regression_tests.jl`.
- **Time to first fit is 0.6 s instead of 3.7 s** (panel 2026-09, item 18).
  Measured in a fresh session on this machine (Julia 1.12.6, `@time using
  REM; @time fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=5,
  seed=1); @time sprint(show, fit)` on a 6-actor / 20-event sequence),
  before → after the `@compile_workload`: `using REM` 0.78 s → 0.81 s, first
  `fit_rem` **3.66 s → 0.58 s**, first `show` **0.38 s → 0.002 s**. The
  remaining 0.6 s is the specialisation of the design loop on the caller's
  particular `StatisticSet` tuple type, which no workload can precompile for
  every combination. The package image takes ~6 s to precompile, once.
- **`generate_observations` streams in O(events)** (panel 2026-09, criterion 4):
  the per-event cost no longer carries the actor universe. `n_dyads` was
  called once per event and built two `Set`s of every actor (139 KB/event at
  2000 actors); it is now an allocation-free sorted merge, and for a static
  risk set the provider computes it once. The Efron `excluded`/`forced` sets
  are built only inside a real tie block (every other policy excludes the case
  by a tuple compare), the rejection sampler reuses one set, and the design is
  accumulated column-major into one `p × n_rows` buffer through
  `compute_all!` instead of one `Observation` with its own `Vector` per row.
  Pinned: 2000 events allocate ≤ 1.5× as much at 2000 actors as at 100, and
  4000 events ≤ 2.2× as much as 2000.
- **Single-pass CSR strata index and an allocation-free derivative kernel**
  (items 14, 26): the `Dict(s => findall(==(s), strata))` index was O(strata ×
  rows) (0.93 s at 20k strata, thirty times the fit itself) and its iteration
  order made the likelihood's summation order non-deterministic; the new
  `_StrataIndex` (`offsets`/`order`/`case`) is built in one pass in ascending
  stratum order. `_clogit_derivatives!` writes into preallocated buffers and
  allocates **0 bytes** per evaluation after warm-up; the closure `newton_fit`
  drives allocates exactly the `(p)` gradient and `(p×p)` Hessian it returns,
  the same at 300 and 3000 strata (was 400 B/call plus per-stratum vectors in
  the sandwich loop, which now reuses the workspace).
- **The triadic statistics allocate nothing** (panel 2026-09, item 26):
  `intersect(out_neighbors, in_neighbors)` + `delete!` materialised a `Set` per
  evaluation (320 B for `TransitiveClosure`, 1488 B for `CommonNeighbors`); the
  new `_count_common`/`_sum_common` iterate the smaller neighbour set and test
  membership in the larger — O(min degree), 0 B — and the shared mutable
  `_EMPTY_NEIGHBORS` is an immutable singleton. `GeometricWeightedFourCycles`
  builds its unweighted `FourCycle` once in the constructor instead of once per
  evaluation (which allocated a name string each call). Pinned at 0 B for every
  exported statistic.
- **Lazy decay:** each count is stored as `(value, last_update_time)` and
  decayed on read, making `update!` O(1) per event; the old eager
  `apply_decay!` scanned every count on each time advance (quadratic in the
  event stream). The eager `apply_decay!` survived unexported as a
  materialiser of the raw tables until round 3, when it was removed as dead
  code (see Changed).
- Incremental in/out-neighbor sets maintained per event — neighbor and
  common-neighbor queries are O(degree) instead of rescanning the event
  history.
- Single-pass strata validation in `fit_rem` (was effectively quadratic in
  the number of events).
- Conditional-logit derivatives use preallocated workspaces, `mul!`, and
  `BLAS.ger!` rank-1 updates instead of per-row `x·x'` outer-product
  allocations; tuple-backed `StatisticSet` gives statically dispatched
  statistic loops.

## [0.1.0] - 2026-02-09

Initial release: relational event sequences, decaying network state,
event-history statistics, and stratified case-control (conditional logit)
estimation.
