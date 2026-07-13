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
- **`Observation` and `REMResult` gained fields** (`risk_set_size`,
  `sampling_prob`; `strata`, `risk_set_sizes`, `sampling_probs`). The old
  positional constructors still work. *Migration:* none, unless you
  destructured the structs positionally.

### Added

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
  3.8-3, R 4.6.1) and `test/fixtures/r/rem_clogit.R` regenerates it. The
  design matrix is rebuilt from the raw edgelist *in plain R*, so this checks
  the STATISTICS (`Repetition`, `Reciprocity`, `SenderActivity`,
  `ReceiverPopularity`, `TransitiveClosure`) as well as the estimator, and the
  risk set is enumerated in full on both sides so there are no sampled
  controls to reconcile across two RNGs. Tolerance **1e-8** — both sides
  maximize the same exact conditional-logit likelihood by Newton-Raphson, so
  nothing is allowed to differ but floating-point summation order. **Observed
  agreement: < 1e-13** on every coefficient and standard error.
- **Robust standard errors: `fit_rem(seq, stats; se=:hessian|:sandwich|:bootstrap)`**
  (issue REM#2). The inverse-Hessian SEs are computed conditional on ONE sampled
  control set *and* assume the observed information equals the score variance;
  the two new options relax the two assumptions separately, and neither is a
  parametric bootstrap — what is missing here is variance from the *sampling
  design*, not from the model.
  - `se=:sandwich` — the **event-clustered Godambe sandwich** `H⁻¹ B H⁻¹`, with
    the meat `B = Σ_e u_e u_eᵀ` the outer product of the per-event score
    contributions (each event is exactly one stratum, so the event is the
    clustering unit). Robust to misspecification of the within-stratum
    conditional model; still computed on the one drawn control set. Available on
    both the `EventSequence` and the observations-`DataFrame` methods.
  - `se=:bootstrap` — **repeated control sampling**: redraw the case-control risk
    set `n_boot=100` times (independent seeds from `rng`, so a fixed `rng`
    reproduces the SEs), refit on each, and combine by the **law of total
    variance**: `V = W̄ + (1 + 1/B)·B_between`, the mean of the per-draw
    inverse-information covariances plus the empirical covariance of the refits.
    That second term is exactly the risk-set sampling variability the other two
    options condition away, so these SEs necessarily **exceed** the `:hessian`
    ones (1.03–1.17× on the test fixture) and the gap *is* the missing component.
    The between-draw covariance *alone* would be a smaller number than `:hessian`,
    not a larger one — it is one component of the total, not the total. Refused
    (with a pointer to the `EventSequence` method) on the observations-`DataFrame`
    method, where the controls have already been drawn once and for all.
    Runs on the ONE shared `Networks.bootstrap_cov` loop.
  The point estimates are unchanged under all three — they remain those of the
  original (`seed`) control draw; only the covariance is replaced.
- **`sampling_probs(fit)` and `risk_set_sizes(fit)`** (exported): the control
  inclusion probabilities each stratum conditioned on, and its risk-set size.
  These are part of the estimand, not an implementation detail — they are what
  `is_exact` reads and what the missing variance component is *about*.
- `se_method(fit)` now reports what was actually used
  (`:hessian`/`:sandwich`/`:bootstrap`), read off the new `REMResult.se_type`
  field; `show` names the estimator and prints a *different* caveat for each,
  because each accounts for something different (only `:hessian` is
  "understated"; only `:bootstrap` includes the sampling variance). With the full
  risk set no caveat is printed at all — there is nothing to condition away.

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
  columns, strata without exactly one case throw `ArgumentError`) and a
  tied-timestamp warning in `generate_observations` (the ordinal likelihood
  applies no tie correction).

### Changed

- `REMResult` prints through the shared `Networks.print_coeftable` (R-style
  coefficient table, significance codes, p-values floored at `<1e-16`).
- `EventNetworkState` is fully concretely typed (no `Any` fields); risk-set
  actor ordering is deterministic (sorted) before sampling.
- Constructing an `Event` with `sender == receiver` no longer warns
  (exclude self-loops via `CaseControlSampler(exclude_self_loops=true)`).

### Fixed

- **`EventSequence(::DynamicNetwork)` silently converted masked dyads.** An
  `Event` is an instant and cannot record that a dyad is *unobserved*, so an
  unobserved dyad became a never-happened non-event — which biases a likelihood
  that is *conditional on the risk set*. The adapter now rejects a masked
  dynamic network unless `missing=:face` is passed.

- P-values computed via `2·ccdf(Normal(), |z|)` no longer underflow to
  exactly `0.0` in the far tail.
- Newton–Raphson hardened: step halving when the log-likelihood would
  decrease, a combined `|Δll|`/gradient-norm stopping rule, and standard
  errors always computed from the post-update Hessian (never stale).

### Performance

- **Lazy decay:** each count is stored as `(value, last_update_time)` and
  decayed on read, making `update!` O(1) per event; the old eager
  `apply_decay!` scanned every count on each time advance (quadratic in the
  event stream). `apply_decay!` is retained for compatibility but no longer
  needed.
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
