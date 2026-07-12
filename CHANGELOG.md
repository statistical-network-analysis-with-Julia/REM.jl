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
- **`has_edge` is no longer exported** (it collided with `Graphs.has_edge`).
  The function still exists. *Migration:* call it qualified as
  `REM.has_edge(state, s, r)`.
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

### Added

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

- `REMResult` prints through the shared `Network.print_coeftable` (R-style
  coefficient table, significance codes, p-values floored at `<1e-16`).
- `EventNetworkState` is fully concretely typed (no `Any` fields); risk-set
  actor ordering is deterministic (sorted) before sampling.
- Constructing an `Event` with `sender == receiver` no longer warns
  (exclude self-loops via `CaseControlSampler(exclude_self_loops=true)`).

### Fixed

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
