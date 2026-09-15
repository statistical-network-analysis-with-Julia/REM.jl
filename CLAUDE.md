# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REM.jl is a Julia implementation of Relational Event Models for statistical analysis of time-stamped relational events in networks. It is a port of [eventnet](https://github.com/juergenlerner/eventnet).

## Development Commands

```bash
# Run tests from shell (flat @testset blocks in test/runtests.jl; filter by
# editing the file — there are no separate test files)
julia --project -e 'using Pkg; Pkg.test()'

# Build the docs (Documenter, STRICT: warnonly=false, checkdocs=:exports)
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl

# Execute every ```julia block in README.md and docs/src/** (site repo tooling;
# ../.snippet-env is built per tools/README.md there). Must report 0 failures.
(cd ../statistical-network-analysis-with-Julia.github.io &&
 SNWJ_ROOT=.. julia --project=../.snippet-env tools/check_snippets.jl REM.jl)

# Allocation-regression gates (standalone @test blocks; CI runs them on the
# ubuntu / 1.12 cell after the suite). benchmark/Project.toml sources Networks
# at ../../Networks.jl, so this instantiates from the sibling layout.
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/regression_tests.jl

# BenchmarkTools suite: BENCHJL timing lines plus the SCALING assertions
# (events 2000→4000 ≤ 2.5× time / ≤ 2.2× bytes, actors 100→2000 ≤ 1.5× bytes;
# exits non-zero on a violation). The site's tools/run_benchmarks.jl REM runs
# both files and reports PASS/FAIL.
julia --project=benchmark benchmark/benchmarks.jl
```

```julia
# Run tests from Julia REPL
using Pkg; Pkg.test("REM")

# Load the package in development
using Pkg; Pkg.develop(path=".")
using REM

# Run specific test file directly
include("test/runtests.jl")
```

Every ```julia block in the docs is executed **in file order** by the snippet
checker (one session per file), so a block may reuse the variables of the
blocks above it; a fragment that is not meant to run is preceded by
`<!-- skip-check -->`. The printed `Output:` blocks are pasted from real runs
— refresh them when `show` or a default changes.

## Architecture

### Core Data Types (`src/types.jl`)
- `Event{T}` - Single relational event with sender, receiver, timestamp, type, and weight
- `EventSequence{T}` - Time-sorted collection of events with an actor universe. Declare the universe with `EventSequence(events; actors=ActorSet(...))` (isolates and noncontiguous IDs included); when omitted it falls back to the observed event endpoints and `actors_declared` is `false` (fitting against such a sequence warns)
- `ActorSet` - Set of actors with optional ID-to-name mapping
- `NodeAttribute{T}` - Actor-level attribute storage with an *optional* default (no default → a missing actor is an `ArgumentError`, never a silent fill); prints one line (`NodeAttribute{Float64}(:icr, 37 actors, no default)`), never the values Dict
- `RiskSet` - Defines potential dyads (sender × receiver) for case-control sampling

### `show` conventions (criterion 5)

`show(io, ::REMResult)` — the 2-argument method — prints the **R-style block** (header lines, the shared `Networks.print_coeftable`, the estimator-matched caveat): that is the family convention (ERGM.jl `src/terms/base.jl`, Relevent.jl) and what `println(fit)` in every tutorial relies on, so it must stay multi-line. The header's standard-error line says `inverse Hessian (full risk set)` when nothing was sampled and `(one control draw)` otherwise. Every other type prints **one informative line** on the 2-argument `show` (so containers and logs get it too), never Dict internals: `EventSequence{Float64}(481 events, 37 actors, declared universe)`, `EventNetworkState{Float64}(37 actors, 481 events absorbed, no memory decay, current_time = 481.0)` (memory model named: `halflife h (decay λ)` / `window w`; the count is the `n_events` field, kept whether or not the log is), `ActorSet(37 actors)`, `RiskSet(event 5: 3 senders × 4 receivers, 9 dyads, self-loops excluded)`, `CaseControlSampler(n_controls=100, exclude_self_loops=true, seed=42)`, `StatisticSet(3 statistics: repetition, reciprocity, transitive_closure)` (elided after 7 names), `Event(1 → 2 @ 3.5)`, `NodeAttribute{Float64}(:icr, 37 actors, no default)` (and `_NoDefault` prints `no default`, so the statistics wrapping an attribute inherit a readable line). Pinned by the "show is compact and informative" testset via `sprint(show, x)`.

### Docstrings: every export carries a runnable example

The "Every exported docstring carries a runnable example" testset walks `names(REM)`, requires a fenced ```julia (or jldoctest) block in the REM-owned docstring of every export — including the docstrings REM attaches to the shared Networks/StatsAPI bindings (`compute`, `name`, `compute_all`, `has_edge`, `coef` … `coeftable`) — and **executes every block** in a fresh module. So a new export needs a docstring with a self-contained example (`using REM` first; no variables from another block), and the docs build (`checkdocs=:exports`) needs it listed in `docs/src/api/*.md`.

### The teaching dataset

README, `docs/src/index.md`, `getting_started.md`, the estimation and decay guides all teach on **`Networks.load_dataset(:wtc_police_calls)`** — relevent's WTC police radio calls: 481 events, `ActorSet(1:37)` (two officers never appear; they belong in the risk set), the ICR covariate as `NodeAttribute(:icr, …)`, an ordinal clock (no ties, so `ties=` never bites), and the **full risk set** (`n_controls = 37·36 − 1 = 1331`, ~0.3 s) as the headline fit, because it is exact and reproduces relevent's `wtcfit1` (`NodeSum(icr)` = `CovInt`, coefficient 2.1045; pinned by `rem_relevent_wtc.toml`). Sampled fits (`n_controls=100, seed=42`) are shown as the large-network tool with the honest caveat: for a misspecified model the sampled estimate converges to a pseudo-true value that depends on the draw and moves toward the full-risk-set one as `n_controls` grows (repetition −0.50 ± 0.07 over ten draws at 20 controls, −0.41 ± 0.04 at 100, −0.33 ± 0.02 at 400, −0.34 in full on the 7-statistic model, against per-draw standard errors of 0.08 / 0.05 / 0.04 / 0.03 — all with `Xoshiro(1)`/`seed=1`, the seeds the guide's runnable block uses; the `control_draw_cov` docstring quotes the same run — the guide tabulates mean ± sd over draws via `control_draw_cov`, never one seed, because seed 42's −0.72 at 20 controls is 1.8 draw-sds out); the "Case-control sampling is consistent for a correctly specified model" testset pins that with a correctly specified model the sampled estimates centre on the truth, and the calibration testset that their standard errors are right. The simulate-and-recover example survives only where it teaches simulation (the site's event-modelling page and the recovery testset).

### Network State (`src/network.jl`)
- `EventNetworkState{T}` - Tracks cumulative network state for efficient statistic computation
- Maintains dyad counts, degrees, last-event times and adjacency with optional exponential decay; decayed counts are stored as `(value, last_update_time)` pairs and decayed lazily on read relative to `current_time`, so advancing the clock is O(1) and each event is absorbed exactly once. The per-event log `event_history` (`(sender, receiver, time, weight)` per absorbed event — public, read-only) is kept only while `keep_history` is set: none of REM's own statistics reads it, so `generate_observations`/`compute_statistics` build the state with `keep_history = any(needs_history, stats)` and a REM-native fit retains O(actors + dyads), not O(events). The exported trait `needs_history(stat)` defaults to **`true`** for a statistic REM does not know (Relevent.jl's `PShift`/`Momentum`/`PriorInteraction`/`SendingCapacity`/`ReceivingCapacity` read the log through the field) and to `false` for the six REM families — conservative: a foreign statistic is never silently handed an empty log

### Statistics (`src/statistics/`)
All statistics implement the `compute(stat, state, sender, receiver) -> Float64` interface:

- **Dyad** (`dyad.jl`): `Repetition`, `Reciprocity`, `InertiaStatistic`, `RecencyStatistic`
- **Degree** (`degree.jl`): `SenderActivity`, `ReceiverActivity`, `SenderPopularity`, `ReceiverPopularity`
- **Triangle** (`triangle.jl`): `TransitiveClosure`, `CyclicClosure`, `SharedSender`, `SharedReceiver`, `CommonNeighbors`, `GeometricWeightedTriads`
- **FourCycle** (`fourcycle.jl`): `FourCycle` with various cycle type configurations, `GeometricWeightedFourCycles`
- **Node** (`node.jl`): `AttributeMatch`, `ActorMix`, `NodeDifference`, `NodeSum`, `NodeProduct`, `SenderAttribute`, `ReceiverAttribute`, `SenderCategorical`, `ReceiverCategorical`

### eventnet's definitions (WP2, 2026-09)

REM.jl is a port of eventnet, and the triadic family is where a port can drift: since 0.2.0 `TransitiveClosure`, `CyclicClosure`, `SharedSender`, `SharedReceiver`, `CommonNeighbors` and `FourCycle` **default to eventnet's weighted form** — `weighted=true, aggregation=:min`: for a candidate `s → r`, every two-path through a third party `k ∉ {s, r}` contributes `aggregation(w(first dyad), w(second dyad))` of its (decayed, event-weighted) counts, and the values of the parallel two-paths are added; `:max`, `:sum`, `:product` are eventnet's other combining functions (`_AGGREGATIONS`, checked at construction). A two-path exists only when both legs have a positive count (adjacency), which matters for `:sum`/`:max`. `CommonNeighbors` is the SYM form on the undirected counts; `FourCycle` aggregates its three weights. `weighted=false` is the pre-0.2 count of distinct third parties (adjacency only, never decays). `GeometricWeightedTriads`/`GeometricWeightedFourCycles` stay count-based (`e^α(1 − (1 − e^{−α})^n)`). Every statistic has a hand-computed testset with the expected numbers written out, and `test/fixtures/rem_eventnet.toml` pins the weighted forms, the four aggregations, decayed counts and windowed counts against `survival::clogit` on a design rebuilt in plain R.

Participation shifts (`PSAB-BA`, …) are **not** eventnet statistics: they are relevent's, and live in Relevent.jl (`PShift`), which dispatches inside `fit_rem` through the shared `compute` generic; REM's README and statistics guide point there.

### Two memory models: `decay` and `window`

`EventNetworkState` has two mutually exclusive memory models (an `ArgumentError` combines them): eventnet's **halflife decay** (`decay=halflife_to_decay(h)`; lazy `(value, last_update)` pairs; adjacency never expires) — the only one eventnet offers — and a **sliding window** (`window=`; the `remstats`-style alternative): an event older than `current_time − window` stops counting in every count, degree and adjacency set. The window is a FIFO of the live events (`window_log`) with an expiry cursor (`window_head`) advanced by `_expire!`, which every accessor calls first because `generate_observations`/`compute_statistics` move the clock by assigning `current_time` directly; the `live_*` dicts count live events per key so an emptied key is deleted (exact `0.0`, no residue, no edge). Amortized O(1) per event; the FIFO is compacted once the expired prefix outweighs the live suffix. Units: the clock's for numeric time, **seconds** for `Date`/`DateTime` (`_elapsed_seconds`); an event exactly `window` old still counts; `window=Inf`/`nothing` is no window and is tested to give the identical design row for row. `last_event_time` (`RecencyStatistic`) and `event_history` are not windowed. `window=` is a keyword on `EventNetworkState`, `generate_observations`, `compute_statistics`, `fit_rem` and `control_draw_cov` (threaded through its redraws), typed `_WindowSpec = Union{Nothing, Real, Dates.Period, Dates.CompoundPeriod}` (`network.jl`): on a calendar clock `window=Day(2)` is converted to seconds by `_window_length`, and `halflife_to_decay(Day(7))` likewise — the calendar idiom must not fail with a `TypeError` that never mentions seconds; on a **numeric** clock a `Period` window is refused by the `EventNetworkState{T}` constructor (`T <: Dates.TimeType` is the test), the one choke point every entry point routes through, because 172 800 clock units would silently be no window (both pinned by the "Calendar clocks take Dates.Period" testset).

### Allocation-free statistics (item 26)

`compute(stat, state, i, j)` allocates **0 bytes** for every exported statistic (pinned on a warmed 30-actor state under no memory, decay and a window, and for `compute_all!`), and so does `update!` on a warmed state (the adjacency update must use the lazy `get!(() -> Set{Int}(), dict, key)` — the eager three-argument form evaluates the empty set on every call, 160 B per event; pinned at 0 B with no log and no window, ≤ 64 B otherwise, in the "update! is allocation-free" testset and `benchmark/regression_tests.jl`). The triadic family uses `_count_common(a, b, x1, x2)` / `_sum_common(f, a, b, x1, x2)` (`network.jl`): iterate the smaller neighbour set, test membership in the larger, skip `x1`/`x2` — instead of `intersect` + `delete!`. `get_out_neighbors`/`get_in_neighbors` are `@inline` and return `_NeighborSet = Union{Set{Int}, _EmptyNeighbors}`, where `_EMPTY_NEIGHBORS` is an **immutable** singleton `AbstractSet{Int}` (a stray `push!` throws instead of corrupting every caller). Two things to keep: the accessors must stay `@inline` (a non-inlined union return boxes, 16 B), and no helper may take more than two union-typed neighbour sets in one signature (`CommonNeighbors` writes its two loops out for that reason — four union arguments exceed Julia's union-splitting limit and fall back to dynamic dispatch).

`compute`, `name` and `compute_all` are **not REM's generics** — they are the shared statistic protocol defined in Networks.jl (`src/statistics.jl`) and imported by name, exactly as `gof` is. Every model package extends the same three functions with methods for its own statistic types, so `REM.compute === ERGM.compute` and the co-load that the whole statnet workflow depends on —

```julia
using ERGM, REM     # cross-sections with ERGM, dynamics with REM
compute(Edges(), net)                 # ERGM's method
compute(Repetition(), state, 1, 2)    # REM's method — same generic
```

— leaves the verbs usable unqualified instead of undefined by Julia's conflicting-export rule. Adding a statistic anywhere (Relevent.jl does exactly this) means `import REM: compute, name` and adding methods; never define a local `compute`. The rule is pinned by the "Namespace: co-loading ERGM and REM" testset, which also runs the check in a fresh process (ERGM is a test-only dependency).

A name that means something *different* in another package gets renamed, never re-exported: `NodeMatch` → `AttributeMatch`, `NetworkState` → `EventNetworkState`, `NodeMix` → `ActorMix` (vs `ERGM.NodeMix`, a cross-sectional mixing term). `REM.NodeMix` survives as a deprecated, **non-exported** alias — exporting it would recreate the collision. A name that means the *same* thing, in contrast, becomes a method on the shared generic and stays exported: `has_edge(state, s, r)` is a method of `Graphs.has_edge` (via `import Networks: has_edge`), because asking "is there a tie from i to j" of an accumulated event network is the question Graphs already asks of a graph.

`StatisticSet` (in `statistics/base.jl`) is tuple-backed: `compute_all`/`compute_all!` over a set compile to statically dispatched per-statistic calls (no dynamic dispatch in the observation/likelihood inner loop). Vectors of statistics passed to `generate_observations`/`compute_statistics`/`fit_rem` are converted to a `StatisticSet` internally; reuse a set across calls to avoid recompiling per tuple type.

### Estimation Pipeline (`src/observation.jl`, `src/estimation.jl`)
1. `CaseControlSampler` - Generates case-control observations from event sequence
2. `generate_observations()` - Computes statistics for cases and sampled controls, streaming in **O(events)**: a static risk set's dyad count is computed once by the provider (`_riskset_provider` returns `(RiskSet, n_dyads)`; `n_dyads` itself is an allocation-free sorted merge), the Efron `excluded`/`forced` sets are built only inside a real tie block (every other policy excludes the case by a tuple compare), and the design accumulates column-major into a `_ObsBuffer` (one flat `p × n_rows` vector filled through `compute_all!`) rather than one `Observation` with its own `Vector` per row. Pinned: the bytes allocated for 2000 events are ≤ 1.5× at 2000 actors vs 100, and ≤ 2.2× for twice the events
3. `fit_rem()` - Fits the stratified conditional logistic regression on the shared `Networks.newton_fit`

`fit_rem` is the canonical (and only) entry point. There is deliberately **no `rem` alias** — the statnet-style short verb the other model packages carry (`ergm`, `stergm`, …): `rem` is `Base.rem`, the remainder function, and a rival export would make `rem(7, 2)` an ambiguity error in every `using REM` session. relevent's R names (`rem`, `rem.dyad`) live in Relevent.jl as `rem_dyad`. The "No export collides with Base" testset pins that no exported name of REM is exported by Base. Keyword vocabulary is the family's: `maxiter`, `tol`, `se`, `rng`, plus `seed` (below); `control_draw_cov` adds `n_draws` and `threaded`.

### StatsAPI surface (item 15)

`REMResult` answers all ten verbs — `coef`, `stderror`, `vcov`, `confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable` — each a method on the `StatsAPI` generic imported by name in `src/REM.jl` (never a REM-local function), and the "StatsAPI co-loading" testset pins `Networks.check_statsapi(fit; strict=true)` for the `:hessian`, `:sandwich` and DataFrame-method fits. Two rules keep it honest: `var_cov` on the result is **the covariance that matches `se_type`** (inverse information / sandwich), so `stderror == sqrt.(diag(vcov))` under both; and `coeftable` returns the shared `Networks.CoefficientTable` built from the fit's own z- and p-values (not a `DataFrame`, since 0.2.0). `nobs` is the number of **events** (one stratum each — `survival::clogit`'s `nevent`), which is what `bic` scales by; `n_observations` (case-control rows) is a property of the sampling design. The `log_likelihood` field keeps its name; `loglikelihood(fit)` reads it.

### Standard errors: two estimators, and a diagnostic that is NOT one (REM#2, round 2)

`REMResult.se_type` records which estimator was used and `Networks.se_method(fit)` reports it, so a `show` method can never claim an estimator that was not run. **The point estimates are identical under both** — only the covariance differs.

- `se=:hessian` (default) — the inverse observed information of the partial likelihood on the risk set that was used. **With a sampled risk set this is NOT "understated"**: under nested case-control sampling the observed information of the *sampled* partial likelihood is a consistent estimator of the estimator's variance (Goldstein & Langholz 1992, Ann. Statist. 20; Borgan, Goldstein & Langholz 1995, Ann. Statist. 23) — the information lost by sampling fewer controls is what makes it larger than the full-risk-set one, and nothing further needs adding. The "Sampled-risk-set standard errors are calibrated" testset pins it by simulation from `_simulate_rem` (the package's own correctly specified generator) with events *and* controls redrawn on every replicate: 95 % Wald coverage in [0.92, 0.98] (measured 0.953 / 0.967), and no output anywhere says "understated".
- `se=:sandwich` — **event-clustered Godambe sandwich** `H⁻¹BH⁻¹`, meat `B = Σ_e u_e u_eᵀ` from the per-event score contributions (`_clogit_sandwich_cov`). Each event IS one stratum, so the event is the clustering unit; in R that is `clogit(... + strata(s) + cluster(s), method="breslow")` — the `cluster(s)` matters (without it coxph's robust variance is the per-row dfbeta sandwich, a different number) — and `rem_clogit.toml`'s `robust_std_errors` pins it (agreement 4e-16). Works on both the `EventSequence` and the `DataFrame` methods.
- **There is no `se=:bootstrap`, and it must not come back.** The first 2026-09 round shipped one that combined the mean within-draw covariance with `(1 + 1/B)·`(between-draw covariance of refits) by "the law of total variance"; the panel's simulation (and ours) showed it double-counts — 97–99 % coverage — because the sampled likelihood's own information already carries the sampling loss, and the `(1 + 1/B)` factor was Rubin's rule for the mean of B imputations misapplied to a single draw. Redrawing the controls measures something real but different: how far the **point estimate** of a *misspecified* model depends on the draw (a pseudo-true value that moves with `n_controls`). That is exported as the diagnostic `control_draw_cov(seq, stats; n_controls, n_draws=20, rng, threaded=true) -> (cov, sd, replicates, mean, n_unconverged)`, on the shared `Networks.bootstrap_cov` loop (a *parametric* bootstrap would still be the wrong tool: what is resampled is the sampling design, not the model). It is never combined with the Hessian and never reported by `stderror`; `se=:bootstrap` on either `fit_rem` method is refused by `_check_rem_se` with the reason and the pointer (before the shared `check_se`).

The control inclusion probabilities are exposed as `sampling_probs(fit)` (and `risk_set_sizes(fit)`) — part of the estimand, not an implementation detail. With the full risk set (`sampling_probs` all `1.0`) `is_exact` holds and `show` prints no note at all; a sampled fit prints a `Note:` (not a `Warning:`) stating the facts — the risk set was sampled (`_controls_note`: "100 of 1331 controls per event"), a misspecified model's estimate depends on the draw, refit with more controls or measure the spread — and `approximations` carries the same two entries.

### Collinearity and separation are loud (round 2, criterion 2)

Singular information now makes the shared `newton_fit` return `converged == false` and NaN uncertainty, even if its objective stopping rule was satisfied. Separation can still return `converged == true` with an unbounded coefficient. `_fit_stratified_clogit` runs one more kernel pass to identify the responsible statistics; `fit_rem(::DataFrame)` maps their column indices to names, warns, and stores them on the result. **Singular** (`fit.singular`): `newton_fit` returned a NaN covariance because −H was not positive definite or failed its scale-invariant numerical rank check; `_singular_columns(hess)` names the columns loading (> 0.1 of the max) on the eigenvectors of the non-positive eigenvalues — the null vector of `NodeSum(x)`/`SenderAttribute(x)`/`ReceiverAttribute(x)` is `(1, −1, −1)/√3` so all three are `singular_suspects`; a column constant within every stratum has a zero diagonal and is named alone. **Separated** (`fit.separated`): `survival::coxph`'s "coefficient may be infinite" rule on the Newton increment `δ = H⁻¹g` at the converged solution — ~1e-15 on a healthy coordinate (quadratic convergence overshoots the tolerance by orders of magnitude), O(1) on a separated one (for `ll ≈ −c·e^{−β}` the increment is exactly 1 however far β has run; measured −1.0000 on the WTC `NodeProduct(icr)` at β = −20, SE 17931); `_separated_columns` flags `|δ_j| > 0.01·max(1, |β_j|)`, which a finite maximum cannot reach at `newton_fit`'s stopping rule unless its standard error is ~100 (not identified either way). The rule is applied to the **standardised** coefficient and increment — `β_j·s_j`, `δ_j·s_j` with `s_j = std(X[:, j])` (`_column_scales`; 1.0 for a constant column) — because β and δ rescale together with the column, which makes the verdict scale-free like coxph's: on the raw scale the same separated `NodeProduct` with the attribute coded 0/1000 came back as β = −2.4e-5, SE 0.13, `is_exact == true` (round 3). Pinned at ×1, ×1000 and ×0.001 in the collinearity testset. Both set `is_exact` false, add an `approximations` entry and a `Warning:` under the table naming the statistics. **Beware in tests and docs**: a tiny toy sequence is often separated for real (the 3-event metadata toy had coefficients −42 / +25 with SEs of 7e4), so a toy that must be `is_exact` needs a case that repeats *and* one that does not.

### Tied event times: `ties=:error|:ordered|:breslow|:efron` (REM#2, finding 12)

The partial likelihood is a **Cox partial likelihood** (one stratum per event), so a tied timestamp is the classical Cox tie problem and Breslow/Efron are the classical answers — but note what a tie does *here*: statistics are read off the network state **before** the event, so ordering two simultaneous events lets the one placed first enter the **statistics** of the one placed second (its `Repetition`, its `Reciprocity`, its degrees). That is not a tie-break, it is invented information — which is why the default is refusal, not a sort.

The vocabulary is **`Networks.TIE_POLICIES`**, defined once in Networks.jl (`src/results.jl`) and shared with Relevent.jl; `Networks.check_tie_policy` is the guard, and it makes an option a model cannot honour **fail loudly instead of no-op**.

- `:error` (**default**) — names the tie (which events, which timestamp, how many in all) and throws.
- `:ordered` — sequence order, no correction: the pre-0.2 behaviour, now opt-in.
- `:breslow` — the network state is **frozen across the tie block** (the block is absorbed as a whole afterwards), each tied event is its own stratum, all share one denominator.
- `:efron` — as `:breslow`, plus the denominator weights `1 − (j−1)/d` on the `d` tied cases, carried per row in the **`tie_weight` column** and applied by `_fit_stratified_clogit` (`tie_weights=`) as `Σ_a w_a·exp(η_a)` in the stratum denominator, numerator unchanged. Requires the tied cases to be **distinct dyads** — a dyad competing with itself has no fractional weight, and inventing one can even go negative, so it throws. Efron on the same design *unweighted* is exactly Breslow (tested).
- `:batch` — **refused**, pointing at `:breslow`: with the risk set held fixed, a "simultaneous batch" in an ordinal likelihood IS the Breslow correction. (It is `Relevent.fit_timing`'s policy, where there is an exposure interval for a batch to consume.)

The policy is applied where the **design is built** (`generate_observations`), and rides with it: the applied policy is attached as the `"tie_method"` DataFrame metadata (`:note` style), so `fit_rem(::DataFrame, ...)` reports the truth without being told twice. A frame marked `:efron` that has *lost* its `tie_weight` column is refused rather than fitted unweighted while claiming Efron.

`tie_method(fit)` reports **what actually happened**, so it is `:none` when the data had no ties (a correction on tie-free data corrected nothing) and never `:error` (a tie under `:error` throws instead of returning). `is_exact(fit)` requires *both* the full risk set and `:none`. On tie-free data all four policies produce the identical design, row for row — the sharpest available check that a correction is a correction.

**Golden fixture** `test/fixtures/rem_ties.toml` (`test/fixtures/r/rem_ties.R`) pins `:breslow`/`:efron` against **`survival::coxph(..., ties="breslow"/"efron")`** on a sequence observed on a coarse clock (25 of 53 timestamps tied, up to 4 deep). The R script builds the counting-process design — one interval per distinct time, all `n(n−1)` dyads at risk, covariates frozen across each block — from the raw edgelist in plain R. Agreement is < 1e-11 on coefficients, standard errors and log-likelihood. That is what makes "we implemented Breslow" checkable.

### Package Extensions (`ext/`)
- `REMNetworkDynamicExt` (weak dep on NetworkDynamic.jl): `EventSequence(::DynamicNetwork)` converts edge activation spells to events at their onset times (onset-censored spells skipped by default). Tested by adding NetworkDynamic to the test target.

  It honours the **ecosystem conversion contract** (Networks.jl `src/conversion.jl`; per-path table in `Networks.jl/docs/src/guide/conversion_invariants.md`). An `Event` is an instant and has no way to say a dyad is *unobserved*, so a `DynamicNetwork` whose base network carries a missing-dyad mask is **rejected** (`missing=:error`, the ecosystem default; `missing=:face` is the auditable opt-in). This is not fussiness: silently turning an unobserved dyad into a never-happened non-event biases a likelihood that is **conditional on the risk set**, which is the estimand. `report=true` returns `(seq, ::Networks.ConversionReport)` naming what an event sequence cannot carry — spell termini (a dissolution is not an event), terminus censoring, vertex spells (the declared actor universe is flat over time, not a time-varying risk set), attributes, and the observation window. Pinned by the "NetworkDynamic extension: conversion invariants" testset.

## Modeling Assumptions and Behaviors

- **Ordinal likelihood only**: each event is one stratum of a conditional logistic regression; exact waiting times enter only through optional decay weighting, not as a hazard term (unlike `relevent::rem.dyad`'s interval likelihood).
- **The risk set is the estimand**: the likelihood is conditional on it, so the actor universe must be declared, never inferred from outcomes. `generate_observations`/`fit_rem` take `at_risk` (alias `riskset` in `fit_rem`) as a static actor set, a static `RiskSet` (asymmetric sender/receiver), a vector of per-event risk sets, or a callback `(event_index, state) -> RiskSet`. Every case is validated against its own risk set and each risk set must admit ≥ 1 control — both throw before fitting. Risk-set size and control sampling probability are recorded per stratum (`risk_set_size`/`sampling_prob` columns; `REMResult.risk_set_sizes`/`.sampling_probs`).
- **Sampling is without replacement**; when fewer distinct dyads exist than `n_controls`, the full risk set is enumerated instead (one-time warning). **All randomness flows through `rng`** (item 16): `generate_observations(...; rng=)` / `fit_rem(...; rng=)` draw the controls from the caller's `rng` (default `Random.default_rng()`), and `seed` — when given — pins the control draw to a local `Xoshiro(seed)` and takes precedence over `rng`. Nothing ever reads the global RNG behind the caller's back: two `fit_rem(...; rng=Xoshiro(9))` calls are identical. `control_draw_cov` draws its per-draw seeds from the same `rng` up front (through `Networks.bootstrap_cov`), which is what makes the threaded refits thread-count-independent; the "rng contract" testset pins all of it, including that `threaded=true` and `threaded=false` give bit-identical covariances (a real check on the 4-thread CI cell).
- **A self-loop event is refused under the default `exclude_self_loops=true`** by `_validate_case!` (checked first, before the not-in-its-risk-set message, whose fixes — declare the universe, fix `at_risk` — do not apply to a loop) with a message naming the event and both remedies; `exclude_self_loops=false` admits `i → i` as cases and controls. An empty statistics list is refused by `_require_statistics` on every fit entry point ("fit_rem needs at least one statistic") before any internal consistency check can surface.
- `EventNetworkState` maintains incremental `out_neighbors`/`in_neighbors` adjacency sets so neighbor queries are O(degree); adjacency records "ever had an event" and does not expire under decay (counts do decay) — under a `window` it does expire, with the last live event of the dyad.
- **Node attributes do not zero-fill silently**: `NodeAttribute(name, values)` (no default) throws an `ArgumentError` naming the attribute and the actor when a statistic reads an actor it lacks; `NodeAttribute(name, values, default)` is the explicit opt-in (`has_default`). The categorical/numeric mismatches (`ActorMix(gender, 1, 2)`, `NodeSum(gender)`) and the wrong-container mistakes (`fit_rem(::Vector{Event}, …)`, a misspelled statistic name in `fit_rem(::DataFrame, …)`, the arguments swapped — `fit_rem(stats, seq)` / `control_draw_cov(stats, seq)` — which throw `_swapped_arguments_hint`; a single bare statistic, `fit_rem(seq, Repetition())`, is accepted as a one-element model) all produce `ArgumentError`s that say what to do; a hand-built design's `is_event` may be `Bool` or 0/1 (`_event_indicator`), anything else is refused by name; pinned by the "Common-mistake errors" testset.
- **The estimation loop is `Networks.newton_fit`** (item 14). `_fit_stratified_clogit` hosts no Newton loop, no step halving and no Hessian inversion: it builds a single-pass CSR index of the rows by stratum (`_StrataIndex`: `offsets`/`order`/`case`, ascending stratum id when the ids are dense, first appearance otherwise — deterministic summation order, which the old `Dict(s => findall(==(s), strata))` was not, and O(rows) instead of O(strata × rows); item 26), allocates the per-stratum workspace once, and hands `newton_fit` the closure `_clogit_objective` around the in-place kernel `_clogit_derivatives!(grad, hess, X, idx, β, work, tw) -> ll`. The kernel is REM's own — a per-stratum softmax, not a logistic likelihood, so nothing of it belongs in `Networks.logistic_derivatives` — and it allocates **zero bytes** after warm-up (case-centered softmax, mean-centered covariance via BLAS gemm, and compensated accumulation across strata; `information_rtol=max(p, idx.max_size)*eps(Float64)` tells the shared rank guard the precision of the stratum dot products); the closure allocates exactly the `(p)` gradient and `(p×p)` Hessian it returns, the same for 300 and 3000 strata. Pinned by the "Allocation-free clogit kernel" testset. Upstream of the kernel, `fit_rem(::DataFrame)` reads its columns as `AbstractVector`s, so any per-row loop there must sit behind a function barrier (`_stratum_counts` is the one for the case/row validation; an inline loop dispatched per row and cost 10 allocations a row — pinned by `@allocations` in "Streams in O(events)" and `benchmark/regression_tests.jl`). `converged` and `iterations` on the result are `newton_fit`'s; `vcov` for `se=:hessian` is its Cholesky-based inverse information after a scale-invariant numerical rank check (NaN and `converged=false`, with a warning, when `−H` is not positive definite or numerically identifiable). `se=:sandwich` feeds that bread to `_clogit_sandwich_cov`, whose stratum loop reuses the same workspace. The `control_draw_cov` refits call `_fit_stratified_clogit` with the default `se` and keep only the coefficients.
- **An unconverged fit is loud** (criterion 2): `fit_rem(::DataFrame)` warns when `newton_fit` reports `converged == false`, naming `maxiter`, the final gradient norm and `tol`; `approximations(fit)` gets a "did not converge in N iterations — estimates and standard errors are not maximum-partial-likelihood values" entry, `is_exact(fit)` is `false`, and `show` prints `Converged: false (N iterations; see approximations)` plus a caveat under the table. Unconverged `control_draw_cov` refits are counted and warned about once.
- The z- and p-values are `Networks.z_pvalues(coefficients, std_errors)` (item 13; floored at `floatmin`, NaN where the standard error is not positive) and the `se=` keyword is validated by `Networks.check_se` (item 28) — REM keeps only the bespoke `se=:bootstrap` refusal (`_check_rem_se`), which explains why and points at `control_draw_cov`.
- **`RecencyStatistic` at Δ = 0** (a prior event exists, the clock has not moved): `:exp_decay` returns `exp(0) = 1` ("just happened"); `:inverse`/`:inverse_log` are singular there and return the documented cap `0.0`. Only "no prior event" is `0.0` under every transform.

## Release engineering

- **Precompile workload** (`src/REM.jl`, bottom; item 18): a `@setup_workload`/`@compile_workload` pair runs sequence construction, `generate_observations`, `fit_rem` (`se=:hessian` and `:sandwich`, `ties=:efron` on a 20-event stream with one tie), the DataFrame method, `compute_statistics`, all ten StatsAPI verbs and `show` on a 6-actor toy. It cut time-to-first-fit from 3.7 s to 0.6 s (numbers in the CHANGELOG). Keep it **silent**: the toy must converge and declare its actor universe, because any `@warn` it triggers prints at every precompile. When a new default changes what the pipeline compiles (a new `se=`, a new tie policy), add one call here.
- **CI** (`.github/workflows/CI.yml`): the clone list is the `[sources]` section of `Project.toml` (Networks, NetworkDynamic, ERGM — ERGM is test-only); `Documentation.yml` clones only what `docs/Project.toml` sources (Networks). The ubuntu / Julia 1.12 cell runs with `JULIA_NUM_THREADS=4` (so the `control_draw_cov` threaded-vs-serial pin is a real check) and then `benchmark/regression_tests.jl`. `docs/make.jl` keeps `"stable" => "dev"` until 0.2.0 is tagged (panel item 21).
- **Dead code policy**: `REMResult` has one constructor (all nineteen fields; the positional back-compat forms are gone), the pre-0.2 `Observation` row type and `observations_to_dataframe` are gone (the design is the `DataFrame` that `_ObsBuffer`/`_to_dataframe` build; nothing produced or consumed the row type), `apply_decay!` (the eager materialiser of the raw count tables, which nothing but a test ever called) is gone — counts decay lazily on read and there is no other read path — and `src/` carries no `TODO`/`FIXME`. `CHANGELOG.md` `[0.2.0] - Unreleased` lists every behavioural change; `Project.toml` `[compat]` covers every dependency (PrecompileTools `"1"`).

## Golden fixtures (R)

Four fixtures, all loaded with Networks.jl's `load_golden` (which refuses a fixture without a `[provenance]` block naming an existing script) and asserted with `check_golden`:

| fixture | script | reference | pins |
|---|---|---|---|
| `rem_clogit.toml` | `rem_clogit.R` | `survival::clogit` 3.8.6 | count statistics + estimator (below); `robust_std_errors` = `clogit(... + cluster(stratum), method="breslow")`, REM's `se=:sandwich` (4e-16) |
| `rem_ties.toml` | `rem_ties.R` | `survival::coxph(ties=)` | Breslow/Efron tie corrections |
| `rem_eventnet.toml` | `rem_eventnet.R` | `survival::clogit` | eventnet's weighted triadic family (`weighted=true`, each `aggregation`), undirected repetition, halflife-decayed counts (`decay=`), windowed counts (`window=`); same sequence as `rem_clogit` |
| `rem_relevent_wtc.toml` | `rem_relevent_wtc.R` | `relevent::rem.dyad` 1.2.1 | the ordinal model on the **bundled** WTC police calls (`Networks.load_dataset(:wtc_police_calls)`, R reads the same TSVs): `CovInt` = `NodeSum`, `CovSnd`/`CovRec` = `SenderAttribute`/`ReceiverAttribute`, `CovEvent(D)` = `DyadCovariate`; tolerance 1e-6 (BFGS vs Newton termination slack; log-likelihoods 1e-8) |

Regenerate any of them from the package root with `Rscript test/fixtures/r/<name>.R > test/fixtures/<name>.toml`. Two relevent facts the WTC script records: `CovInt` is the sender-**plus**-receiver covariate (collinear with `CovSnd + CovRec`, singular together), and the ICR-to-ICR product separates on these data (no ICR actor ever calls another), so `NodeProduct` is pinned by hand-computed tests only.

`test/fixtures/rem_clogit.toml` freezes `survival::clogit` (survival 3.8.6, R 4.6.1 — the `[provenance]` block is authoritative) on a simulated 10-actor / 80-event sequence. The R script rebuilds the **whole stratified design matrix from the raw edgelist in plain R** — it does not import anything Julia computed — so the fixture checks the *statistics* (`Repetition`, `Reciprocity`, `SenderActivity`, `ReceiverPopularity`, `TransitiveClosure(weighted=false)` — the count form, passed explicitly since the 0.2 default is eventnet's weighted one) as well as the estimator. The risk set is enumerated in **full** on both sides (pass `n_controls = n(n-1) - 1`, which makes `generate_observations` enumerate rather than sample), so there are no sampled controls to reconcile across two RNGs and the comparison is exact rather than distributional.

Tolerance **1e-8**: both sides maximize the same exact conditional-logit likelihood by Newton-Raphson, so nothing may differ but floating-point summation order. Observed agreement is <1e-13. Case-control *sampling* is a variance/compute tradeoff on top of this likelihood; it is not what R would disagree with, and it is not what this fixture tests. Both golden testsets also compare `sqrt.(diag(vcov(fit)))` to R's standard errors, so the covariance the StatsAPI surface exposes is pinned, not only its diagonal as stored in `std_errors`.

## Key Design Patterns

- Statistics are computed lazily using `EventNetworkState`, which updates incrementally (O(1) per event under both memory models) and is read **before** each event is absorbed
- Two memory models — eventnet's halflife decay or a sliding window — never both
- Case-control sampling enables estimation for large networks; the full risk set (`n_controls = n(n−1) − 1`) is the exact ordinal likelihood and the recommended fit whenever affordable
- All statistics return `Float64` and allocate nothing
- Everything a fit did is machine-readable (`Networks.fit_metadata`), and every warning has a matching entry in `approximations`

## Example Usage

```julia
using Networks, REM

wtc = load_dataset(:wtc_police_calls)
events = [Event(wtc.events[k, 2], wtc.events[k, 3], Float64(wtc.events[k, 1]))
          for k in 1:size(wtc.events, 1)]
seq = EventSequence(events; actors=ActorSet(1:37))       # declare the universe
icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:37))
stats = [Repetition(), Reciprocity(), SenderActivity(), ReceiverPopularity(),
         SenderAttribute(icr), ReceiverAttribute(icr)]

result = fit_rem(seq, stats; n_controls=37 * 36 - 1)      # full risk set: exact
sampled = fit_rem(seq, stats; n_controls=100, seed=42)   # case-control sampling
coef(result); vcov(result); confint(result); coeftable(result)   # StatsAPI
Networks.is_exact(result), Networks.approximations(sampled)
```

## Cross-repo pointers

- `Networks.jl/src/datasets.jl` owns `load_dataset(:wtc_police_calls)` and the TSVs under `Networks.jl/data/`; `test/fixtures/r/rem_relevent_wtc.R` reads the same files.
- The site's `examples/modelling-interaction-events.md` is the simulate-and-recover example; it should declare `actors=` on its `EventSequence` (otherwise `fit_rem` warns).
- Relevent.jl reads `EventNetworkState.event_history` (public, read-only) and extends `compute`/`name` for its own statistics; keep the field, the `needs_history` default and the `show` block stable for it.
