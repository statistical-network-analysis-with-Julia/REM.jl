"""
Model estimation for Relational Event Models.

Implements Cox proportional hazard model estimation using stratified
case-control data.
"""

"""
    REMResult

Results from fitting a relational event model.

# Fields
- `coefficients::Vector{Float64}`: Estimated coefficients
- `std_errors::Vector{Float64}`: Standard errors of coefficients
- `z_values::Vector{Float64}`: Z-statistics
- `p_values::Vector{Float64}`: P-values (two-sided)
- `stat_names::Vector{String}`: Names of statistics
- `n_events::Int`: Number of events in the model
- `n_observations::Int`: Total number of observations
- `log_likelihood::Float64`: Log-likelihood at convergence
- `converged::Bool`: Whether the optimization converged (see [`fit_rem`](@ref):
  an unconverged fit warns, is never `is_exact`, and says so in
  `Networks.approximations`)
- `strata::Vector{Int}`: Stratum IDs, in the order of the risk-set bookkeeping below
- `risk_set_sizes::Vector{Int}`: Size of each stratum's risk set (case included);
  empty when the observations carry no risk-set bookkeeping
- `sampling_probs::Vector{Float64}`: Probability with which each non-case dyad of
  the stratum's risk set entered the sample as a control (1.0 = full risk set).
  These are the **control inclusion probabilities**; read them with
  [`sampling_probs`](@ref) and the risk-set sizes with [`risk_set_sizes`](@ref).
- `se_type::Symbol`: How `std_errors` were ACTUALLY computed — `:hessian` or
  `:sandwich` (see [`fit_rem`](@ref)). This is what `Networks.se_method(fit)`
  reports.
- `tie_type::Symbol`: What was ACTUALLY done with tied event times — `:none`
  (the data had no ties, so no policy could bite), or the policy that did:
  `:ordered`, `:breslow` or `:efron` (see `ties=` in [`fit_rem`](@ref)).
  `:error` can never appear: under it a tie throws instead of fitting. This is
  what `Networks.tie_method(fit)` reports.
- `var_cov::Matrix{Float64}`: The covariance matrix of the coefficients **that
  matches `se_type`** — the inverse observed information for `:hessian`, the
  Godambe sandwich `H⁻¹BH⁻¹` for `:sandwich` — so that `std_errors ==
  sqrt.(diag(var_cov))` always. Read it with `vcov(fit)`. `NaN` throughout
  when the Hessian at the solution is not negative definite (see
  `Networks.newton_fit` and `singular` below).
- `iterations::Int`: Newton–Raphson iterations taken by `Networks.newton_fit`
- `singular::Bool`: `true` when the observed information at the solution is
  **singular** (not positive definite), so the standard errors are undefined
  (`NaN`) — collinear statistics (`NodeSum(x)` next to `SenderAttribute(x)`
  and `ReceiverAttribute(x)`), a statistic constant within every stratum, or
  a separated one. The fit warns, `converged` and `Networks.is_exact` are
  `false`, and `Networks.approximations` and `show` name the suspects.
- `singular_suspects::Vector{String}`: The statistics that load on the null
  direction of the information matrix (empty unless `singular`) — the ones to
  drop or combine.
- `separated::Vector{String}`: The statistics whose coefficient **may be
  infinite** (separation: in every stratum the case sits on one side of the
  controls on it, so the partial likelihood keeps improving as the coefficient
  runs away). Detected as `survival::coxph` does — the Newton increment at the
  converged solution is still O(1) on that coordinate although the
  log-likelihood no longer moves. The fit warns, `Networks.is_exact` is
  `false`, and `show` prints the caveat under the table. Empty when nothing
  separated.

The StatsAPI verbs `coef`, `stderror`, `vcov`, `confint`, `loglikelihood`,
`nobs`, `dof`, `aic`, `bic` and `coeftable` all have methods for `REMResult`
(`Networks.check_statsapi(fit; strict=true)` passes), and the result-metadata
accessors `Networks.is_exact`, `se_method`, `tie_method`, `approximations` and
`fit_metadata` read off it what the fit actually did. `show(fit)` prints the
R-style block every tutorial relies on: the header (events, observations,
risk-set size and control sampling probability, log-likelihood, convergence
with the iteration count, the standard-error estimator, the tie policy when
one bit), the coefficient table, and the caveat that matches the estimator.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=19, seed=1)   # full risk set
fit isa REMResult                       # true
fit.stat_names                          # ["repetition", "reciprocity"]
fit.converged, fit.n_events, fit.se_type, fit.tie_type   # (true, 8, :hessian, :none)
coef(fit) == fit.coefficients           # true
println(fit)                            # the R-style results block
```
"""
struct REMResult
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    z_values::Vector{Float64}
    p_values::Vector{Float64}
    stat_names::Vector{String}
    n_events::Int
    n_observations::Int
    log_likelihood::Float64
    converged::Bool
    strata::Vector{Int}
    risk_set_sizes::Vector{Int}
    sampling_probs::Vector{Float64}
    se_type::Symbol
    tie_type::Symbol
    var_cov::Matrix{Float64}
    iterations::Int
    singular::Bool
    singular_suspects::Vector{String}
    separated::Vector{String}
end

"""
    sampling_probs(result::REMResult) -> Vector{Float64}

The **control inclusion probabilities**: the probability with which each non-case
dyad of a stratum's risk set entered the sample as a control, one entry per
stratum (in `result.strata` order). `1.0` means the full risk set was used (no
sampling). Empty when the fit carries no risk-set bookkeeping.

The likelihood is conditional on the sampled risk set, so these probabilities are
part of the estimand and not an implementation detail — they are what determines
whether `Networks.is_exact` holds, and how far a misspecified model's estimate
can depend on the particular control draw (see [`control_draw_cov`](@ref)).

See also [`risk_set_sizes`](@ref).

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition()]; n_controls=19, seed=1)   # 20 dyads: full risk set
sampling_probs(fit)          # all 1.0 — nothing was sampled away
sampling_probs(fit_rem(seq, [Repetition()]; n_controls=10, seed=1))   # all 10/19
```
"""
sampling_probs(result::REMResult) = result.sampling_probs

"""
    risk_set_sizes(result::REMResult) -> Vector{Int}

The size of each stratum's risk set (the case included), one entry per stratum in
`result.strata` order. Empty when the fit carries no risk-set bookkeeping.

See also [`sampling_probs`](@ref) for the control inclusion probabilities.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition()]; n_controls=10, seed=1)
risk_set_sizes(fit)          # all 20 — 5·4 ordered dyads per event, case included
```
"""
risk_set_sizes(result::REMResult) = result.risk_set_sizes

# Human-readable description of what the standard errors ACTUALLY are, shared by
# `show` and the approximations list so the two cannot disagree.
_se_description(result::REMResult) =
    result.se_type === :sandwich  ? "event-clustered sandwich (Godambe)" :
    _full_risk_set(result)        ? "inverse Hessian (full risk set)" :
                                    "inverse Hessian (one control draw)"

# The list of statistic names a caveat prints: backquoted, comma-separated
_name_list(names::Vector{String}) = join(("`" * n * "`" for n in names), ", ")

# "N iterations" when at least one Newton step was taken (`maxiter=0` leaves the
# count at 0 and prints nothing)
_iterations_note(result::REMResult) =
    result.iterations > 0 ? "$(result.iterations) iteration$(result.iterations == 1 ? "" : "s")" : ""

function Base.show(io::IO, result::REMResult)
    println(io, "Relational Event Model Results")
    println(io, "==============================")
    println(io, "Events: $(result.n_events), Observations: $(result.n_observations)")
    if !isempty(result.risk_set_sizes)
        rmin, rmax = extrema(result.risk_set_sizes)
        rs = rmin == rmax ? "$(rmin)" : "$(rmin)–$(rmax)"
        pmin, pmax = extrema(result.sampling_probs)
        ps = pmin ≈ pmax ? "$(round(pmin, digits=4))" :
             "$(round(pmin, digits=4))–$(round(pmax, digits=4))"
        println(io, "Risk-set size: $rs dyads, control sampling probability: $ps")
    end
    println(io, "Log-likelihood: $(round(result.log_likelihood, digits=4))")
    # An unconverged fit is a loud result: say so on the line a reader looks at
    # first, and point at `approximations` for the consequence
    iters = _iterations_note(result)
    if result.converged
        println(io, "Converged: true", isempty(iters) ? "" : " ($iters)")
    else
        println(io, "Converged: false (", isempty(iters) ? "" : "$iters; ",
                "see approximations)")
    end
    println(io, "Std. errors: $(_se_description(result))")
    # Only when ties actually occurred: on tie-free data the policy did nothing
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type)")
    println(io)
    # Shared ecosystem coefficient table (Networks.jl), with significance codes
    # and the p-value display floor (an underflowed p prints as "<1e-16")
    print_coeftable(io, result.stat_names, result.coefficients, result.std_errors,
                    result.p_values; z_values=result.z_values)

    if !result.converged
        println(io)
        println(io, "Warning: the optimizer did not converge, so the estimates and standard")
        println(io, "errors above are NOT maximum-partial-likelihood values. Increase `maxiter`,")
        println(io, "check for separation or collinearity, or simplify the model.")
    end

    # Name identification failures under the table. Singular information also
    # makes the shared optimizer report nonconvergence; separation may still
    # satisfy its objective stopping rule (the prose twin of `approximations`).
    if result.singular
        println(io)
        println(io, "Warning: the observed information at the solution is singular, so the")
        println(io, "standard errors are undefined (NaN). Suspect statistics: ",
                _name_list(result.singular_suspects), ".")
        println(io, "They are collinear, constant within every stratum, or separated: drop")
        println(io, "or combine them (`NodeSum(x)` next to `SenderAttribute(x)` and")
        println(io, "`ReceiverAttribute(x)` is the classic case).")
    end
    if !isempty(result.separated)
        println(io)
        println(io, "Warning: the coefficient on ", _name_list(result.separated),
                " may be infinite (separation):")
        println(io, "the partial likelihood keeps improving as it runs away, so the estimate")
        println(io, "and its standard error above are not meaningful. In every stratum the")
        println(io, "case sits on one side of its controls on that statistic; remove or")
        println(io, "transform it.")
    end

    # The sampled-risk-set caveat, and the prose twin of what
    # `approximations(result)` reports. It applies only when the risk set was
    # SAMPLED — with the full risk set the likelihood IS the ordinal one. What
    # it says is factual, not an apology: the inverse observed information of
    # the sampled partial likelihood is a consistent variance estimator under
    # nested case-control sampling (Goldstein & Langholz 1992; Borgan,
    # Goldstein & Langholz 1995 — the information loss from sampling is
    # already in it), so nothing is "understated"; what a sampled fit of a
    # MISSPECIFIED model does carry is a point estimate that depends on the
    # control draw, and the cure for that is more controls, not another
    # standard error.
    _full_risk_set(result) && return
    println(io)
    println(io, "Note: the risk set was sampled (", _controls_note(result), "), so this")
    println(io, "partial likelihood approximates the full-risk-set one. If the model is")
    println(io, "misspecified the estimates depend on the control draw: refit with a")
    println(io, "larger `n_controls` or the full risk set, or measure the draw-to-draw")
    println(io, "spread with `control_draw_cov`.")
    if result.se_type === :sandwich
        println(io, "The event-clustered sandwich standard errors are robust to")
        println(io, "misspecification of the within-stratum conditional model.")
    end
end

# "100 of 1331 controls per event" (or a range when the risk sets differ)
function _controls_note(result::REMResult)
    rmin, rmax = extrema(result.risk_set_sizes)
    pmin, pmax = extrema(result.sampling_probs)
    cmin = round(Int, pmin * (rmin - 1))
    cmax = round(Int, pmax * (rmax - 1))
    controls = cmin == cmax ? "$cmin" : "$(cmin)–$(cmax)"
    available = rmin == rmax ? "$(rmin - 1)" : "$(rmin - 1)–$(rmax - 1)"
    return "$controls of $available controls per event"
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors, so the two approximations that
# matter here — case-control sampling of the risk set, and arbitrary ordering of
# tied timestamps — are machine-readable instead of being a warning the user has
# already scrolled past.

"""
    _full_risk_set(result::REMResult) -> Bool

Whether every stratum used its FULL risk set (no control sampling): the
bookkeeping is present and every non-case dyad entered the stratum with
probability 1. This is the predicate `is_exact` reads.
"""
_full_risk_set(result::REMResult) =
    !isempty(result.sampling_probs) && all(≈(1.0), result.sampling_probs)

estimand(::REMResult) = :relational_event

"""
    objective(::REMResult) -> Symbol

`:partial_likelihood` — the stratified conditional-logit partial likelihood (one
stratum per event: the case dyad against its sampled controls), equivalent to the
Cox partial likelihood for case-control data.
"""
objective(::REMResult) = :partial_likelihood

"""
    is_exact(result::REMResult) -> Bool

`true` only when the risk set was NOT sampled, no tie correction was needed,
the optimizer converged, the information matrix is not singular and no
coefficient separated. With the full risk set (`sampling_prob == 1`) on
strictly ordered data, the conditional-logit partial likelihood is the exact
ordinal relational-event likelihood. Five things make the result an
approximation instead:

- **case-control sampling** (the default, `n_controls = 100`);
- **tied event times** — Breslow, Efron and arbitrary ordering are all
  approximations to a likelihood over an order the data does not determine (a
  fit on tie-free data has `tie_type == :none` and is unaffected); and
- **non-convergence** — an unconverged Newton–Raphson leaves estimates that are
  not the maximum of anything (`converged == false`; the fit warned, and
  `approximations` says so);
- a **singular information matrix** (`singular == true`) — the standard errors
  are undefined, and the coefficients of the collinear statistics are not
  identified; and
- **separation** (`separated` non-empty) — a coefficient whose maximum is at
  infinity; the returned value is where the optimizer stopped.

It also reports `false` for a `REMResult` built without risk-set bookkeeping,
where the sampling is unknown — the conservative answer.
"""
is_exact(result::REMResult) =
    _full_risk_set(result) && result.tie_type === :none && result.converged &&
    !result.singular && isempty(result.separated)

"""
    se_method(result::REMResult) -> Symbol

What the reported standard errors ACTUALLY are:

- `:hessian` — the inverse negative Hessian (observed information) of the
  conditional-logit partial likelihood on the risk set that was used (the
  default)
- `:sandwich` — the event-clustered Godambe sandwich `H⁻¹ B H⁻¹`, with the meat
  `B` the outer product of the per-event score contributions

Read straight off the fit, so it can never claim an estimator that was not used.
See [`fit_rem`](@ref) for what each one does and does not account for.
"""
se_method(result::REMResult) = result.se_type

# Relational-event data is an event stream, not a sociomatrix with a dyad mask:
# the unobserved-tie concept does not arise. (What *is* at stake is the risk set
# — declare it with `EventSequence(events; actors=...)` or `at_risk`.)
missing_method(::REMResult) = :none

"""
    tie_method(result::REMResult) -> Symbol

What was ACTUALLY done with tied event times, not what the estimator is willing
to do:

- `:none` — the data had no tied timestamps, so no policy bit. (The default
  policy `ties=:error` guarantees that any fit with ties in it was *asked* for.)
- `:ordered` — ties were broken in sequence order and no correction applied
- `:breslow` — the Breslow correction: one risk set per tie block
- `:efron` — the Efron correction: Breslow plus the `1 − (j−1)/d` denominator
  weights on the tied cases

`:error` never appears: under it a tie throws rather than fitting. See `ties=`
in [`fit_rem`](@ref).
"""
tie_method(result::REMResult) = result.tie_type

# Prose for the tie policy that was actually applied — the twin of `tie_method`,
# and empty when there were no ties (a correction on tie-free data corrected
# nothing, and claiming otherwise would be a caveat about nothing).
function _tie_approximation(result::REMResult)
    t = result.tie_type
    if t === :ordered
        return "tied event times were ordered arbitrarily (sequence order) with NO " *
               "tie correction (`ties=:ordered`): the event placed first enters " *
               "the statistics of the events placed after it, so the estimate " *
               "depends on a sort the data does not determine"
    elseif t === :breslow
        return "tied event times were handled by the BRESLOW correction " *
               "(`ties=:breslow`): the tied events share one risk set and each " *
               "contributes the same denominator. This is an approximation to the " *
               "average over the d! orderings, and the cruder of the two — it " *
               "biases coefficients toward zero as ties get heavier (`ties=:efron` " *
               "is the better approximation)"
    elseif t === :efron
        return "tied event times were handled by the EFRON correction " *
               "(`ties=:efron`): the tied cases enter the denominator of the j-th " *
               "of their strata with weight 1 − (j−1)/d. An approximation to the " *
               "average over the d! orderings — a close one, and what " *
               "`survival::coxph` defaults to — but the order of simultaneous " *
               "events remains unobserved"
    end
    return nothing
end

function approximations(result::REMResult)
    out = String[]
    # Non-convergence first: it makes every other caveat moot, because the
    # numbers are not the maximum of the likelihood the other caveats are about
    if !result.converged
        iters = _iterations_note(result)
        push!(out, "the optimizer did not converge" *
                   (isempty(iters) ? "" : " in $iters") *
                   " — estimates and standard errors are not " *
                   "maximum-partial-likelihood values (increase `maxiter`, check " *
                   "for separation or collinearity, or simplify the model)")
    end
    if result.singular
        push!(out, "the observed information at the solution is singular — standard " *
                   "errors are undefined (NaN) and the coefficients of the suspect " *
                   "statistics are not identified; check " *
                   _name_list(result.singular_suspects) *
                   " for collinearity (e.g. `NodeSum(x)` with both `SenderAttribute(x)` " *
                   "and `ReceiverAttribute(x)`), zero within-stratum variance, or separation")
    end
    if !isempty(result.separated)
        push!(out, "the coefficient on " * _name_list(result.separated) *
                   " may be infinite (separation): the Newton increment is still O(1) " *
                   "on that coordinate although the log-likelihood no longer moves, so " *
                   "in every stratum the case sits on one side of its controls on that " *
                   "statistic — the estimate and its standard error are not meaningful")
    end
    tie_note = _tie_approximation(result)
    isnothing(tie_note) || push!(out, tie_note)
    if !_full_risk_set(result)
        if isempty(result.sampling_probs)
            push!(out, "risk-set bookkeeping is absent from this fit, so the " *
                       "control sampling cannot be reported: assume the risk set " *
                       "was sampled")
        else
            pmin, pmax = extrema(result.sampling_probs)
            rng_str = pmin ≈ pmax ? "$(round(pmin, digits=4))" :
                      "$(round(pmin, digits=4))–$(round(pmax, digits=4))"
            push!(out, "case-control sampling of the risk set (each non-case dyad " *
                       "entered its stratum with probability $rng_str): the " *
                       "partial likelihood is an approximation to the full-risk-set " *
                       "ordinal likelihood, and if the model is misspecified the " *
                       "estimates depend on the control draw — refit with a larger " *
                       "`n_controls` or the full risk set, or measure the draw-to-draw " *
                       "spread with `control_draw_cov`")
        end
        # What the standard errors are, on a sampled risk set. The inverse
        # observed information of the sampled partial likelihood is a
        # consistent variance estimator under nested case-control sampling
        # (Goldstein & Langholz 1992; Borgan, Goldstein & Langholz 1995): the
        # information lost by sampling is already in it.
        if result.se_type === :sandwich
            push!(out, "the event-clustered sandwich standard errors (`se=:sandwich`) are " *
                       "robust to misspecification of the within-stratum conditional " *
                       "model, computed on the sampled risk set")
        else
            push!(out, "the inverse-Hessian standard errors (`se=:hessian`) are the " *
                       "observed information of the sampled partial likelihood — a " *
                       "consistent variance estimator under nested case-control " *
                       "sampling (the information lost by sampling is already in it)")
        end
    end
    return out
end

# The non-statistic columns of an observations frame (what `fit_rem` does not
# offer as statistics when a name is misspelled)
const _BOOKKEEPING_COLUMNS = ("event_index", "sender", "receiver", "is_event", "stratum",
                              "risk_set_size", "sampling_prob", "tie_weight")

"""
    fit_rem(observations::DataFrame, stat_names::Vector{String}; kwargs...) -> REMResult

Fit a relational event model using stratified Cox regression.

# Arguments
- `observations::DataFrame`: Output from `generate_observations`
- `stat_names::Vector{String}`: Names of statistic columns to include in the model

# Keyword Arguments
- `maxiter::Int=100`: Maximum Newton–Raphson iterations (the optimizer is the
  shared `Networks.newton_fit`)
- `tol::Float64=1e-8`: Convergence tolerance: the objective change must fall
  below `tol` and the gradient norm below `sqrt(tol)`
- `se::Symbol=:hessian`: How to compute the standard errors — `:hessian` or
  `:sandwich` (see the `fit_rem(::EventSequence, ...)` method for what each one
  accounts for). There is no `se=:bootstrap` in this package: the draw-to-draw
  spread of a sampled fit is a *sensitivity diagnostic*, not a standard error,
  and lives in [`control_draw_cov`](@ref).

**Non-convergence, collinearity and separation are loud.** When `maxiter` is
exhausted (or no Newton step improves the likelihood) the function warns —
naming `maxiter`, the final gradient norm and `tol` — and returns the last
iterate with `converged == false`. When the observed information at the
solution is singular (collinear statistics, a statistic constant within every
stratum) the standard errors are `NaN`, the fit warns naming the suspect
statistics, `fit.singular` is set and `converged == false`. When a coefficient separates (the
Newton increment is still O(1) on it at convergence — `survival::coxph`'s
"coefficient may be infinite" rule) the fit warns naming it and lists it in
`fit.separated`. In all three cases `Networks.approximations(fit)` records
it, `Networks.is_exact(fit)` is `false` and `show` prints the caveat.

Tied event times are handled where the observations are *generated* (see `ties=`
in [`generate_observations`](@ref)): by the time they are a DataFrame the design
is fixed. What this method does is *report* the policy honestly — it reads the
`tie_weight` column (the Efron denominator weights) and the `"tie_method"`
DataFrame metadata that `generate_observations` attaches, so `tie_method(fit)`
cannot claim a correction the design does not carry. A hand-built DataFrame with
neither reports `:none`, and is fitted unweighted, exactly as before.

# Returns
- `REMResult`: Fitted model results

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
stats = [Repetition(), Reciprocity()]
obs = generate_observations(seq, stats, CaseControlSampler(n_controls=10, seed=1))
fit = fit_rem(obs, [name(s) for s in stats])
fit.converged                # true
coeftable(fit)               # the shared Networks.CoefficientTable
```
"""
# `is_event` as a Bool vector: `Bool` columns pass through, 0/1 numeric ones
# are converted, anything else is refused by name.
_event_indicator(y::AbstractVector{Bool}) = y
function _event_indicator(y::AbstractVector)
    all(v -> v isa Real && (v == 0 || v == 1), y) || throw(ArgumentError(
        "`is_event` must be a Bool column (or 0/1 integers: 1 for the case row " *
        "of each stratum, 0 for its controls); got eltype $(eltype(y))"))
    return Vector{Bool}(y)
end

# Cases and rows per stratum in one pass. A function of its own so that the
# loop specialises on the concrete column types (the caller holds them as
# `AbstractVector`); allocates O(strata) Dict storage, never per row.
function _stratum_counts(strata::AbstractVector{S}, y::AbstractVector{Bool}) where S
    case_counts = Dict{S, Int}()
    row_counts = Dict{S, Int}()
    for (s, yi) in zip(strata, y)
        case_counts[s] = get(case_counts, s, 0) + (yi ? 1 : 0)
        row_counts[s] = get(row_counts, s, 0) + 1
    end
    return case_counts, row_counts
end

# `:bootstrap` is in the family `se=` vocabulary but not in REM's: the
# draw-to-draw spread of a sampled fit is a sensitivity diagnostic, not a
# standard error (see `control_draw_cov`). A bespoke pointer, then the ONE
# shared validator for the rest.
function _check_rem_se(se::Symbol, context::String)
    se === :bootstrap && throw(ArgumentError(
        "$context: se=:bootstrap is not offered. The inverse-Hessian standard " *
        "errors of the sampled partial likelihood are already a consistent " *
        "variance estimator under case-control sampling; what redrawing the " *
        "controls measures is how far the POINT ESTIMATE of a misspecified " *
        "model depends on the draw, which is a sensitivity diagnostic, not a " *
        "standard error. Call `control_draw_cov(seq, stats; n_controls=..., " *
        "n_draws=...)` for it, or use `se=:sandwich`."))
    check_se(se, (:hessian, :sandwich); context=context)
    return se
end

# No statistics, no model: say so before an internal consistency check does
_require_statistics(n::Int) =
    n >= 1 || throw(ArgumentError(
        "fit_rem needs at least one statistic (e.g. `[Repetition(), " *
        "Reciprocity()]`); got an empty statistics list"))

function fit_rem(observations::DataFrame, stat_names::Vector{String};
                 maxiter::Int=100, tol::Float64=1e-8, se::Symbol=:hessian)
    _check_rem_se(se, "fit_rem(::DataFrame)")
    _require_statistics(length(stat_names))
    # Validate input
    missing_cols = setdiff(stat_names, names(observations))
    if !isempty(missing_cols)
        available = setdiff(names(observations), _BOOKKEEPING_COLUMNS)
        throw(ArgumentError(
            "Statistic column$(length(missing_cols) == 1 ? "" : "s") not found in " *
            "observations: $(join(repr.(missing_cols), ", ")). The statistic " *
            "columns available are: $(join(repr.(available), ", ")) — pass " *
            "`[name(s) for s in stats]` (the names of the statistics the " *
            "observations were generated with)."))
    end
    for col in ("is_event", "stratum")
        col in names(observations) ||
            throw(ArgumentError("observations must have an `$col` column (from generate_observations)"))
    end

    # Extract data. A hand-built design naturally writes `is_event` as 0/1:
    # accept that, and refuse anything else with a message that names the
    # column (not an internal MethodError from the stratum counter)
    X = Matrix{Float64}(observations[!, stat_names])
    y = _event_indicator(observations.is_event)
    strata = observations.stratum
    eltype(strata) <: Integer || throw(ArgumentError(
        "`stratum` must be an integer column (one id per event, shared by the " *
        "case and its controls); got eltype $(eltype(strata))"))

    # Each stratum must contain exactly one case and at least one control
    # (single pass over the rows, behind a function barrier: `df.col` is
    # `AbstractVector`-typed here, and iterating it in this scope dispatched
    # dynamically on every row — 10 allocations and 250 B per row)
    case_counts, row_counts = _stratum_counts(strata, y)
    for (s, n_cases) in case_counts
        n_cases == 1 ||
            throw(ArgumentError("Stratum $s has $n_cases cases; each stratum must have exactly one"))
        row_counts[s] >= 2 ||
            throw(ArgumentError("Stratum $s has no controls; each stratum must have at least one"))
    end

    n_obs = nrow(observations)
    n_events = sum(y)

    # Risk-set bookkeeping (present when the observations came from
    # `generate_observations`): one entry per stratum, in sorted stratum order
    uniq_strata = sort!(collect(Int, keys(case_counts)))
    has_bookkeeping = "risk_set_size" in names(observations) &&
                      "sampling_prob" in names(observations)
    if has_bookkeeping
        size_by_stratum = Dict(zip(strata, observations.risk_set_size))
        prob_by_stratum = Dict(zip(strata, observations.sampling_prob))
        risk_set_sizes = [size_by_stratum[s] for s in uniq_strata]
        sampling_probs = [prob_by_stratum[s] for s in uniq_strata]
    else
        risk_set_sizes = Int[]
        sampling_probs = Float64[]
    end

    # Tie handling, as it was applied when the design was BUILT: the Efron
    # denominator weights ride in the `tie_weight` column, and the policy that
    # produced them in the DataFrame's metadata (`:none` for a design with no
    # ties in it, and for a hand-built frame that carries neither).
    tie_weights = "tie_weight" in names(observations) ? observations.tie_weight : nothing
    tie_type = Symbol(metadata(observations, "tie_method", "none"; style=false))
    tie_type in (:none, :ordered, :breslow, :efron) || throw(ArgumentError(
        "observations carry an unknown \"tie_method\" metadata value :$tie_type"))
    # The Efron weights ARE the Efron correction. A design that says it carries
    # them and does not would be fitted unweighted (i.e. Breslow) while reporting
    # `tie_method == :efron` — the one thing the metadata protocol exists to
    # prevent. Refuse it.
    (tie_type === :efron && isnothing(tie_weights)) && throw(ArgumentError(
        "observations are marked as Efron-corrected (`tie_method` metadata) but " *
        "carry no `tie_weight` column: the weights ARE the correction, so this " *
        "design would be fitted unweighted (that is Breslow) while claiming " *
        "Efron. Regenerate with `generate_observations(...; ties=:efron)`."))

    # Fit stratified conditional logistic regression (equivalent to Cox model for case-control)
    result = _fit_stratified_clogit(X, y, strata; maxiter=maxiter, tol=tol, se=se,
                                    tie_weights=tie_weights)

    # An unconverged fit is a LOUD result in this ecosystem: the numbers are
    # returned (they are what the user has to diagnose with) but never silently
    result.converged || @warn(
        "fit_rem: Newton–Raphson did not converge in $(result.iterations) of " *
        "maxiter = $maxiter iterations (final gradient norm " *
        "$(round(result.grad_norm, sigdigits=3)); tol = $tol requires < " *
        "$(round(sqrt(tol), sigdigits=3))). The estimates and standard errors " *
        "are NOT maximum-partial-likelihood values: increase `maxiter`, check " *
        "the statistics for separation (a coefficient running away) or " *
        "collinearity, or simplify the model. `Networks.approximations(fit)` " *
        "records this and `Networks.is_exact(fit)` is false.")
    # Name the identification failure separately: singular information also
    # makes Newton report nonconvergence, while separation can still satisfy
    # its objective stopping rule.
    singular_suspects = stat_names[result.singular_columns]
    result.singular && @warn(
        "fit_rem: the observed information at the solution is singular, so the " *
        "standard errors are undefined (NaN). Suspect statistics: " *
        _name_list(singular_suspects) * " — collinear (e.g. `NodeSum(x)` with " *
        "both `SenderAttribute(x)` and `ReceiverAttribute(x)`), constant within " *
        "every stratum, or separated; drop or combine them. " *
        "`Networks.approximations(fit)` records this and `Networks.is_exact(fit)` " *
        "is false.")
    separated = stat_names[result.separated_columns]
    isempty(separated) || @warn(
        "fit_rem: the coefficient on " * _name_list(separated) * " may be " *
        "infinite (separation): the log-likelihood converged but the Newton " *
        "increment on that coordinate is still O(1), so in every stratum the " *
        "case sits on one side of its controls on that statistic. The estimate " *
        "and its standard error are not meaningful; remove or transform the " *
        "statistic. `Networks.approximations(fit)` records this and " *
        "`Networks.is_exact(fit)` is false.")

    return REMResult(
        result.coefficients,
        result.std_errors,
        result.z_values,
        result.p_values,
        stat_names,
        n_events,
        n_obs,
        result.log_likelihood,
        result.converged,
        uniq_strata,
        risk_set_sizes,
        sampling_probs,
        se,
        tie_type,
        result.var_cov,
        result.iterations,
        result.singular,
        singular_suspects,
        separated
    )
end

"""
    fit_rem(seq::EventSequence, stats; kwargs...) -> REMResult

Fit a relational event model directly from an event sequence. This is the
canonical entry point of the package (the `fit_<model>` name of the ecosystem
vocabulary); there is deliberately no `rem` alias — see the package notes on
why (`rem` is `Base.rem`, the remainder function).

# Arguments
- `seq::EventSequence`: The event sequence
- `stats`: Statistics to include in the model (a `StatisticSet`, a vector of
  statistics, or a single statistic — `fit_rem(seq, Repetition())` is a
  one-element model). The sequence comes first; `fit_rem(stats, seq)` is
  refused with the order spelled out

# Keyword Arguments
- `n_controls::Int=100`: Number of controls per case
- `decay::Float64=0.0`: Exponential decay rate — eventnet's halflife memory
  model (`halflife_to_decay(h)`), the only one eventnet offers
- `window=nothing`: Sliding window — events older than `current_time −
  window` stop counting in every count, degree and adjacency set. A `Real` in
  the clock's units (seconds for `Date`/`DateTime` clocks) or, on a calendar
  clock, a `Dates.Period` such as `Day(2)` (converted to seconds). A
  `Period` on a numeric clock is refused (`ArgumentError`): the clock counts
  in its own units, not seconds, so `Day(2)` has no meaning there. The
  fixed-memory alternative to decay, **mutually exclusive with `decay > 0`**
  (`ArgumentError`); `Inf` is no window. See [`EventNetworkState`](@ref)
- `exclude_self_loops::Bool=true`: Exclude self-loops from the risk set. A
  self-loop *event* in the sequence is then refused at fit time (it cannot be
  a member of its own risk set); drop such events or pass `false`
- `at_risk=nothing`: The risk set (see `generate_observations`). One of `nothing`
  (the sequence's actor universe), an `ActorSet`/`Set{Int}`/`Vector{Int}` (static
  universe), a `RiskSet`, a vector of per-event risk sets, or a callback
  `(event_index, state) -> RiskSet`. `riskset` is accepted as an alias.
- `seed::Union{Int,Nothing}=nothing`: Random seed for the control draw. When
  given it takes precedence over `rng` for the control draw (a local
  `Xoshiro(seed)` is used, so the same `seed` gives the same controls whatever
  `rng` is)
- `ties::Symbol=:error`: How to handle tied event times (below)
- `maxiter::Int=100`: Maximum Newton–Raphson iterations
- `tol::Float64=1e-8`: Convergence tolerance
- `se::Symbol=:hessian`: How to compute the standard errors (below)
- `rng::AbstractRNG=Random.default_rng()`: Source of the control draw when
  `seed === nothing`. A fixed `rng` reproduces the fit exactly.

The actor universe determines the estimand: the likelihood is conditional on the
risk set, so nonparticipants that are eligible but never observed must be part of
it. Declare it with `EventSequence(events; actors=...)` or pass `at_risk`. When
neither is given, the universe falls back to the observed event endpoints (a
"participants only" risk set) and a warning is issued.

Every case is validated against its own risk set before fitting, and each stratum
must admit at least one control.

# Standard errors

- `se=:hessian` (default) — the inverse negative Hessian (observed
  information) of the partial likelihood on the risk set that was used. With
  the full risk set this is the classical Cox-model variance. With a
  **sampled** risk set it is the observed information of the *sampled*
  partial likelihood, which nested case-control theory shows to be a
  consistent estimator of the estimator's variance (Goldstein & Langholz
  1992; Borgan, Goldstein & Langholz 1995): the information lost by sampling
  fewer controls is what makes it larger than the full-risk-set one, and
  nothing further needs adding. Under a correctly specified model its Wald
  intervals are calibrated (the "Sampled-risk-set standard errors are
  calibrated" testset pins 95 % coverage over replicate sequences).
- `se=:sandwich` — the **event-clustered Godambe sandwich** `H⁻¹ B H⁻¹`, with
  the bread `H` the observed information and the meat `B = Σ_e u_e u_eᵀ` the
  outer product of the per-event score contributions (each event is one stratum,
  so the event is the clustering unit). This drops the information-equals-
  score-variance assumption — it is robust to misspecification of the
  within-stratum conditional model. Equals `survival::coxph(..., robust=TRUE)`
  with the stratum as the cluster (pinned by `rem_clogit.toml`).

There is deliberately **no `se=:bootstrap`**. Redrawing the controls and
refitting does not expose a missing variance component — the sampled
likelihood's own information already accounts for the sampling — what it
exposes is how far the *point estimate* of a **misspecified** model depends on
the particular draw (a pseudo-true value that moves with the draw, cured by a
larger `n_controls`, not by a wider interval). That draw-to-draw spread is a
sensitivity diagnostic and is offered as one: [`control_draw_cov`](@ref).

Whichever was asked for, `vcov(fit)` is the full covariance matrix that the
reported `stderror(fit)` are the square-rooted diagonal of.

The control inclusion probabilities are on the result: see
[`sampling_probs`](@ref) and [`risk_set_sizes`](@ref). With the full risk set
(`sampling_probs` all `1.0`) the partial likelihood is the exact ordinal one.

# Collinearity and separation

Two things make a "converged" fit meaningless, and both are reported loudly
(a warning naming the statistics, an entry in `Networks.approximations`,
`Networks.is_exact == false`, a caveat under the table in `show`): a
**singular** information matrix (`fit.singular`, standard errors `NaN`;
`fit.singular_suspects` names the statistics that load on the null direction
— collinear ones such as `NodeSum(x)` next to `SenderAttribute(x)` and
`ReceiverAttribute(x)`, or one constant within every stratum), and
**separation** (`fit.separated`: a coefficient whose Newton increment is
still O(1) at convergence although the log-likelihood no longer moves —
`survival::coxph`'s "coefficient may be infinite" rule — e.g. a covariate
product that no event ever realises). The check is scale-free: it is applied
to the standardised coefficient and increment (both multiplied by the
column's standard deviation), so a covariate measured in metres and the same
one in kilometres get the same verdict.

# Tied event times (issue REM#2, review finding 12)

This is a Cox partial likelihood — one stratum per event, the case against the
risk set — so a tied timestamp is the classical Cox tie problem and the classical
vocabulary (`Networks.TIE_POLICIES`) applies. Note what a tie does *here*: the
statistics are read off the network state as it stands **before** the event, so
ordering two simultaneous events lets the one placed first enter the *statistics*
of the one placed second. That is not a tie-break, it is invented information.

- `ties=:error` (default) — a tie is named and refused. The model claims a strict
  order; if the data does not have one, the user is told.
- `ties=:ordered` — sequence order, no correction (what the package did before).
- `ties=:breslow` — the Breslow correction: the network state is frozen across
  the tie block (the tied events cannot see each other), each is a stratum, and
  all share one denominator. Equals `survival::coxph(..., ties="breslow")`.
- `ties=:efron` — the Efron correction: as Breslow, plus the `1 − (j−1)/d`
  denominator weights on the `d` tied cases. Equals
  `survival::coxph(..., ties="efron")`, R's own default, and it is the better
  approximation — prefer it.
- `ties=:batch` — refused, with a pointer: with the risk set held fixed, a
  "simultaneous batch" in an ordinal likelihood IS the Breslow correction.

On tie-free data all four produce the identical design and the identical fit
(pinned by the tests). `tie_method(fit)` reports what actually happened —
`:none` when the data had no ties — and `approximations(fit)` carries the caveat
when a correction was in fact applied.

# Convergence

The optimizer is the shared `Networks.newton_fit` (Newton–Raphson with step
halving). A fit that exhausts `maxiter` **warns**, naming the gradient norm it
stopped at, and comes back with `converged == false`, `is_exact == false` and
an entry in `Networks.approximations(fit)`; `fit.iterations` is the count of
Newton steps taken.

# Returns
- `REMResult`: Fitted model results (`se_method(fit)` reports which of the two
  standard-error estimators was used, `tie_method(fit)` what happened to ties)

# Example
```julia
using REM, Random
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
stats = [Repetition(), Reciprocity()]

fit = fit_rem(seq, stats; n_controls=10, seed=42)          # seeded control draw
fit2 = fit_rem(seq, stats; n_controls=10, rng=Xoshiro(9))  # rng-driven, reproducible
coef(fit); stderror(fit); vcov(fit); aic(fit); confint(fit)
```
"""
function fit_rem(seq::EventSequence, stats::Vector{<:AbstractStatistic}; kwargs...)
    return fit_rem(seq, StatisticSet(stats); kwargs...)
end

# A `Vector{Event}` is not an `EventSequence`: name the missing actor universe
function fit_rem(events::AbstractVector{<:Event}, args...; kwargs...)
    throw(ArgumentError(_event_vector_hint("fit_rem")))
end

const _StatisticsArg = Union{AbstractStatistic, StatisticSet, AbstractVector{<:AbstractStatistic}}

# One statistic instead of a vector (relevent's `effects="CovInt"` habit) is
# accepted as a one-element model; the arguments in the other order are
# refused with the order spelled out — never a bare MethodError with a
# "Closest candidates" dump.
function fit_rem(seq::EventSequence, stat::AbstractStatistic; kwargs...)
    return fit_rem(seq, StatisticSet([stat]); kwargs...)
end
function fit_rem(stats::_StatisticsArg, seq::EventSequence; kwargs...)
    throw(ArgumentError(_swapped_arguments_hint("fit_rem")))
end

function fit_rem(seq::EventSequence, stats::StatisticSet;
                 n_controls::Int=100, decay::Float64=0.0,
                 window::_WindowSpec=nothing,
                 exclude_self_loops::Bool=true, seed::Union{Int,Nothing}=nothing,
                 at_risk=nothing, riskset=nothing, ties::Symbol=:error,
                 maxiter::Int=100, tol::Float64=1e-8,
                 se::Symbol=:hessian,
                 rng::AbstractRNG=Random.default_rng())
    _check_rem_se(se, "fit_rem")
    _require_statistics(length(stats))
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)
    (isnothing(at_risk) || isnothing(riskset)) ||
        throw(ArgumentError("Pass either `at_risk` or `riskset`, not both"))
    spec = isnothing(at_risk) ? riskset : at_risk

    # An actor universe read off the observed event endpoints silently excludes
    # eligible nonparticipants and changes the estimand
    if isnothing(spec) && !seq.actors_declared
        @warn "Fitting against an actor universe inferred from observed event " *
              "endpoints: isolates and other eligible nonparticipants are excluded " *
              "from the risk set, which changes the estimand. Declare the universe " *
              "with `EventSequence(events; actors=...)` or pass `at_risk` to " *
              "`fit_rem`." maxlog = 1
    end

    # The control draw: `seed` pins it (a local Xoshiro, so the same seed gives
    # the same controls whatever `rng` is); otherwise it flows from `rng`, like
    # every other random draw in the ecosystem — never from the global RNG
    # behind the caller's back.
    sampler = CaseControlSampler(n_controls=n_controls, exclude_self_loops=exclude_self_loops, seed=seed)
    observations = generate_observations(seq, stats, sampler; decay=decay, window=window,
                                         at_risk=spec, ties=ties, rng=rng)
    return fit_rem(observations, stats.names; maxiter=maxiter, tol=tol, se=se)
end

"""
    control_draw_cov(seq::EventSequence, stats; n_controls=100, n_draws=20, kwargs...)
        -> (cov, sd, replicates, mean, n_unconverged)

**Control-draw sensitivity diagnostic** for a case-control fit: redraw the
sampled risk set `n_draws` times (independent seeds from `rng`), refit
[`fit_rem`](@ref) on each draw, and return the covariance of the refitted
coefficients **across draws**. It answers one question — *how much does my
point estimate depend on which controls happened to be drawn?* — and it is
**not a standard error**: it is neither reported by `stderror` nor combined
with the inverse-Hessian covariance, because under nested case-control
sampling the sampled likelihood's own observed information is already a
consistent variance estimator (Goldstein & Langholz 1992; Borgan, Goldstein &
Langholz 1995), and adding a between-draw term to it would double-count.

What the spread *does* measure. Under a correctly specified model the
draws scatter around one value: their spread is the information the
sampling threw away, which the standard error already accounts for, and
their `mean` stays put as `n_controls` changes. Under a **misspecified**
model (every real model) the sampled fit converges to a pseudo-true value
that itself depends on the draw and moves with `n_controls`: the `mean`
drifts toward the full-risk-set estimate as the controls grow, and the
draw-to-draw `sd` is of the order of the standard error with few controls
(on the WTC calls' 7-statistic model — the estimation guide's table — 20 of
1331 controls gives `repetition` a mean of −0.50 with a draw-to-draw `sd` of
0.07–0.08 over ten draws (`Xoshiro(1)`/`Xoshiro(7)`) against a per-draw
standard error of 0.08–0.10 (`seed=1`/`seed=42`), and a single draw — seed
42 — sits at −0.72; 400 controls gives −0.33 ± 0.02 against 0.04; the full
risk set −0.34). Either symptom says the same thing — raise `n_controls`,
or enumerate the full risk set — and neither is cured by a wider interval.

# Arguments
- `seq`, `stats`: as for [`fit_rem`](@ref)

# Keyword Arguments
- `n_controls::Int=100`: controls per event in each draw
- `n_draws::Int=20`: number of independent control draws (at least 2)
- `decay`, `window`, `exclude_self_loops`, `at_risk`/`riskset`, `ties`,
  `maxiter`, `tol`: as for [`fit_rem`](@ref); the design of every draw is
  built with them
- `rng::AbstractRNG=Random.default_rng()`: the source of the per-draw seeds
  (drawn up front, so the result is reproducible for a fixed `rng` and
  independent of the thread count)
- `threaded::Bool=true`: refit the draws on all available threads

# Returns
A NamedTuple:
- `cov::Matrix{Float64}`: the `p × p` covariance of the refitted coefficients
  across draws (centred on their mean — the spread, not the bias)
- `sd::Vector{Float64}`: its square-rooted diagonal, the draw-to-draw
  standard deviation of each coefficient
- `replicates::Matrix{Float64}`: the `n_draws × p` refitted coefficients, in
  `name.(stats)` order
- `mean::Vector{Float64}`: their mean — compare it with the estimate at a
  larger `n_controls` or with the full-risk-set fit
- `n_unconverged::Int`: how many refits did not converge (warned about once)

Runs on the shared `Networks.bootstrap_cov` resampling loop; what is
resampled is the sampling design, not the model.

# Example
```julia
using REM, Random
rng0 = MersenneTwister(4)
events = [Event(rand(rng0, 1:12), rand(rng0, 1:12), Float64(k)) for k in 1:80]
events = [e for e in events if e.sender != e.receiver]
seq = EventSequence(events; actors=ActorSet(1:12))
stats = [Repetition(), Reciprocity()]
fit = fit_rem(seq, stats; n_controls=5, seed=42)
draws = control_draw_cov(seq, stats; n_controls=5, n_draws=10, rng=Xoshiro(7))
size(draws.replicates)                  # (10, 2)
draws.sd                                # draw-to-draw sd; compare with stderror(fit)
draws.cov == control_draw_cov(seq, stats; n_controls=5, n_draws=10, rng=Xoshiro(7)).cov  # true
```
"""
function control_draw_cov(seq::EventSequence, stats::Vector{<:AbstractStatistic}; kwargs...)
    return control_draw_cov(seq, StatisticSet(stats); kwargs...)
end

function control_draw_cov(events::AbstractVector{<:Event}, args...; kwargs...)
    throw(ArgumentError(_event_vector_hint("control_draw_cov")))
end

function control_draw_cov(seq::EventSequence, stat::AbstractStatistic; kwargs...)
    return control_draw_cov(seq, StatisticSet([stat]); kwargs...)
end
function control_draw_cov(stats::_StatisticsArg, seq::EventSequence; kwargs...)
    throw(ArgumentError(_swapped_arguments_hint("control_draw_cov")))
end

function control_draw_cov(seq::EventSequence, stats::StatisticSet;
                          n_controls::Int=100, n_draws::Int=20,
                          decay::Float64=0.0, window::_WindowSpec=nothing,
                          exclude_self_loops::Bool=true,
                          at_risk=nothing, riskset=nothing, ties::Symbol=:error,
                          maxiter::Int=100, tol::Float64=1e-8,
                          rng::AbstractRNG=Random.default_rng(),
                          threaded::Bool=true)
    _require_statistics(length(stats))
    n_draws >= 2 ||
        throw(ArgumentError("n_draws must be at least 2 to form a covariance " *
                            "(got $n_draws)"))
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)
    (isnothing(at_risk) || isnothing(riskset)) ||
        throw(ArgumentError("Pass either `at_risk` or `riskset`, not both"))
    spec = isnothing(at_risk) ? riskset : at_risk
    p = length(stats)
    unconverged = zeros(Bool, n_draws)

    # One seed per draw, drawn up front from `rng` — the ecosystem's rng
    # discipline, and what makes the threaded refits thread-count-independent
    function simulate(rng, B)
        seeds = rand(rng, 1:typemax(Int), B)
        return [(b, generate_observations(
                        seq, stats,
                        CaseControlSampler(n_controls=n_controls,
                                           exclude_self_loops=exclude_self_loops,
                                           seed=seeds[b]);
                        decay=decay, window=window, at_risk=spec, ties=ties))
                for b in 1:B]
    end

    function refit(replicate)
        b, obs = replicate
        X = Matrix{Float64}(obs[!, stats.names])
        fit = _fit_stratified_clogit(X, obs.is_event, obs.stratum;
                                     maxiter=maxiter, tol=tol,
                                     tie_weights=obs.tie_weight)
        unconverged[b] = !fit.converged
        return fit.coefficients
    end

    boot = bootstrap_cov(refit, simulate, zeros(p); n_boot=n_draws, rng=rng,
                         threaded=threaded)

    n_bad = count(unconverged)
    n_bad == 0 || @warn(
        "control_draw_cov: $n_bad of $n_draws refits did not converge in " *
        "maxiter = $maxiter iterations; their coefficients are not " *
        "maximum-partial-likelihood values. Increase `maxiter`.")

    return (cov=boot.vcov, sd=boot.se, replicates=boot.replicates,
            mean=vec(sum(boot.replicates; dims=1) ./ n_draws), n_unconverged=n_bad)
end

# ============================================================================
# The stratified conditional logit
# ============================================================================
#
# One stratum per event: the case dyad against its controls. The per-stratum
# softmax kernel below (`_clogit_derivatives!`) is REM's own — it is NOT a
# logistic likelihood, so nothing of it belongs in `Networks.logistic_derivatives`
# — but the OPTIMIZER that drives it is the ecosystem's one `Networks.newton_fit`
# (panel 2026-09, item 14): REM hosts no Newton loop, no step halving and no
# Hessian inversion of its own.

"""
    _StrataIndex

Compressed-sparse-row index of the observation rows by stratum, built in ONE
pass over the `stratum` column (panel 2026-09, item 26: the old
`Dict(s => findall(==(s), strata))` was O(strata × rows), 0.9 s at 20k strata):

- `offsets[s]:offsets[s+1]-1` is the block of `order` holding the row indices of
  stratum `s`;
- `case[s]` is the 1-based position of the case row within that block (`0` when
  the stratum has no case; such strata are skipped by the kernels);
- `max_size` sizes the per-stratum workspace.

Strata are numbered in ascending stratum id when the ids are dense (a counting
sort; the common case, one id per event) and in order of first appearance
otherwise, so the summation order of the likelihood is deterministic — a `Dict`
iteration order was not.
"""
struct _StrataIndex
    offsets::Vector{Int}
    order::Vector{Int}
    case::Vector{Int}
    max_size::Int
end

_n_strata(idx::_StrataIndex) = length(idx.offsets) - 1

function _strata_index(strata::Vector{Int}, y::Vector{Bool})
    n = length(strata)
    length(y) == n || throw(ArgumentError(
        "is_event has $(length(y)) entries for $n observation rows"))
    n == 0 && return _StrataIndex([1], Int[], Int[], 0)

    # Slot of each row's stratum: a counting sort over the id range when it is
    # dense (ascending id order), a first-appearance Dict otherwise
    lo, hi = extrema(strata)
    span = hi - lo + 1
    row_slot = Vector{Int}(undef, n)
    if span <= 2 * n + 64
        present = zeros(Int, span)
        @inbounds for s in strata
            present[s - lo + 1] = 1
        end
        cumsum!(present, present)           # present[id] == slot for present ids
        n_strata = present[end]
        @inbounds for i in 1:n
            row_slot[i] = present[strata[i] - lo + 1]
        end
    else
        slot_of = Dict{Int, Int}()
        @inbounds for i in 1:n
            row_slot[i] = get!(slot_of, strata[i]) do
                length(slot_of) + 1
            end
        end
        n_strata = length(slot_of)
    end

    counts = zeros(Int, n_strata)
    @inbounds for k in row_slot
        counts[k] += 1
    end
    offsets = Vector{Int}(undef, n_strata + 1)
    offsets[1] = 1
    @inbounds for k in 1:n_strata
        offsets[k + 1] = offsets[k] + counts[k]
    end
    next = offsets[1:n_strata]
    order = Vector{Int}(undef, n)
    case = zeros(Int, n_strata)
    @inbounds for i in 1:n
        k = row_slot[i]
        pos = next[k]
        order[pos] = i
        next[k] = pos + 1
        if y[i] && case[k] == 0
            case[k] = pos - offsets[k] + 1
        end
    end
    return _StrataIndex(offsets, order, case, maximum(counts))
end

# Internal: preallocated per-stratum buffers for `_clogit_derivatives!` and the
# sandwich meat, sized for the largest stratum. Allocated ONCE per fit; every
# derivative evaluation after that allocates nothing.
function _clogit_workspace(idx::_StrataIndex, p::Int)
    max_ns = idx.max_size
    return (
        Xs = Matrix{Float64}(undef, max_ns, p),      # stratum design matrix
        Xw = Matrix{Float64}(undef, max_ns, p),      # sqrt(prob)-weighted rows
        eta = Vector{Float64}(undef, max_ns),        # linear predictor
        probs = Vector{Float64}(undef, max_ns),      # softmax probabilities
        xexp = Vector{Float64}(undef, p),            # E[X] within stratum
        hess_s = Matrix{Float64}(undef, p, p),       # stratum Hessian
        hess_c = Matrix{Float64}(undef, p, p),       # summation correction
    )
end

# A common covariate shift within a stratum cancels from its conditional
# likelihood, including weighted Efron denominators. Center on the case before
# taking moments: a constant column then has exactly zero score and curvature,
# instead of cancellation noise that can masquerade as identifiable information.
@inline function _center_stratum!(work, X::Matrix{Float64}, idx::_StrataIndex,
                                  s::Int)
    lo = idx.offsets[s]
    n_s = idx.offsets[s + 1] - lo
    case_row = idx.order[lo + idx.case[s] - 1]
    X_s = view(work.Xs, 1:n_s, :)
    @inbounds for k in axes(X, 2), a in 1:n_s
        X_s[a, k] = X[idx.order[lo + a - 1], k] - X[case_row, k]
    end
    return X_s
end

"""
    _clogit_derivatives!(grad, hess, X, idx, beta, work, tw) -> ll

The per-stratum softmax kernel of the stratified conditional logit: overwrites
`grad` and `hess` with the gradient and Hessian of the log partial likelihood at
`beta` and returns its value. **Allocation-free** after warm-up (pinned by the
"Allocation-free clogit kernel" testset): every buffer lives in `work`
(`_clogit_workspace`). BLAS forms each stratum's negative covariance from
sqrt-probability-weighted, mean-centered rows; compensated summation combines
the stratum Hessians. There are no per-row outer products or per-stratum
allocations. Covariates are first centered on the case before the softmax and
moment calculations, preserving exact zeros for constant columns.

`tw` holds the per-row DENOMINATOR weights (`tie_weight`): the stratum's
normalizing sum is `Σ_a tw_a·exp(η_a)`, while the numerator stays `exp(η_case)`.
They are 1 everywhere except under the Efron tie correction, so the weighted
likelihood REDUCES to the unweighted one on tie-free data — which is why a tie
correction is a no-op there.
"""
function _clogit_derivatives!(grad::Vector{Float64}, hess::Matrix{Float64},
                              X::Matrix{Float64}, idx::_StrataIndex,
                              beta::AbstractVector{Float64}, work,
                              tw::Vector{Float64})
    p = size(X, 2)
    fill!(grad, 0.0)
    fill!(hess, 0.0)
    fill!(work.hess_c, 0.0)
    # The log-likelihood is a sum over strata whose partial sums reach the
    # hundreds while a Newton step near the optimum improves it by ~1e-13:
    # accumulate it with Neumaier (compensated) summation, so that the value
    # `newton_fit` compares between iterates carries only the per-term rounding
    # and never refuses a genuine improvement as a rounding-level decrease.
    ll = 0.0
    ll_c = 0.0
    xexp = work.xexp
    order = idx.order
    offsets = idx.offsets

    @inbounds for s in 1:_n_strata(idx)
        case_pos = idx.case[s]
        case_pos == 0 && continue            # no case: nothing to condition on
        lo = offsets[s]
        n_s = offsets[s + 1] - lo

        X_s = _center_stratum!(work, X, idx, s)

        # Linear predictor
        eta = view(work.eta, 1:n_s)
        mul!(eta, X_s, beta)

        # Numerical stability: subtract max
        eta_max = maximum(eta)
        probs = view(work.probs, 1:n_s)
        sum_exp_eta = 0.0
        for a in 1:n_s
            probs[a] = tw[order[lo + a - 1]] * exp(eta[a] - eta_max)
            sum_exp_eta += probs[a]
        end

        # Log-likelihood contribution (compensated)
        term = eta[case_pos] - eta_max - log(sum_exp_eta)
        t = ll + term
        ll_c += abs(ll) >= abs(term) ? (ll - t) + term : (term - t) + ll
        ll = t

        # Probabilities
        probs ./= sum_exp_eta

        # Gradient contribution: X_case - E[X]
        mul!(xexp, transpose(X_s), probs)
        for k in 1:p
            grad[k] += X_s[case_pos, k] - xexp[k]
        end

        # Form -Var[X] from centered rows, avoiding the subtraction of two
        # large moments. Compensate the sum across strata so accumulated
        # rounding cannot manufacture curvature along a collinear direction.
        Xw = view(work.Xw, 1:n_s, :)
        for a in 1:n_s
            probs[a] = sqrt(probs[a])
        end
        for k in 1:p, a in 1:n_s
            Xw[a, k] = probs[a] * (X_s[a, k] - xexp[k])
        end
        mul!(work.hess_s, transpose(Xw), Xw, -1.0, 0.0)
        for k in 1:p, j in 1:p
            increment = work.hess_s[j, k] - work.hess_c[j, k]
            updated = hess[j, k] + increment
            work.hess_c[j, k] = (updated - hess[j, k]) - increment
            hess[j, k] = updated
        end
    end

    return ll + ll_c
end

# The `(ll, grad, hess)` closure `Networks.newton_fit` drives. The workspace is
# captured once; each evaluation allocates exactly the length-p gradient and the
# p×p Hessian it hands back (the same contract as `Networks.logistic_derivatives`,
# pinned by the allocation testset).
function _clogit_objective(X::Matrix{Float64}, idx::_StrataIndex, work,
                           tw::Vector{Float64})
    p = size(X, 2)
    return function (beta)
        grad = Vector{Float64}(undef, p)
        hess = Matrix{Float64}(undef, p, p)
        ll = _clogit_derivatives!(grad, hess, X, idx, beta, work, tw)
        return ll, grad, hess
    end
end

# Internal: fit the stratified conditional logistic regression with the shared
# `Networks.newton_fit`. `se` selects the covariance estimator applied at the
# final β: `:hessian` (the inverse observed information — `newton_fit`'s own
# Cholesky-based `vcov`, NaN with a warning when −H is not positive definite)
# or `:sandwich` (the event-clustered Godambe sandwich, see
# `_clogit_sandwich_cov`). The POINT ESTIMATE does not depend on it.
#
# `tie_weights` are the per-row DENOMINATOR weights (the `tie_weight` column):
# all 1 unless the Efron tie correction put fractional weights on the cases tied
# at one timestamp. With all-ones weights this is bit-for-bit the likelihood it
# always was.
#
# Returns a NamedTuple with `var_cov` MATCHING `se`, `info_cov` (the inverse
# observed information regardless of `se`), `converged`, `iterations`,
# `grad_norm` (the gradient norm at the final iterate, NaN when converged —
# only computed to make a non-convergence warning informative), and the two
# identification diagnostics: `singular` with `singular_columns` (the observed
# information is not positive definite or numerically identifiable, and the
# shared optimizer reports nonconvergence), and `separated_columns` (coefficients
# whose Newton increment is still O(1) at convergence — `survival::coxph`'s
# "coefficient may be infinite" rule).
function _fit_stratified_clogit(X::Matrix{Float64}, y::AbstractVector{Bool},
                                 strata::AbstractVector{Int};
                                 maxiter::Int=100, tol::Float64=1e-8,
                                 se::Symbol=:hessian,
                                 tie_weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    X = Matrix{Float64}(X)
    y = Vector{Bool}(y)
    strata = Vector{Int}(strata)
    n, p = size(X)
    tw = isnothing(tie_weights) ? ones(n) : Vector{Float64}(tie_weights)
    length(tw) == n || throw(ArgumentError(
        "tie_weights has $(length(tw)) entries for $n observation rows"))
    all(w -> w > 0, tw) || throw(ArgumentError(
        "tie weights must be positive (they multiply exp(η) in the stratum " *
        "denominator); got a non-positive one"))

    # Single-pass CSR index of the rows by stratum, and the per-stratum work
    # buffers shared across all derivative evaluations
    idx = _strata_index(strata, y)
    work = _clogit_workspace(idx, p)
    objective = _clogit_objective(X, idx, work, tw)

    # The ONE Newton–Raphson of the ecosystem: step halving, the combined
    # |Δll| / gradient-norm stopping rule, and the Cholesky-based covariance
    # (NaN + warning when −H is not positive definite) all live there
    # Each stratum covariance contains dot products of up to max_size rows.
    # Their rounding error can exceed the p-dimensional eigensolve's default
    # tolerance, even with compensated accumulation across strata. Tell the
    # shared rank guard the precision of this assembled information matrix.
    information_rtol = max(p, idx.max_size) * eps(Float64)
    fit = newton_fit(objective, zeros(p); maxiter=maxiter, tol=tol,
                     information_rtol=information_rtol)

    info_cov = fit.vcov                       # inverse observed information
    var_cov = se === :sandwich ?
        _clogit_sandwich_cov(X, idx, fit.θ, info_cov, tw, work) : info_cov
    # `max(NaN, 0) === NaN`: an undefined covariance stays undefined
    std_errors = sqrt.(max.(diag(var_cov), 0.0))
    # The ONE z → p helper (floored, NaN-aware) — never a local 2·ccdf copy
    zp = z_pvalues(fit.θ, std_errors)

    # One more kernel pass at the solution, for the diagnostics: the gradient
    # norm (non-convergence), the null direction of the information matrix
    # (singularity) and the Newton increment (separation)
    _, grad, hess = objective(fit.θ)
    grad_norm = fit.converged ? NaN : norm(grad)
    singular = !all(isfinite, info_cov)
    singular_columns = singular ? _singular_columns(hess) : Int[]
    # The separation rule is applied to the STANDARDISED coefficient and
    # increment (β·s, δ·s with `s` the column's sd): both rescale together
    # with the column, so the verdict is the same whether a covariate is
    # measured in metres or kilometres — coxph's test is scale-free for the
    # same reason, and the raw-scale version silently passed a separated
    # 0/1000 covariate as a finite β ≈ −2e-5 with a tidy standard error.
    scales = (fit.converged && !singular) ? _column_scales(X) : Float64[]
    separated_columns = (fit.converged && !singular) ?
        _separated_columns(fit.θ .* scales, (info_cov * grad) .* scales) : Int[]

    return (
        coefficients = fit.θ,
        std_errors = std_errors,
        z_values = zp.z,
        p_values = zp.p,
        log_likelihood = fit.loglik,
        converged = fit.converged,
        iterations = fit.iterations,
        var_cov = var_cov,
        info_cov = info_cov,
        grad_norm = grad_norm,
        singular = singular,
        singular_columns = singular_columns,
        separated_columns = separated_columns,
    )
end

# The columns that load on the null direction(s) of the observed information
# `-hess`: the eigenvectors whose eigenvalue is at most 1e-8 of the largest
# (or non-positive), thresholded at a tenth of their largest loading. With
# `NodeSum(x)`, `SenderAttribute(x)`, `ReceiverAttribute(x)` the null vector
# is (1, -1, -1)/√3 and all three are named; a column constant within every
# stratum has a zero diagonal and is named alone.
function _singular_columns(hess::Matrix{Float64})
    p = size(hess, 1)
    all(isfinite, hess) || return collect(1:p)
    E = eigen(Symmetric(-hess))
    λmax = maximum(E.values)
    cutoff = λmax > 0 ? 1e-8 * λmax : 0.0
    cols = Set{Int}()
    for k in 1:p
        E.values[k] <= cutoff || continue
        v = abs.(E.vectors[:, k])
        vmax = maximum(v)
        for j in 1:p
            v[j] > 0.1 * vmax && push!(cols, j)
        end
    end
    isempty(cols) && return collect(1:p)      # singular yet no direction found: name all
    return sort!(collect(cols))
end

# `survival::coxph`'s "coefficient may be infinite" rule, applied to the
# standardised coefficient and increment (see `_column_scales`): at a
# converged solution the Newton increment `δ = H⁻¹ g` is ~1e-15 on a healthy
# coordinate (Newton converges quadratically, so the last accepted step
# overshoots the tolerance by orders of magnitude) and O(1) on a separated
# one — for an exponentially flat log-likelihood `-c·exp(-β)` the increment
# is exactly 1 however far `β` has run (measured: -1.0000 on the WTC
# `NodeProduct(icr)` at β = -20, 1e-15 on every healthy coefficient). Flag
# `|δ_j| > 0.01·max(1, |β_j|)`: the worst case at `newton_fit`'s stopping rule
# (`‖g‖ < sqrt(tol)`) is `|δ_j| ≤ SE_j²·sqrt(tol)`, so a finite maximum can
# only trip it with a standard error above 10/tol^(1/4) ≈ 100 at the default
# tol — a coefficient that is not identified either way.
function _separated_columns(θ::Vector{Float64}, δ::Vector{Float64})
    return [j for j in eachindex(θ) if isfinite(δ[j]) && abs(δ[j]) > 0.01 * max(1.0, abs(θ[j]))]
end

# The per-column standard deviations of the design, 1.0 where a column does
# not vary (or the design has a single row): multiplying a coefficient and its
# Newton increment by them gives the standardised-covariate quantities
# (β·s, δ·s), which do not change when a column is rescaled — so neither
# does `_separated_columns`' verdict, and the "standard error of ~100"
# a finite maximum needs to trip it is a standardised one, unit-free.
function _column_scales(X::Matrix{Float64})
    n, p = size(X)
    return [let v = n > 1 ? std(view(X, :, j)) : NaN
                (isfinite(v) && v > 0) ? v : 1.0
            end for j in 1:p]
end

"""
    _clogit_sandwich_cov(X, idx, beta, bread, tw, work) -> Matrix{Float64}

Event-clustered (Godambe) sandwich covariance for the stratified conditional
logit: `V = H⁻¹ B H⁻¹`, with

- the **bread** `H⁻¹ = bread`, the inverse observed information at β̂ (as
  `Networks.newton_fit` returns it), and
- the **meat** `B = Σ_e u_e u_eᵀ`, the outer product of the per-event score
  contributions `u_e = x_case(e) − E_e[X]`.

Each event is exactly one stratum (one case against its sampled controls), so the
event IS the clustering unit and no `cluster` argument is needed.

The default inverse-Hessian standard errors are `H⁻¹`, which is `V` only under
the information equality `B = H` — i.e. only if the within-stratum conditional
model is correctly specified. The sandwich drops that assumption; it is what
`coxph(..., robust = TRUE)` reports in R **with the stratum as the cluster**
(`clogit(y ~ x + strata(s) + cluster(s), method = "breslow")` — one event per
stratum, so Breslow is the exact likelihood there; without `cluster(s)` R's
robust variance is the per-row dfbeta sandwich, a different number). Pinned
by `robust_std_errors` in `test/fixtures/rem_clogit.toml` (agreement 4e-16).

At the optimum `Σ_e u_e = 0` (the gradient vanishes), so no centring is applied.
The stratum loop reuses the fit's workspace (`work.eta`, `work.probs`) rather
than allocating per stratum; a NaN bread (−H not positive definite) gives a NaN
covariance rather than a finite number.
"""
function _clogit_sandwich_cov(X::Matrix{Float64}, idx::_StrataIndex,
                              beta::Vector{Float64}, bread::Matrix{Float64},
                              tw::Vector{Float64}, work)
    p = size(X, 2)
    all(isfinite, bread) || return fill(NaN, p, p)
    meat = zeros(p, p)
    u = Vector{Float64}(undef, p)
    xexp = work.xexp
    order = idx.order
    offsets = idx.offsets

    @inbounds for s in 1:_n_strata(idx)
        case_pos = idx.case[s]
        case_pos == 0 && continue
        lo = offsets[s]
        n_s = offsets[s + 1] - lo

        # Softmax probabilities within the stratum (same log-sum-exp, and the
        # same denominator weights, as the likelihood — so the score cannot
        # drift from the model it scores)
        X_s = _center_stratum!(work, X, idx, s)
        eta = view(work.eta, 1:n_s)
        mul!(eta, X_s, beta)
        eta_max = maximum(eta)
        probs = view(work.probs, 1:n_s)
        total = 0.0
        for a in 1:n_s
            probs[a] = tw[order[lo + a - 1]] * exp(eta[a] - eta_max)
            total += probs[a]
        end
        probs ./= total

        # E[X] within the stratum, then the score u_e = x_case − E[X]
        mul!(xexp, transpose(X_s), probs)
        for k in 1:p
            u[k] = X_s[case_pos, k] - xexp[k]
        end

        BLAS.ger!(1.0, u, u, meat)
    end

    return bread * meat * bread           # Godambe sandwich
end

# ============================================================================
# The StatsAPI surface (panel 2026-09, item 15)
# ============================================================================
#
# Every verb is a method on the StatsAPI generic (imported by name in REM.jl),
# so `using REM, StatsBase` (or GLM, or Networks) dispatches ONE `coef`. The
# quantities are read straight off the result — nothing here is recomputed from
# something else, which is what keeps `stderror(fit) == sqrt.(diag(vcov(fit)))`
# true under all three `se=` estimators.

"""
    coef(result::REMResult) -> Vector{Float64}

The estimated coefficients (log hazard ratios), one per statistic in
`result.stat_names`. A method of `StatsAPI.coef`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
coef(fit)                    # 2-element Vector{Float64}
```
"""
coef(result::REMResult) = result.coefficients

"""
    stderror(result::REMResult) -> Vector{Float64}

The standard errors of the coefficients, computed the way `se_method(result)`
says (`:hessian` or `:sandwich`); always `sqrt.(diag(vcov(result)))`.
A method of `StatsAPI.stderror`.

# Example
```julia
using REM, LinearAlgebra
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
stderror(fit) ≈ sqrt.(diag(vcov(fit)))    # true
```
"""
stderror(result::REMResult) = result.std_errors

"""
    vcov(result::REMResult) -> Matrix{Float64}

The `p × p` covariance matrix of the coefficients **that matches
`se_method(result)`**: the inverse observed information for `:hessian`, the
Godambe sandwich `H⁻¹BH⁻¹` for `:sandwich` (see [`fit_rem`](@ref)). `NaN`
throughout when the Hessian at the solution is not negative definite
(`result.singular`). A method of `StatsAPI.vcov`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
size(vcov(fit))              # (2, 2)
```
"""
vcov(result::REMResult) = result.var_cov

"""
    loglikelihood(result::REMResult) -> Float64

The log partial likelihood at the returned coefficients (the field
`result.log_likelihood`): the stratified conditional-logit / Cox partial
likelihood, evaluated on the sampled risk set. A method of
`StatsAPI.loglikelihood`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition()]; n_controls=10, seed=1)
loglikelihood(fit) == fit.log_likelihood    # true
```
"""
loglikelihood(result::REMResult) = result.log_likelihood

"""
    nobs(result::REMResult) -> Int

The number of **events** (`result.n_events`): one stratum per event, so this is
what `survival::clogit` reports as `nevent` and what `bic` scales its penalty
by — not the number of case-control rows (`result.n_observations`), which grows
with `n_controls` and is a property of the sampling design rather than of the
data. A method of `StatsAPI.nobs`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition()]; n_controls=10, seed=1)
nobs(fit)                    # 8 — the events, not the 88 case-control rows
```
"""
nobs(result::REMResult) = result.n_events

"""
    dof(result::REMResult) -> Int

The number of estimated coefficients (`length(coef(result))`); the model has
no intercept and no nuisance parameters. A method of `StatsAPI.dof`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
dof(fit)                     # 2
```
"""
dof(result::REMResult) = length(result.coefficients)

"""
    aic(result::REMResult) -> Float64

Akaike's information criterion `−2ℓ + 2k` with `ℓ = loglikelihood(result)` the
log partial likelihood and `k = dof(result)`. A method of `StatsAPI.aic`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
aic(fit) ≈ -2 * loglikelihood(fit) + 2 * dof(fit)    # true
```
"""
aic(result::REMResult) = -2 * result.log_likelihood + 2 * dof(result)

"""
    bic(result::REMResult) -> Float64

The Bayesian information criterion `−2ℓ + k·log(n)` with `n = nobs(result)`
the number of events (one stratum each — the `nevent` of a Cox model, which is
what `survival` uses in `BIC(coxph)` too). A method of `StatsAPI.bic`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
bic(fit) ≈ -2 * loglikelihood(fit) + dof(fit) * log(nobs(fit))    # true
```
"""
bic(result::REMResult) = -2 * result.log_likelihood + dof(result) * log(nobs(result))

"""
    confint(result::REMResult; level::Real=0.95) -> Matrix{Float64}

Wald confidence intervals `coef ± z_{1−α/2} · stderror`, one row per
coefficient (`[:, 1]` lower, `[:, 2]` upper), on the normal reference
distribution — the same standard errors `se_method(result)` names, so the
interval is conditional on whatever they are conditional on. `level` must lie
strictly in `(0, 1)`. A method of `StatsAPI.confint`.

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
ci = confint(fit)            # 2×2 Matrix: lower / upper at 95 %
all(ci[:, 1] .<= coef(fit) .<= ci[:, 2])    # true
confint(fit; level=0.9)      # narrower
```
"""
function confint(result::REMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError(
        "confint: level must lie strictly between 0 and 1 (got $level)"))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    return hcat(result.coefficients .- q .* result.std_errors,
                result.coefficients .+ q .* result.std_errors)
end

"""
    coeftable(result::REMResult) -> Networks.CoefficientTable

The coefficient table as the ecosystem's inspectable
`Networks.CoefficientTable` — names, estimates, standard errors, z-values and
p-values exactly as `show(result)` prints them (the p-values are the fit's own,
floored and NaN-aware). Index rows by position (`tbl[1]`) or by name
(`tbl["repetition"]`), iterate them, or read the vectors off the fields. A
method of `StatsAPI.coeftable`.

Until 0.2.0 this returned a `DataFrame`; the example shows how to get one back.

# Example
```julia
using REM, DataFrames
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(3, 1, 5.0), Event(1, 3, 6.0), Event(2, 1, 7.0), Event(3, 2, 8.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=10, seed=1)
tbl = coeftable(fit)
tbl.names                    # ["repetition", "reciprocity"]
tbl["reciprocity"].estimate == coef(fit)[2]    # true
tbl                          # prints the R-style table
# The pre-0.2 DataFrame, if you need one
DataFrame(statistic=tbl.names, coefficient=tbl.estimates, std_error=tbl.std_errors,
          z_value=tbl.z_values, p_value=tbl.p_values)
```
"""
function coeftable(result::REMResult)
    return CoefficientTable(result.stat_names, result.coefficients, result.std_errors;
                            z_values=result.z_values, p_values=result.p_values)
end
