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
- `converged::Bool`: Whether the optimization converged
- `strata::Vector{Int}`: Stratum IDs, in the order of the risk-set bookkeeping below
- `risk_set_sizes::Vector{Int}`: Size of each stratum's risk set (case included);
  empty when the observations carry no risk-set bookkeeping
- `sampling_probs::Vector{Float64}`: Probability with which each non-case dyad of
  the stratum's risk set entered the sample as a control (1.0 = full risk set).
  These are the **control inclusion probabilities**; read them with
  [`sampling_probs`](@ref) and the risk-set sizes with [`risk_set_sizes`](@ref).
- `se_type::Symbol`: How `std_errors` were ACTUALLY computed — `:hessian`,
  `:sandwich` or `:bootstrap` (see [`fit_rem`](@ref)). This is what
  `Networks.se_method(fit)` reports.
- `tie_type::Symbol`: What was ACTUALLY done with tied event times — `:none`
  (the data had no ties, so no policy could bite), or the policy that did:
  `:ordered`, `:breslow` or `:efron` (see `ties=` in [`fit_rem`](@ref)).
  `:error` can never appear: under it a tie throws instead of fitting. This is
  what `Networks.tie_method(fit)` reports.
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
end

# Backwards-compatible constructors. A result built without an `se_type` reports
# the inverse-Hessian standard errors it in fact had; one built without a
# `tie_type` reports `:none` — no tie correction was applied to it, because none
# was available when it was built.
REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
          n_observations, log_likelihood, converged) =
    REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
              n_observations, log_likelihood, converged, Int[], Int[], Float64[],
              :hessian, :none)

REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
          n_observations, log_likelihood, converged, strata, risk_set_sizes,
          sampling_probs) =
    REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
              n_observations, log_likelihood, converged, strata, risk_set_sizes,
              sampling_probs, :hessian, :none)

REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
          n_observations, log_likelihood, converged, strata, risk_set_sizes,
          sampling_probs, se_type::Symbol) =
    REMResult(coefficients, std_errors, z_values, p_values, stat_names, n_events,
              n_observations, log_likelihood, converged, strata, risk_set_sizes,
              sampling_probs, se_type, :none)

"""
    sampling_probs(result::REMResult) -> Vector{Float64}

The **control inclusion probabilities**: the probability with which each non-case
dyad of a stratum's risk set entered the sample as a control, one entry per
stratum (in `result.strata` order). `1.0` means the full risk set was used (no
sampling). Empty when the fit carries no risk-set bookkeeping.

The likelihood is conditional on the sampled risk set, so these probabilities are
part of the estimand and not an implementation detail — they are what determines
whether `Networks.is_exact` holds and how much variability the inverse-Hessian
standard errors are ignoring (see `se=:bootstrap` in [`fit_rem`](@ref)).

See also [`risk_set_sizes`](@ref).
"""
sampling_probs(result::REMResult) = result.sampling_probs

"""
    risk_set_sizes(result::REMResult) -> Vector{Int}

The size of each stratum's risk set (the case included), one entry per stratum in
`result.strata` order. Empty when the fit carries no risk-set bookkeeping.

See also [`sampling_probs`](@ref) for the control inclusion probabilities.
"""
risk_set_sizes(result::REMResult) = result.risk_set_sizes

# Human-readable description of what the standard errors ACTUALLY are, shared by
# `show` and the approximations list so the two cannot disagree.
_se_description(result::REMResult) =
    result.se_type === :sandwich  ? "event-clustered sandwich (Godambe)" :
    result.se_type === :bootstrap ? "repeated control sampling (bootstrap)" :
                                    "inverse Hessian (one control draw)"

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
    println(io, "Converged: $(result.converged)")
    println(io, "Std. errors: $(_se_description(result))")
    # Only when ties actually occurred: on tie-free data the policy did nothing
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type)")
    println(io)
    # Shared ecosystem coefficient table (Networks.jl), with significance codes
    # and the p-value display floor (an underflowed p prints as "<1e-16")
    print_coeftable(io, result.stat_names, result.coefficients, result.std_errors,
                    result.p_values; z_values=result.z_values)

    # Honest-uncertainty caveat (issue REM#2), and the prose twin of what
    # `approximations(result)` reports. It applies only when the risk set was
    # SAMPLED — with the full risk set there is no control-sampling variability
    # to ignore — and it says something different for each estimator, because
    # the three do genuinely different things:
    #   :hessian   — conditional on ONE control draw, and assumes the information
    #                equals the score variance
    #   :sandwich  — drops the second assumption, keeps the first
    #   :bootstrap — drops the first: the controls are redrawn
    _full_risk_set(result) && return
    println(io)
    if result.se_type === :bootstrap
        println(io, "Note: the standard errors come from refitting across independent redraws")
        println(io, "of the control set (within-draw + between-draw variance components), so")
        println(io, "they DO include the variance the risk-set sampling induces. The point")
        println(io, "estimates remain those of the original draw.")
    elseif result.se_type === :sandwich
        println(io, "Warning: the event-clustered sandwich standard errors are robust to")
        println(io, "misspecification of the within-stratum conditional model, but they are")
        println(io, "still computed on the ONE sampled control set that was drawn: they do")
        println(io, "not include the variance the risk-set sampling itself induces. Refit")
        println(io, "with `se=:bootstrap` to include it.")
    else
        println(io, "Warning: the inverse-Hessian standard errors are conditional on the ONE")
        println(io, "sampled control set that was drawn: they do not include the variance the")
        println(io, "risk-set sampling itself induces, so they are understated. Refit with")
        println(io, "`se=:sandwich` (robust to misspecification) or `se=:bootstrap` (repeated")
        println(io, "control sampling, which does include it).")
    end
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

`true` only when the risk set was NOT sampled **and** no tie correction was
needed. With the full risk set (`sampling_prob == 1`) on strictly ordered data,
the conditional-logit partial likelihood is the exact ordinal relational-event
likelihood. Two things make it an approximation instead:

- **case-control sampling** (the default, `n_controls = 100`); and
- **tied event times** — Breslow, Efron and arbitrary ordering are all
  approximations to a likelihood over an order the data does not determine (a
  fit on tie-free data has `tie_type == :none` and is unaffected).

It also reports `false` for a `REMResult` built without risk-set bookkeeping,
where the sampling is unknown — the conservative answer.
"""
is_exact(result::REMResult) =
    _full_risk_set(result) && result.tie_type === :none

"""
    se_method(result::REMResult) -> Symbol

What the reported standard errors ACTUALLY are:

- `:hessian` — the inverse negative Hessian of the conditional-logit partial
  likelihood, computed on the one sampled control set (the default)
- `:sandwich` — the event-clustered Godambe sandwich `H⁻¹ B H⁻¹`, with the meat
  `B` the outer product of the per-event score contributions
- `:bootstrap` — the empirical covariance of refits across independent redraws of
  the case-control risk set

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
                       "ordinal likelihood")
        end
        # Issue REM#2: what each standard-error estimator does and does not
        # account for. Only the repeated-control-sampling bootstrap sees the
        # variability that the risk-set sampling induces.
        if result.se_type === :bootstrap
            push!(out, "standard errors come from refitting across independent redraws of " *
                       "the control set (`se=:bootstrap`) and combining the within-draw and " *
                       "between-draw variance components by the law of total variance, so " *
                       "they DO include the variance the risk-set sampling induces; the " *
                       "point estimates are still those of the original control draw")
        elseif result.se_type === :sandwich
            push!(out, "the event-clustered sandwich standard errors (`se=:sandwich`) are " *
                       "robust to misspecification of the within-stratum conditional model, " *
                       "but they are computed on the ONE sampled control set that was drawn: " *
                       "they do not include the variance induced by the risk-set sampling " *
                       "itself (`se=:bootstrap` does)")
        else
            push!(out, "the inverse-Hessian standard errors are conditional on the ONE " *
                       "sampled control set that was drawn: they do not include the " *
                       "variance induced by the risk-set sampling itself, so they are " *
                       "understated (refit with `se=:bootstrap` to include it, or " *
                       "`se=:sandwich` for a misspecification-robust covariance)")
        end
    end
    return out
end

"""
    fit_rem(observations::DataFrame, stat_names::Vector{String}; kwargs...) -> REMResult

Fit a relational event model using stratified Cox regression.

# Arguments
- `observations::DataFrame`: Output from `generate_observations`
- `stat_names::Vector{String}`: Names of statistic columns to include in the model

# Keyword Arguments
- `maxiter::Int=100`: Maximum iterations for optimization
- `tol::Float64=1e-8`: Convergence tolerance
- `se::Symbol=:hessian`: How to compute the standard errors — `:hessian` or
  `:sandwich` (see the `fit_rem(::EventSequence, ...)` method for what each one
  accounts for). `:bootstrap` is **not** available on this method: it redraws the
  case-control risk set, and by the time the observations are a DataFrame the
  controls have already been drawn once and for all. Pass the `EventSequence`
  instead.

Tied event times are handled where the observations are *generated* (see `ties=`
in [`generate_observations`](@ref)): by the time they are a DataFrame the design
is fixed. What this method does is *report* the policy honestly — it reads the
`tie_weight` column (the Efron denominator weights) and the `"tie_method"`
DataFrame metadata that `generate_observations` attaches, so `tie_method(fit)`
cannot claim a correction the design does not carry. A hand-built DataFrame with
neither reports `:none`, and is fitted unweighted, exactly as before.

# Returns
- `REMResult`: Fitted model results
"""
function fit_rem(observations::DataFrame, stat_names::Vector{String};
                 maxiter::Int=100, tol::Float64=1e-8, se::Symbol=:hessian)
    se in (:hessian, :sandwich) || (se === :bootstrap ?
        throw(ArgumentError(
            "se=:bootstrap redraws the case-control risk set, which cannot be " *
            "done from an already-sampled observations DataFrame: the controls " *
            "are fixed in it. Call `fit_rem(seq, stats; se=:bootstrap)` on the " *
            "EventSequence instead.")) :
        throw(ArgumentError("se must be :hessian or :sandwich, got :$se")))
    # Validate input
    missing_cols = setdiff(stat_names, names(observations))
    isempty(missing_cols) ||
        throw(ArgumentError("Statistic columns not found in observations: $(join(missing_cols, ", "))"))
    for col in ("is_event", "stratum")
        col in names(observations) ||
            throw(ArgumentError("observations must have an `$col` column (from generate_observations)"))
    end

    # Extract data
    X = Matrix{Float64}(observations[!, stat_names])
    y = observations.is_event
    strata = observations.stratum

    # Each stratum must contain exactly one case and at least one control
    # (single pass over the rows)
    case_counts = Dict{eltype(strata), Int}()
    row_counts = Dict{eltype(strata), Int}()
    for (s, yi) in zip(strata, y)
        case_counts[s] = get(case_counts, s, 0) + (yi ? 1 : 0)
        row_counts[s] = get(row_counts, s, 0) + 1
    end
    for (s, n_cases) in case_counts
        n_cases == 1 ||
            throw(ArgumentError("Stratum $s has $n_cases cases; each stratum must have exactly one"))
        row_counts[s] >= 2 ||
            throw(ArgumentError("Stratum $s has no controls; each stratum must have at least one"))
    end

    n_obs = nrow(observations)
    n_params = length(stat_names)
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
        tie_type
    )
end

"""
    fit_rem(seq::EventSequence, stats; kwargs...) -> REMResult

Fit a relational event model directly from an event sequence.

# Arguments
- `seq::EventSequence`: The event sequence
- `stats`: Statistics to include in the model (a `StatisticSet` or a vector
  of statistics)

# Keyword Arguments
- `n_controls::Int=100`: Number of controls per case
- `decay::Float64=0.0`: Exponential decay rate
- `exclude_self_loops::Bool=true`: Exclude self-loops from risk set
- `at_risk=nothing`: The risk set (see `generate_observations`). One of `nothing`
  (the sequence's actor universe), an `ActorSet`/`Set{Int}`/`Vector{Int}` (static
  universe), a `RiskSet`, a vector of per-event risk sets, or a callback
  `(event_index, state) -> RiskSet`. `riskset` is accepted as an alias.
- `seed::Union{Int,Nothing}=nothing`: Random seed for the control draw
- `ties::Symbol=:error`: How to handle tied event times (below)
- `maxiter::Int=100`: Maximum iterations
- `tol::Float64=1e-8`: Convergence tolerance
- `se::Symbol=:hessian`: How to compute the standard errors (below)
- `n_boot::Int=100`: Number of control redraws (`se=:bootstrap` only)
- `rng::AbstractRNG=Random.default_rng()`: Source of the bootstrap randomness —
  a fixed `rng` reproduces the standard errors exactly

The actor universe determines the estimand: the likelihood is conditional on the
risk set, so nonparticipants that are eligible but never observed must be part of
it. Declare it with `EventSequence(events; actors=...)` or pass `at_risk`. When
neither is given, the universe falls back to the observed event endpoints (a
"participants only" risk set) and a warning is issued.

Every case is validated against its own risk set before fitting, and each stratum
must admit at least one control.

# Standard errors (issue REM#2)

The partial likelihood is evaluated on ONE draw of the controls, and it treats
the strata as independent. The three options relax different parts of that:

- `se=:hessian` (default) — the inverse negative Hessian. It assumes the
  observed information equals the score variance, and it is computed on the one
  control set that was drawn: it therefore **understates** the uncertainty,
  because the risk-set sampling is itself a source of variance and none of it
  appears here.
- `se=:sandwich` — the **event-clustered Godambe sandwich** `H⁻¹ B H⁻¹`, with
  the bread `H` the observed information and the meat `B = Σ_e u_e u_eᵀ` the
  outer product of the per-event score contributions (each event is one stratum,
  so the event is the clustering unit). This drops the information-equals-
  score-variance assumption — it is robust to misspecification of the
  within-stratum conditional model — but it is still computed on the one drawn
  control set.
- `se=:bootstrap` — **repeated control sampling**: redraw the case-control risk
  set `n_boot` times (independent seeds from `rng`), refit on each draw, and
  combine the two variance components by the law of total variance,

      Var(β̂) = E[Var(β̂ | controls)] + Var(E[β̂ | controls])
             ≈  W̄ (mean of the per-draw inverse-information covariances)
                + (1 + 1/n_boot) · (empirical covariance of the refits).

  The second term is exactly the risk-set sampling variability that the other two
  options condition away, so these standard errors necessarily **exceed** the
  `:hessian` ones and the gap is the missing component. (The empirical covariance
  of the refits *alone* would be a smaller number than `:hessian`, not a larger
  one — it is one component of the total, not the total, and reporting it as the
  standard error would understate the uncertainty even further.) A *parametric*
  bootstrap would be the wrong tool here altogether: the uncertainty being missed
  is in the sampling design, not in the model. The point estimates are unchanged
  — they remain those of the original (`seed`) control draw; only the covariance
  is replaced. Runs on the ONE shared `Networks.bootstrap_cov` loop.

The control inclusion probabilities that drive all of this are on the result:
see [`sampling_probs`](@ref) and [`risk_set_sizes`](@ref). With the full risk set
(`sampling_probs` all `1.0`) there is nothing to redraw and `:hessian` is
conditional on nothing.

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

# Returns
- `REMResult`: Fitted model results (`se_method(fit)` reports which of the three
  standard-error estimators was used, `tie_method(fit)` what happened to ties)
"""
function fit_rem(seq::EventSequence, stats::Vector{<:AbstractStatistic}; kwargs...)
    return fit_rem(seq, StatisticSet(stats); kwargs...)
end

function fit_rem(seq::EventSequence, stats::StatisticSet;
                 n_controls::Int=100, decay::Float64=0.0,
                 exclude_self_loops::Bool=true, seed::Union{Int,Nothing}=nothing,
                 at_risk=nothing, riskset=nothing, ties::Symbol=:error,
                 maxiter::Int=100, tol::Float64=1e-8,
                 se::Symbol=:hessian, n_boot::Int=100,
                 rng::AbstractRNG=Random.default_rng())
    se in (:hessian, :sandwich, :bootstrap) ||
        throw(ArgumentError("se must be :hessian, :sandwich or :bootstrap, got :$se"))
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

    sampler = CaseControlSampler(n_controls=n_controls, exclude_self_loops=exclude_self_loops, seed=seed)
    observations = generate_observations(seq, stats, sampler; decay=decay, at_risk=spec,
                                         ties=ties)

    # `:hessian` and `:sandwich` are both computed from THIS control draw, so the
    # DataFrame method handles them. `:bootstrap` needs the sequence itself,
    # because what it resamples is the control draw.
    se === :bootstrap ||
        return fit_rem(observations, stats.names; maxiter=maxiter, tol=tol, se=se)

    fit = fit_rem(observations, stats.names; maxiter=maxiter, tol=tol, se=:hessian)
    std_errors = _rem_control_bootstrap_se(seq, stats, fit.coefficients;
                                           n_boot=n_boot, n_controls=n_controls,
                                           decay=decay,
                                           exclude_self_loops=exclude_self_loops,
                                           spec=spec, ties=ties, maxiter=maxiter,
                                           tol=tol, rng=rng)

    # The point estimate is the original control draw's; only the covariance is
    # replaced (and with it the z- and p-values that are functions of it).
    z_values = fit.coefficients ./ std_errors
    p_values = 2 .* ccdf.(Normal(), abs.(z_values))

    return REMResult(fit.coefficients, std_errors, z_values, p_values,
                     fit.stat_names, fit.n_events, fit.n_observations,
                     fit.log_likelihood, fit.converged, fit.strata,
                     fit.risk_set_sizes, fit.sampling_probs, :bootstrap,
                     fit.tie_type)
end

"""
    _rem_control_bootstrap_se(seq, stats, θ̂; n_boot, ..., rng) -> Vector{Float64}

Repeated-control-sampling standard errors (issue REM#2), by the **law of total
variance**.

The inverse-Hessian standard errors are computed on ONE draw of the case-control
risk set and are therefore blind to the variance that *sampling* the risk set
induces. Redraw the controls `n_boot` times (independent seeds drawn from the
caller's `rng`, so a fixed `rng` reproduces the standard errors exactly), refit
the conditional logit on each draw, and combine the two variance components that
the redraws expose:

    Var(β̂) = E[Var(β̂ | controls)] + Var(E[β̂ | controls])
           ≈  W̄                    +  (1 + 1/B) · B_between

- **within**, `W̄` — the average of the per-draw inverse-information covariances.
  This is what a single `se=:hessian` fit reports (one draw of it).
- **between**, `B_between` — the empirical covariance of the refitted
  coefficients across the redraws. **This term is exactly the risk-set sampling
  variability that `se=:hessian` conditions away.** The `(1 + 1/B)` factor is the
  usual finite-`B` correction (Rubin's rule) for estimating it from `B` draws.

The between term ALONE would be the wrong answer, and a smaller one than the
Hessian: it is one component of the total, not the total. Reporting it as the
standard error would understate the uncertainty even more than `:hessian` does.
Because `W̄ ≈ H⁻¹`, the returned standard errors necessarily *exceed* the
single-draw Hessian ones, and the gap IS the missing component.

The resampling loop is the shared `Networks.bootstrap_cov` — the same one the
parametric bootstraps of ERGM/ERGMCount/ERGMRank/ERGMMulti run on. Only the
callbacks differ: what is resampled here is the SAMPLING DESIGN, not the model.
Each replicate carries its index so that its within-draw covariance can be
stashed alongside the coefficients the shared loop collects.
"""
function _rem_control_bootstrap_se(seq::EventSequence, stats::StatisticSet,
                                   θ̂::Vector{Float64}; n_boot::Int,
                                   n_controls::Int, decay::Float64,
                                   exclude_self_loops::Bool, spec,
                                   ties::Symbol, maxiter::Int, tol::Float64,
                                   rng::AbstractRNG)
    n_boot >= 2 ||
        throw(ArgumentError("n_boot must be at least 2 to form a covariance " *
                            "(got $n_boot)"))
    p = length(θ̂)

    # Within-draw covariances, one per replicate (written from the threaded
    # refits at distinct indices, which is why the replicates carry their index)
    within = Vector{Matrix{Float64}}(undef, n_boot)

    function simulate(rng, B)
        seeds = rand(rng, 1:typemax(Int), B)
        return [(b, generate_observations(
                        seq, stats,
                        CaseControlSampler(n_controls=n_controls,
                                           exclude_self_loops=exclude_self_loops,
                                           seed=seeds[b]);
                        decay=decay, at_risk=spec, ties=ties))
                for b in 1:B]
    end

    function refit(replicate)
        b, obs = replicate
        X = Matrix{Float64}(obs[!, stats.names])
        fit = _fit_stratified_clogit(X, obs.is_event, obs.stratum;
                                     maxiter=maxiter, tol=tol,
                                     tie_weights=obs.tie_weight)
        within[b] = fit.var_cov
        return fit.coefficients
    end

    boot = bootstrap_cov(refit, simulate, θ̂; n_boot=n_boot, rng=rng)

    W̄ = sum(within) ./ n_boot                       # E[Var(β̂ | controls)]
    between = boot.vcov                              # Var(E[β̂ | controls])
    V = W̄ .+ (1 + 1 / n_boot) .* between             # law of total variance

    return sqrt.(max.(diag(V), 0.0))
end

# Internal: Fit stratified conditional logistic regression via Newton-Raphson.
# `se` selects the covariance estimator applied at the final β: `:hessian` (the
# inverse observed information) or `:sandwich` (the event-clustered Godambe
# sandwich, see `_clogit_sandwich_se`). The POINT ESTIMATE does not depend on it.
#
# `tie_weights` are the per-row DENOMINATOR weights (the `tie_weight` column):
# all 1 unless the Efron tie correction put fractional weights on the cases tied
# at one timestamp. With all-ones weights this is bit-for-bit the likelihood it
# always was.
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

    # Initialize coefficients
    beta = zeros(p)

    # Get unique strata
    unique_strata = unique(strata)
    strata_indices = Dict(s => findall(==(s), strata) for s in unique_strata)

    # Preallocated per-stratum work buffers shared across all derivative
    # evaluations (sized for the largest stratum)
    work = _clogit_workspace(strata_indices, p)

    converged = false
    ll, grad, hess = _compute_clogit_derivatives(X, y, strata_indices, beta, work, tw)

    for iter in 1:maxiter
        # Convergence: small likelihood change AND small gradient
        # (log-likelihood change alone can stall away from the optimum)
        # Newton-Raphson update with step-halving if the likelihood drops
        delta = try
            -hess \ grad
        catch e
            @warn "Hessian inversion failed at iteration $iter: $e"
            break
        end

        step = 1.0
        ll_new = -Inf
        local grad_new, hess_new
        for _ in 1:10
            ll_new, grad_new, hess_new =
                _compute_clogit_derivatives(X, y, strata_indices, beta .+ step .* delta,
                                            work, tw)
            ll_new >= ll && break
            step /= 2
        end

        beta .+= step .* delta
        ll_change = abs(ll_new - ll)
        ll, grad, hess = ll_new, grad_new, hess_new

        if ll_change < tol && norm(grad) < sqrt(tol)
            converged = true
            break
        end
    end

    # Standard errors at the final β (derivatives above are always recomputed
    # post-update, so hess is never stale)
    std_errors = if se === :sandwich
        _clogit_sandwich_se(X, y, strata_indices, beta, hess, tw)
    else
        try
            var_cov = -inv(hess)
            sqrt.(diag(var_cov))
        catch
            fill(NaN, p)
        end
    end

    z_values = beta ./ std_errors
    # ccdf keeps precision in the tail: 2*(1 - cdf(...)) underflows to
    # exactly 0 for |z| ≳ 8
    p_values = 2 .* ccdf.(Normal(), abs.(z_values))

    # The inverse observed information, always: the repeated-control-sampling
    # standard errors need it per draw as the WITHIN-draw variance component
    # (see `_rem_control_bootstrap_se`), whatever `se` the caller asked for.
    var_cov = try
        Matrix{Float64}(-inv(hess))
    catch
        fill(NaN, p, p)
    end

    return (
        coefficients = beta,
        std_errors = std_errors,
        z_values = z_values,
        p_values = p_values,
        log_likelihood = ll,
        converged = converged,
        var_cov = var_cov
    )
end

"""
    _clogit_sandwich_se(X, y, strata_indices, beta, hess) -> Vector{Float64}

Event-clustered (Godambe) sandwich standard errors for the stratified
conditional logit: `V = H⁻¹ B H⁻¹`, with

- the **bread** `H = −hess`, the observed information at β̂, and
- the **meat** `B = Σ_e u_e u_eᵀ`, the outer product of the per-event score
  contributions `u_e = x_case(e) − E_e[X]`.

Each event is exactly one stratum (one case against its sampled controls), so the
event IS the clustering unit and no `cluster` argument is needed.

The default inverse-Hessian standard errors are `H⁻¹`, which is `V` only under
the information equality `B = H` — i.e. only if the within-stratum conditional
model is correctly specified. The sandwich drops that assumption; it is what
`coxph(..., robust = TRUE)` reports in R. It does **not**, however, account for
the variance induced by *sampling* the risk set: it is still computed on the one
control set that was drawn (that is what `fit_rem(...; se=:bootstrap)` is for).

At the optimum `Σ_e u_e = 0` (the gradient vanishes), so no centring is applied.
"""
function _clogit_sandwich_se(X::Matrix{Float64}, y::Vector{Bool},
                             strata_indices::Dict{Int, Vector{Int}},
                             beta::Vector{Float64}, hess::Matrix{Float64},
                             tw::Vector{Float64}=ones(size(X, 1)))
    p = size(X, 2)
    meat = zeros(p, p)
    u = Vector{Float64}(undef, p)
    xexp = Vector{Float64}(undef, p)

    for (_, indices) in strata_indices
        n_s = length(indices)

        case_idx = 0
        @inbounds for (a, idx) in enumerate(indices)
            if y[idx]
                case_idx = a
                break
            end
        end
        case_idx == 0 && continue

        # Softmax probabilities within the stratum (same log-sum-exp, and the
        # same denominator weights, as the likelihood — so the score cannot
        # drift from the model it scores)
        eta = [dot(view(X, indices[a], :), beta) for a in 1:n_s]
        eta_max = maximum(eta)
        probs = [tw[indices[a]] * exp(eta[a] - eta_max) for a in 1:n_s]
        probs ./= sum(probs)

        # E[X] within the stratum, then the score u_e = x_case − E[X]
        fill!(xexp, 0.0)
        @inbounds for a in 1:n_s, k in 1:p
            xexp[k] += probs[a] * X[indices[a], k]
        end
        @inbounds for k in 1:p
            u[k] = X[indices[case_idx], k] - xexp[k]
        end

        BLAS.ger!(1.0, u, u, meat)
    end

    return try
        bread = inv(-hess)                 # H⁻¹, the observed-information inverse
        V = bread * meat * bread           # Godambe sandwich
        sqrt.(max.(diag(V), 0.0))
    catch
        fill(NaN, p)
    end
end

# Internal: preallocated per-stratum buffers for _compute_clogit_derivatives,
# sized for the largest stratum
function _clogit_workspace(strata_indices::Dict{Int, Vector{Int}}, p::Int)
    max_ns = isempty(strata_indices) ? 0 :
             maximum(length(ix) for ix in values(strata_indices))
    return (
        Xs = Matrix{Float64}(undef, max_ns, p),      # stratum design matrix
        Xw = Matrix{Float64}(undef, max_ns, p),      # sqrt(prob)-weighted rows
        eta = Vector{Float64}(undef, max_ns),        # linear predictor
        probs = Vector{Float64}(undef, max_ns),      # softmax probabilities
        xexp = Vector{Float64}(undef, p),            # E[X] within stratum
    )
end

# Internal: Compute derivatives for conditional logistic regression.
# Accumulates the Hessian in place via BLAS (gemm on sqrt-weighted rows for
# -E[XX'], ger! for the +E[X]E[X]' rank-1 update) instead of allocating a
# p×p outer product per observation.
#
# `tw` holds the per-row DENOMINATOR weights (`tie_weight`): the stratum's
# normalizing sum is Σ_a tw_a·exp(η_a), while the numerator stays exp(η_case).
# They are 1 everywhere except under the Efron tie correction, so the weighted
# likelihood REDUCES to the unweighted one on tie-free data — which is why a tie
# correction is a no-op there.
function _compute_clogit_derivatives(X::Matrix{Float64}, y::Vector{Bool},
                                     strata_indices::Dict{Int, Vector{Int}}, beta::Vector{Float64},
                                     work=_clogit_workspace(strata_indices, size(X, 2)),
                                     tw::Vector{Float64}=ones(size(X, 1)))
    n, p = size(X)

    ll = 0.0
    grad = zeros(p)
    hess = zeros(p, p)
    xexp = work.xexp

    for (stratum, indices) in strata_indices
        n_s = length(indices)

        # Find the case (event that occurred)
        case_idx = 0
        @inbounds for (a, idx) in enumerate(indices)
            if y[idx]
                case_idx = a
                break
            end
        end
        case_idx == 0 && continue

        # Copy the stratum rows into the contiguous work buffer
        X_s = view(work.Xs, 1:n_s, :)
        @inbounds for k in 1:p, a in 1:n_s
            X_s[a, k] = X[indices[a], k]
        end

        # Linear predictor
        eta = view(work.eta, 1:n_s)
        mul!(eta, X_s, beta)

        # Numerical stability: subtract max
        eta_max = maximum(eta)
        probs = view(work.probs, 1:n_s)
        sum_exp_eta = 0.0
        @inbounds for a in 1:n_s
            probs[a] = tw[indices[a]] * exp(eta[a] - eta_max)
            sum_exp_eta += probs[a]
        end

        # Log-likelihood contribution
        ll += eta[case_idx] - eta_max - log(sum_exp_eta)

        # Probabilities
        probs ./= sum_exp_eta

        # Gradient contribution: X_case - E[X]
        mul!(xexp, transpose(X_s), probs)
        @inbounds for k in 1:p
            grad[k] += X_s[case_idx, k] - xexp[k]
        end

        # Hessian contribution: -Var[X] = -(E[XX'] - E[X]E[X]')
        Xw = view(work.Xw, 1:n_s, :)
        @inbounds for a in 1:n_s
            probs[a] = sqrt(probs[a])
        end
        @inbounds for k in 1:p, a in 1:n_s
            Xw[a, k] = probs[a] * X_s[a, k]
        end
        mul!(hess, transpose(Xw), Xw, -1.0, 1.0)
        BLAS.ger!(1.0, xexp, xexp, hess)
    end

    return ll, grad, hess
end


"""
    coef(result::REMResult) -> Vector{Float64}

Extract coefficients from a fitted model.
"""
coef(result::REMResult) = result.coefficients

"""
    stderror(result::REMResult) -> Vector{Float64}

Extract standard errors from a fitted model.
"""
stderror(result::REMResult) = result.std_errors

"""
    coeftable(result::REMResult) -> DataFrame

Return coefficients as a DataFrame.
"""
function coeftable(result::REMResult)
    return DataFrame(
        statistic = result.stat_names,
        coefficient = result.coefficients,
        std_error = result.std_errors,
        z_value = result.z_values,
        p_value = result.p_values
    )
end
