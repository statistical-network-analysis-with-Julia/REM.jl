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
end

function Base.show(io::IO, result::REMResult)
    println(io, "Relational Event Model Results")
    println(io, "==============================")
    println(io, "Events: $(result.n_events), Observations: $(result.n_observations)")
    println(io, "Log-likelihood: $(round(result.log_likelihood, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    # Shared ecosystem coefficient table (Network.jl), with significance codes
    # and the p-value display floor (an underflowed p prints as "<1e-16")
    print_coeftable(io, result.stat_names, result.coefficients, result.std_errors,
                    result.p_values; z_values=result.z_values)
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

# Returns
- `REMResult`: Fitted model results
"""
function fit_rem(observations::DataFrame, stat_names::Vector{String};
                 maxiter::Int=100, tol::Float64=1e-8)
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

    # Each stratum must contain exactly one case (single pass over the rows)
    case_counts = Dict{eltype(strata), Int}()
    for (s, yi) in zip(strata, y)
        case_counts[s] = get(case_counts, s, 0) + (yi ? 1 : 0)
    end
    for (s, n_cases) in case_counts
        n_cases == 1 ||
            throw(ArgumentError("Stratum $s has $n_cases cases; each stratum must have exactly one"))
    end

    n_obs = nrow(observations)
    n_params = length(stat_names)
    n_events = sum(y)

    # Fit stratified conditional logistic regression (equivalent to Cox model for case-control)
    result = _fit_stratified_clogit(X, y, strata; maxiter=maxiter, tol=tol)

    return REMResult(
        result.coefficients,
        result.std_errors,
        result.z_values,
        result.p_values,
        stat_names,
        n_events,
        n_obs,
        result.log_likelihood,
        result.converged
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
- `seed::Union{Int,Nothing}=nothing`: Random seed
- `maxiter::Int=100`: Maximum iterations
- `tol::Float64=1e-8`: Convergence tolerance

# Returns
- `REMResult`: Fitted model results
"""
function fit_rem(seq::EventSequence, stats::Vector{<:AbstractStatistic}; kwargs...)
    return fit_rem(seq, StatisticSet(stats); kwargs...)
end

function fit_rem(seq::EventSequence, stats::StatisticSet;
                 n_controls::Int=100, decay::Float64=0.0,
                 exclude_self_loops::Bool=true, seed::Union{Int,Nothing}=nothing,
                 maxiter::Int=100, tol::Float64=1e-8)
    sampler = CaseControlSampler(n_controls=n_controls, exclude_self_loops=exclude_self_loops, seed=seed)
    observations = generate_observations(seq, stats, sampler; decay=decay)
    return fit_rem(observations, stats.names; maxiter=maxiter, tol=tol)
end

# Internal: Fit stratified conditional logistic regression via Newton-Raphson
function _fit_stratified_clogit(X::Matrix{Float64}, y::Vector{Bool}, strata::Vector{Int};
                                 maxiter::Int=100, tol::Float64=1e-8)
    n, p = size(X)

    # Initialize coefficients
    beta = zeros(p)

    # Get unique strata
    unique_strata = unique(strata)
    strata_indices = Dict(s => findall(==(s), strata) for s in unique_strata)

    # Preallocated per-stratum work buffers shared across all derivative
    # evaluations (sized for the largest stratum)
    work = _clogit_workspace(strata_indices, p)

    converged = false
    ll, grad, hess = _compute_clogit_derivatives(X, y, strata_indices, beta, work)

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
                _compute_clogit_derivatives(X, y, strata_indices, beta .+ step .* delta, work)
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

    # Standard errors from the Hessian at the final β (derivatives above are
    # always recomputed post-update, so hess is never stale)
    std_errors = try
        var_cov = -inv(hess)
        sqrt.(diag(var_cov))
    catch
        fill(NaN, p)
    end

    z_values = beta ./ std_errors
    # ccdf keeps precision in the tail: 2*(1 - cdf(...)) underflows to
    # exactly 0 for |z| ≳ 8
    p_values = 2 .* ccdf.(Normal(), abs.(z_values))

    return (
        coefficients = beta,
        std_errors = std_errors,
        z_values = z_values,
        p_values = p_values,
        log_likelihood = ll,
        converged = converged
    )
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
function _compute_clogit_derivatives(X::Matrix{Float64}, y::Vector{Bool},
                                     strata_indices::Dict{Int, Vector{Int}}, beta::Vector{Float64},
                                     work=_clogit_workspace(strata_indices, size(X, 2)))
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
            probs[a] = exp(eta[a] - eta_max)
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
