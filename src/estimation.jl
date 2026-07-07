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
    println(io, "Coefficients:")
    println(io, "-"^60)

    # Header
    @printf(io, "%-20s %10s %10s %10s %10s\n", "Statistic", "Coef", "Std.Err", "z", "P>|z|")
    println(io, "-"^60)

    for i in 1:length(result.coefficients)
        sig = result.p_values[i] < 0.001 ? "***" :
              result.p_values[i] < 0.01 ? "**" :
              result.p_values[i] < 0.05 ? "*" :
              result.p_values[i] < 0.1 ? "." : ""
        @printf(io, "%-20s %10.4f %10.4f %10.4f %10.4f %s\n",
                result.stat_names[i][1:min(20, length(result.stat_names[i]))],
                result.coefficients[i], result.std_errors[i],
                result.z_values[i], result.p_values[i], sig)
    end
    println(io, "-"^60)
    println(io, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
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

    # Each stratum must contain exactly one case
    for s in unique(strata)
        n_cases = sum(y[strata .== s])
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
    fit_rem(seq::EventSequence, stats::Vector{<:AbstractStatistic}; kwargs...) -> REMResult

Fit a relational event model directly from an event sequence.

# Arguments
- `seq::EventSequence`: The event sequence
- `stats::Vector{<:AbstractStatistic}`: Statistics to include in the model

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
function fit_rem(seq::EventSequence, stats::Vector{<:AbstractStatistic};
                 n_controls::Int=100, decay::Float64=0.0,
                 exclude_self_loops::Bool=true, seed::Union{Int,Nothing}=nothing,
                 maxiter::Int=100, tol::Float64=1e-8)
    sampler = CaseControlSampler(n_controls=n_controls, exclude_self_loops=exclude_self_loops, seed=seed)
    observations = generate_observations(seq, stats, sampler; decay=decay)
    stat_names = [name(s) for s in stats]
    return fit_rem(observations, stat_names; maxiter=maxiter, tol=tol)
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

    converged = false
    ll, grad, hess = _compute_clogit_derivatives(X, y, strata_indices, beta)

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
                _compute_clogit_derivatives(X, y, strata_indices, beta .+ step .* delta)
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
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(z_values)))

    return (
        coefficients = beta,
        std_errors = std_errors,
        z_values = z_values,
        p_values = p_values,
        log_likelihood = ll,
        converged = converged
    )
end

# Internal: Compute derivatives for conditional logistic regression
function _compute_clogit_derivatives(X::Matrix{Float64}, y::Vector{Bool},
                                     strata_indices::Dict{Int, Vector{Int}}, beta::Vector{Float64})
    n, p = size(X)

    ll = 0.0
    grad = zeros(p)
    hess = zeros(p, p)

    for (stratum, indices) in strata_indices
        # Get data for this stratum
        X_s = X[indices, :]
        y_s = y[indices]
        n_s = length(indices)

        # Linear predictor
        eta = X_s * beta

        # Numerical stability: subtract max
        eta_max = maximum(eta)
        exp_eta = exp.(eta .- eta_max)
        sum_exp_eta = sum(exp_eta)

        # Probabilities
        probs = exp_eta ./ sum_exp_eta

        # Find the case (event that occurred)
        case_idx = findfirst(y_s)
        if isnothing(case_idx)
            continue
        end

        # Log-likelihood contribution
        ll += eta[case_idx] - eta_max - log(sum_exp_eta)

        # Gradient contribution: X_case - E[X]
        x_case = X_s[case_idx, :]
        x_expected = X_s' * probs
        grad .+= x_case .- x_expected

        # Hessian contribution: -Var[X] = -(E[XX'] - E[X]E[X]')
        for i in 1:n_s
            hess .-= probs[i] .* (X_s[i, :] * X_s[i, :]')
        end
        hess .+= x_expected * x_expected'
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
