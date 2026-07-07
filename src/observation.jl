"""
Observation generation for REM estimation.

Implements case-control sampling and observation generation for fitting
relational event models using survival analysis methods.
"""

"""
    Observation

A single observation for model estimation, consisting of:
- Statistics computed for a potential event (sender→receiver)
- Whether the event actually occurred (case) or not (control)

# Fields
- `event_index::Int`: Index of the focal event in the sequence
- `sender::Int`: Sender ID
- `receiver::Int`: Receiver ID
- `statistics::Vector{Float64}`: Computed statistic values
- `is_event::Bool`: True if this dyad actually had an event (case)
- `stratum::Int`: Stratum ID for stratified analysis (events in same stratum share risk set)
"""
struct Observation
    event_index::Int
    sender::Int
    receiver::Int
    statistics::Vector{Float64}
    is_event::Bool
    stratum::Int
end

"""
    CaseControlSampler

Generates observations using case-control sampling.
For each observed event (case), samples a specified number of non-events (controls)
from the risk set.

# Fields
- `n_controls::Int`: Number of control samples per case
- `exclude_self_loops::Bool`: Whether to exclude self-loops from sampling
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
"""
struct CaseControlSampler
    n_controls::Int
    exclude_self_loops::Bool
    seed::Union{Int, Nothing}

    function CaseControlSampler(; n_controls::Int=100, exclude_self_loops::Bool=true,
                                 seed::Union{Int, Nothing}=nothing)
        n_controls > 0 || throw(ArgumentError("n_controls must be positive"))
        new(n_controls, exclude_self_loops, seed)
    end
end

"""
    generate_observations(seq::EventSequence, stats::Vector{<:AbstractStatistic},
                          sampler::CaseControlSampler; kwargs...) -> DataFrame

Generate observations for model estimation using case-control sampling.

# Arguments
- `seq::EventSequence`: The event sequence to analyze
- `stats::Vector{<:AbstractStatistic}`: Statistics to compute
- `sampler::CaseControlSampler`: Sampling configuration

# Keyword Arguments
- `start_index::Int=1`: Index of first event to include
- `end_index::Int=length(seq)`: Index of last event to include
- `decay::Float64=0.0`: Exponential decay rate for network state
- `at_risk::Union{Nothing, Set{Int}}=nothing`: Set of actors "at risk" (if nothing, all actors)

# Returns
- `DataFrame`: Observations with columns for each statistic, plus is_event and stratum
"""
function generate_observations(seq::EventSequence{T}, stats::Vector{<:AbstractStatistic},
                               sampler::CaseControlSampler;
                               start_index::Int=1, end_index::Int=length(seq),
                               decay::Float64=0.0,
                               at_risk::Union{Nothing, Set{Int}}=nothing) where T
    # Local RNG: reproducible without mutating the global RNG
    rng = isnothing(sampler.seed) ? Random.default_rng() : Random.Xoshiro(sampler.seed)

    # The ordinal likelihood assumes a strict event order; tied timestamps
    # are processed in (arbitrary) sequence order without a tie correction
    if end_index > start_index &&
       !allunique(seq[i].time for i in start_index:end_index)
        @warn "Event sequence contains tied timestamps; ties are ordered " *
              "arbitrarily and the ordinal likelihood applies no tie correction" maxlog = 1
    end

    # Initialize network state
    state = NetworkState(seq; decay=decay)

    # Process events before start_index to build initial state
    for i in 1:(start_index - 1)
        update!(state, seq[i])
    end

    # Determine actors in risk set
    actors = isnothing(at_risk) ? sort!(collect(seq.actors)) : sort!(collect(at_risk))
    n_actors = length(actors)

    # Number of distinct dyads available as controls (case dyad excluded)
    max_controls = n_actors * (n_actors - (sampler.exclude_self_loops ? 1 : 0)) - 1
    n_wanted = min(sampler.n_controls, max_controls)
    if n_wanted < sampler.n_controls
        @warn "Requested $(sampler.n_controls) controls but only $max_controls " *
              "distinct dyads are available; using the full risk set instead" maxlog = 1
    end

    # Pre-allocate observation storage
    observations = Observation[]
    stat_names = [name(s) for s in stats]

    # Process each event
    for event_idx in start_index:end_index
        event = seq[event_idx]

        # Update state time without adding the event yet
        if decay > 0 && event.time > state.current_time
            apply_decay!(state, event.time)
        end
        state.current_time = event.time

        # Compute statistics for the actual event (case)
        case_stats = compute_all(stats, state, event.sender, event.receiver)
        push!(observations, Observation(event_idx, event.sender, event.receiver,
                                        case_stats, true, event_idx))

        is_case = (s, r) -> s == event.sender && r == event.receiver

        # Controls are drawn WITHOUT replacement: a dyad drawn k times would
        # contribute k·exp(η) to the stratum denominator and bias estimates.
        if 2 * n_wanted >= max_controls
            # Dense request: enumerate all distinct control dyads, then take
            # a random subset (or all of them)
            all_dyads = Tuple{Int,Int}[]
            for s in actors, r in actors
                (sampler.exclude_self_loops && s == r) && continue
                is_case(s, r) && continue
                push!(all_dyads, (s, r))
            end
            chosen = n_wanted >= length(all_dyads) ? all_dyads :
                     all_dyads[randperm(rng, length(all_dyads))[1:n_wanted]]
            for (s, r) in chosen
                control_stats = compute_all(stats, state, s, r)
                push!(observations,
                      Observation(event_idx, s, r, control_stats, false, event_idx))
            end
        else
            # Sparse request: rejection-sample distinct dyads
            sampled = Set{Tuple{Int,Int}}()
            while length(sampled) < n_wanted
                s = actors[rand(rng, 1:n_actors)]
                r = actors[rand(rng, 1:n_actors)]

                (sampler.exclude_self_loops && s == r) && continue
                is_case(s, r) && continue
                (s, r) in sampled && continue

                push!(sampled, (s, r))
                control_stats = compute_all(stats, state, s, r)
                push!(observations,
                      Observation(event_idx, s, r, control_stats, false, event_idx))
            end
        end

        # Update state with the actual event
        update!(state, event)
    end

    # Convert to DataFrame
    return observations_to_dataframe(observations, stat_names)
end

"""
    observations_to_dataframe(observations::Vector{Observation}, stat_names::Vector{String}) -> DataFrame

Convert observations to a DataFrame.
"""
function observations_to_dataframe(observations::Vector{Observation}, stat_names::Vector{String})
    n_obs = length(observations)
    n_stats = length(stat_names)

    # Create columns
    data = Dict{String, Vector}()
    data["event_index"] = [o.event_index for o in observations]
    data["sender"] = [o.sender for o in observations]
    data["receiver"] = [o.receiver for o in observations]
    data["is_event"] = [o.is_event for o in observations]
    data["stratum"] = [o.stratum for o in observations]

    # Add statistic columns
    for (i, sname) in enumerate(stat_names)
        data[sname] = [o.statistics[i] for o in observations]
    end

    return DataFrame(data)
end

"""
    compute_statistics(seq::EventSequence, stats::Vector{<:AbstractStatistic};
                       decay::Float64=0.0) -> DataFrame

Compute statistics for all events in a sequence (without sampling controls).

# Returns
- `DataFrame`: One row per event with computed statistics
"""
function compute_statistics(seq::EventSequence{T}, stats::Vector{<:AbstractStatistic};
                            decay::Float64=0.0) where T
    state = NetworkState(seq; decay=decay)
    stat_names = [name(s) for s in stats]

    results = Vector{Vector{Float64}}()
    senders = Int[]
    receivers = Int[]
    times = T[]

    for (i, event) in enumerate(seq)
        # Update time and apply decay
        if decay > 0 && event.time > state.current_time
            apply_decay!(state, event.time)
        end
        state.current_time = event.time

        # Compute statistics
        stat_values = compute_all(stats, state, event.sender, event.receiver)
        push!(results, stat_values)
        push!(senders, event.sender)
        push!(receivers, event.receiver)
        push!(times, event.time)

        # Update state
        update!(state, event)
    end

    # Build DataFrame
    df = DataFrame(
        sender = senders,
        receiver = receivers,
        time = times
    )

    for (i, sname) in enumerate(stat_names)
        df[!, sname] = [r[i] for r in results]
    end

    return df
end
