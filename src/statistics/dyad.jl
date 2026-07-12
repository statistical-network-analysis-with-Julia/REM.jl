"""
Dyad-level statistics for REM.

These statistics capture effects based on the history of events between the
focal sender-receiver pair.
"""

"""
    Repetition <: DyadStatistic

Measures the tendency for repeated events from sender to receiver.
Returns the (weighted) count of past events from sender to receiver.

# Fields
- `directed::Bool`: If true, count only events from s→r. If false, count events in both directions.
- `stat_name::String`: Name for this statistic.
"""
struct Repetition <: DyadStatistic
    directed::Bool
    stat_name::String

    Repetition(; directed::Bool=true, name::String="") = new(directed,
        isempty(name) ? (directed ? "repetition" : "undirected_repetition") : name)
end

function compute(stat::Repetition, state::EventNetworkState, sender::Int, receiver::Int)
    if stat.directed
        return get_dyad_count(state, sender, receiver)
    else
        return get_undirected_count(state, sender, receiver)
    end
end

name(stat::Repetition) = stat.stat_name

"""
    Reciprocity <: DyadStatistic

Measures the tendency for reciprocal events.
Returns the (weighted) count of past events from receiver to sender.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct Reciprocity <: DyadStatistic
    stat_name::String

    Reciprocity(; name::String="reciprocity") = new(name)
end

function compute(stat::Reciprocity, state::EventNetworkState, sender::Int, receiver::Int)
    return get_dyad_count(state, receiver, sender)
end

name(stat::Reciprocity) = stat.stat_name

"""
    InertiaStatistic <: DyadStatistic

Measures inertia - the tendency for events to persist in a direction.
Combines repetition and reciprocity effects.

Returns: repetition_weight * repetition + reciprocity_weight * reciprocity

# Fields
- `repetition_weight::Float64`: Weight for repetition component.
- `reciprocity_weight::Float64`: Weight for reciprocity component.
- `stat_name::String`: Name for this statistic.
"""
struct InertiaStatistic <: DyadStatistic
    repetition_weight::Float64
    reciprocity_weight::Float64
    stat_name::String

    function InertiaStatistic(; repetition_weight::Float64=1.0, reciprocity_weight::Float64=1.0,
                               name::String="inertia")
        new(repetition_weight, reciprocity_weight, name)
    end
end

function compute(stat::InertiaStatistic, state::EventNetworkState, sender::Int, receiver::Int)
    rep = get_dyad_count(state, sender, receiver)
    recip = get_dyad_count(state, receiver, sender)
    return stat.repetition_weight * rep + stat.reciprocity_weight * recip
end

name(stat::InertiaStatistic) = stat.stat_name

"""
    RecencyStatistic <: DyadStatistic

Measures recency - how recently the last event occurred on this dyad.
Returns the inverse of elapsed time since last event (or 0 if no prior events).

# Fields
- `directed::Bool`: If true, only consider events from s→r.
- `transform::Symbol`: Transform to apply (:inverse, :log, :exp_decay).
- `decay::Float64`: Decay parameter for :exp_decay transform.
- `stat_name::String`: Name for this statistic.
"""
struct RecencyStatistic <: DyadStatistic
    directed::Bool
    transform::Symbol
    decay::Float64
    stat_name::String

    function RecencyStatistic(; directed::Bool=true, transform::Symbol=:inverse,
                               decay::Float64=1.0, name::String="")
        stat_name = isempty(name) ? "recency_$(transform)" : name
        new(directed, transform, decay, stat_name)
    end
end

function compute(stat::RecencyStatistic, state::EventNetworkState{T}, sender::Int, receiver::Int) where T
    dyad = stat.directed ? (sender, receiver) : minmax(sender, receiver)

    if !haskey(state.last_event_time, dyad)
        # Also check reverse direction for undirected
        if !stat.directed
            rev_dyad = minmax(receiver, sender)
            if !haskey(state.last_event_time, rev_dyad)
                return 0.0
            end
            last_time = state.last_event_time[rev_dyad]
        else
            return 0.0
        end
    else
        last_time = state.last_event_time[dyad]
    end

    elapsed = _elapsed_seconds(state.current_time - last_time)
    if elapsed <= 0
        return 0.0
    end

    if stat.transform == :inverse
        return 1.0 / elapsed
    elseif stat.transform == :log
        return 1.0 / log1p(elapsed)
    elseif stat.transform == :exp_decay
        return exp(-stat.decay * elapsed)
    else
        error("Unknown transform: $(stat.transform)")
    end
end

name(stat::RecencyStatistic) = stat.stat_name

"""
    DyadCovariate <: DyadStatistic

A statistic based on a pre-specified dyad-level covariate matrix.

# Fields
- `values::Dict{Tuple{Int,Int}, Float64}`: Mapping from dyad to covariate value.
- `default::Float64`: Default value for dyads not in the dict.
- `stat_name::String`: Name for this statistic.
"""
struct DyadCovariate <: DyadStatistic
    values::Dict{Tuple{Int,Int}, Float64}
    default::Float64
    stat_name::String

    function DyadCovariate(values::Dict{Tuple{Int,Int}, Float64};
                           default::Float64=0.0, name::String="dyad_covariate")
        new(values, default, name)
    end
end

function compute(stat::DyadCovariate, state::EventNetworkState, sender::Int, receiver::Int)
    return get(stat.values, (sender, receiver), stat.default)
end

name(stat::DyadCovariate) = stat.stat_name
