"""
Network state tracking for computing statistics efficiently.
"""

# Convert a time difference into seconds, handling both numeric types and Dates periods.
function _elapsed_seconds(diff)
    if diff isa Real
        return Float64(diff)
    elseif diff isa Dates.Period || diff isa Dates.CompoundPeriod
        # Normalize date/time differences to seconds to keep decay units consistent.
        return Dates.value(Dates.Millisecond(diff)) / 1000
    else
        throw(ArgumentError("Unsupported time difference type $(typeof(diff))"))
    end
end

"""
    NetworkState

Tracks the cumulative state of the network up to a given point in time.
Used for efficient computation of statistics.

# Fields
- `n_actors::Int`: Number of actors
- `dyad_counts::Dict{Tuple{Int,Int}, Float64}`: Weighted count of events for each directed dyad
- `undirected_counts::Dict{Tuple{Int,Int}, Float64}`: Weighted count for undirected dyads (min,max sorted)
- `out_degree::Dict{Int, Float64}`: Weighted out-degree for each actor
- `in_degree::Dict{Int, Float64}`: Weighted in-degree for each actor
- `last_event_time::Dict{Tuple{Int,Int}, Any}`: Time of last event for each dyad
- `decay::Float64`: Exponential decay rate (0 = no decay)
- `current_time::Any`: Current time in the event sequence
"""
mutable struct NetworkState{T}
    n_actors::Int
    actors::Set{Int}
    dyad_counts::Dict{Tuple{Int,Int}, Float64}
    undirected_counts::Dict{Tuple{Int,Int}, Float64}
    out_degree::Dict{Int, Float64}
    in_degree::Dict{Int, Float64}
    last_event_time::Dict{Tuple{Int,Int}, T}
    decay::Float64
    current_time::T
    event_history::Vector{Tuple{Int,Int,T,Float64}}  # (sender, receiver, time, weight)
    # Incremental adjacency ("ever had an event"), so neighbor queries are
    # O(degree) instead of O(|event_history|). Note membership never
    # expires under decay: counts decay continuously, but structure counts
    # any past event (documented eventnet behavior).
    out_neighbors::Dict{Int, Set{Int}}
    in_neighbors::Dict{Int, Set{Int}}

    function NetworkState{T}(; n_actors::Int=0, decay::Float64=0.0) where T
        new{T}(
            n_actors,
            Set{Int}(),
            Dict{Tuple{Int,Int}, Float64}(),
            Dict{Tuple{Int,Int}, Float64}(),
            Dict{Int, Float64}(),
            Dict{Int, Float64}(),
            Dict{Tuple{Int,Int}, T}(),
            decay,
            zero(T),
            Tuple{Int,Int,T,Float64}[],
            Dict{Int, Set{Int}}(),
            Dict{Int, Set{Int}}()
        )
    end
end

"""
    NetworkState(seq::EventSequence; decay=0.0)

Create a NetworkState from an EventSequence without processing any events.
"""
function NetworkState(seq::EventSequence{T}; decay::Float64=0.0) where T
    state = NetworkState{T}(n_actors=seq.n_actors, decay=decay)
    state.actors = copy(seq.actors)
    return state
end

"""
    reset!(state::NetworkState)

Reset the network state to empty.
"""
function reset!(state::NetworkState{T}) where T
    empty!(state.dyad_counts)
    empty!(state.undirected_counts)
    empty!(state.out_degree)
    empty!(state.in_degree)
    empty!(state.last_event_time)
    empty!(state.event_history)
    empty!(state.out_neighbors)
    empty!(state.in_neighbors)
    state.current_time = zero(T)
    return state
end

"""
    update!(state::NetworkState, event::Event)

Update the network state with a new event.
"""
function update!(state::NetworkState{T}, event::Event{T}) where T
    s, r = event.sender, event.receiver
    t = event.time
    w = event.weight

    # Apply decay to all existing counts if time has advanced
    if state.decay > 0 && t > state.current_time
        apply_decay!(state, t)
    end

    state.current_time = t

    # Update dyad count
    dyad = (s, r)
    state.dyad_counts[dyad] = get(state.dyad_counts, dyad, 0.0) + w

    # Update undirected count
    undirected_dyad = minmax(s, r)
    state.undirected_counts[undirected_dyad] = get(state.undirected_counts, undirected_dyad, 0.0) + w

    # Update degrees
    state.out_degree[s] = get(state.out_degree, s, 0.0) + w
    state.in_degree[r] = get(state.in_degree, r, 0.0) + w

    # Update last event time
    state.last_event_time[dyad] = t

    # Add to history
    push!(state.event_history, (s, r, t, w))

    # Update incremental adjacency
    push!(get!(state.out_neighbors, s, Set{Int}()), r)
    push!(get!(state.in_neighbors, r, Set{Int}()), s)

    # Add actors if new
    push!(state.actors, s)
    push!(state.actors, r)
    state.n_actors = length(state.actors)

    return state
end

"""
    apply_decay!(state::NetworkState, new_time)

Apply exponential decay to all counts based on elapsed time.
"""
function apply_decay!(state::NetworkState{T}, new_time::T) where T
    if state.decay <= 0
        return state
    end

    elapsed = _elapsed_seconds(new_time - state.current_time)
    decay_factor = exp(-state.decay * elapsed)

    for (k, v) in state.dyad_counts
        state.dyad_counts[k] = v * decay_factor
    end

    for (k, v) in state.undirected_counts
        state.undirected_counts[k] = v * decay_factor
    end

    for (k, v) in state.out_degree
        state.out_degree[k] = v * decay_factor
    end

    for (k, v) in state.in_degree
        state.in_degree[k] = v * decay_factor
    end

    return state
end

"""
    get_dyad_count(state::NetworkState, sender::Int, receiver::Int) -> Float64

Get the weighted count of events from sender to receiver.
"""
function get_dyad_count(state::NetworkState, sender::Int, receiver::Int)
    return get(state.dyad_counts, (sender, receiver), 0.0)
end

"""
    get_undirected_count(state::NetworkState, actor1::Int, actor2::Int) -> Float64

Get the weighted count of events between two actors (in either direction).
"""
function get_undirected_count(state::NetworkState, actor1::Int, actor2::Int)
    return get(state.undirected_counts, minmax(actor1, actor2), 0.0)
end

"""
    get_out_degree(state::NetworkState, actor::Int) -> Float64

Get the weighted out-degree of an actor.
"""
function get_out_degree(state::NetworkState, actor::Int)
    return get(state.out_degree, actor, 0.0)
end

"""
    get_in_degree(state::NetworkState, actor::Int) -> Float64

Get the weighted in-degree of an actor.
"""
function get_in_degree(state::NetworkState, actor::Int)
    return get(state.in_degree, actor, 0.0)
end

const _EMPTY_NEIGHBORS = Set{Int}()

"""
    get_common_senders(state::NetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have sent events to both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets.
"""
function get_common_senders(state::NetworkState, actor1::Int, actor2::Int)
    return intersect(get_in_neighbors(state, actor1),
                     get_in_neighbors(state, actor2))
end

"""
    get_common_receivers(state::NetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have received events from both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets.
"""
function get_common_receivers(state::NetworkState, actor1::Int, actor2::Int)
    return intersect(get_out_neighbors(state, actor1),
                     get_out_neighbors(state, actor2))
end

"""
    get_out_neighbors(state::NetworkState, actor::Int) -> Set{Int}

Get the set of actors to whom the given actor has sent events.
Returns the internal set — treat as read-only.
"""
function get_out_neighbors(state::NetworkState, actor::Int)
    return get(state.out_neighbors, actor, _EMPTY_NEIGHBORS)
end

"""
    get_in_neighbors(state::NetworkState, actor::Int) -> Set{Int}

Get the set of actors who have sent events to the given actor.
Returns the internal set — treat as read-only.
"""
function get_in_neighbors(state::NetworkState, actor::Int)
    return get(state.in_neighbors, actor, _EMPTY_NEIGHBORS)
end

"""
    has_edge(state::NetworkState, sender::Int, receiver::Int) -> Bool

Check if there has been at least one event from sender to receiver.
"""
function has_edge(state::NetworkState, sender::Int, receiver::Int)
    return get_dyad_count(state, sender, receiver) > 0
end
