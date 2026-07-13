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
    EventNetworkState

Tracks the cumulative state of the network up to a given point in time.
Used for efficient computation of statistics.

Decay is applied *lazily*: every weighted count is stored as a
`(value, last_update_time)` pair and decayed on read relative to
`current_time`, so updating the state with an event is O(1) instead of
O(number of nonzero counts).

# Fields
- `n_actors::Int`: Number of actors
- `dyad_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}`: `(weighted count, last update)` per directed dyad
- `undirected_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}`: `(weighted count, last update)` per undirected dyad (min,max sorted)
- `out_degree::Dict{Int, Tuple{Float64,T}}`: `(weighted out-degree, last update)` per actor
- `in_degree::Dict{Int, Tuple{Float64,T}}`: `(weighted in-degree, last update)` per actor
- `last_event_time::Dict{Tuple{Int,Int}, T}`: Time of last event for each dyad
- `decay::Float64`: Exponential decay rate (0 = no decay)
- `current_time::T`: Current time in the event sequence
"""
mutable struct EventNetworkState{T}
    n_actors::Int
    actors::Set{Int}
    dyad_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}
    undirected_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}
    out_degree::Dict{Int, Tuple{Float64,T}}
    in_degree::Dict{Int, Tuple{Float64,T}}
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

    function EventNetworkState{T}(; n_actors::Int=0, decay::Float64=0.0) where T
        new{T}(
            n_actors,
            Set{Int}(),
            Dict{Tuple{Int,Int}, Tuple{Float64,T}}(),
            Dict{Tuple{Int,Int}, Tuple{Float64,T}}(),
            Dict{Int, Tuple{Float64,T}}(),
            Dict{Int, Tuple{Float64,T}}(),
            Dict{Tuple{Int,Int}, T}(),
            decay,
            zero(T),
            Tuple{Int,Int,T,Float64}[],
            Dict{Int, Set{Int}}(),
            Dict{Int, Set{Int}}()
        )
    end
end

# Decay factor for a value stored at `t_stored`, read at the state's
# `current_time` (1.0 when decay is off or time has not advanced).
function _decay_to_now(state::EventNetworkState{T}, t_stored::T) where T
    (state.decay > 0 && state.current_time > t_stored) || return 1.0
    return exp(-state.decay * _elapsed_seconds(state.current_time - t_stored))
end

# Read a lazily decayed count (0.0 for absent keys).
function _lazy_get(state::EventNetworkState{T}, dict::Dict{K, Tuple{Float64,T}},
                   key::K) where {T, K}
    entry = get(dict, key, nothing)
    entry === nothing && return 0.0
    value, t_stored = entry
    return value * _decay_to_now(state, t_stored)
end

# Add `w` to a lazily decayed count at time `t`, decaying the stored value
# from its own last-update time first.
function _lazy_add!(state::EventNetworkState{T}, dict::Dict{K, Tuple{Float64,T}},
                    key::K, w::Float64, t::T) where {T, K}
    entry = get(dict, key, nothing)
    if entry === nothing
        dict[key] = (w, t)
    else
        value, t_stored = entry
        if state.decay > 0 && t > t_stored
            value *= exp(-state.decay * _elapsed_seconds(t - t_stored))
        end
        dict[key] = (value + w, t)
    end
    return nothing
end

"""
    EventNetworkState(seq::EventSequence; decay=0.0)

Create a EventNetworkState from an EventSequence without processing any events.
"""
function EventNetworkState(seq::EventSequence{T}; decay::Float64=0.0) where T
    state = EventNetworkState{T}(n_actors=seq.n_actors, decay=decay)
    state.actors = copy(seq.actors)
    return state
end

"""
    reset!(state::EventNetworkState)

Reset the network state to empty.
"""
function reset!(state::EventNetworkState{T}) where T
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
    update!(state::EventNetworkState, event::Event)

Update the network state with a new event.
"""
function update!(state::EventNetworkState{T}, event::Event{T}) where T
    s, r = event.sender, event.receiver
    t = event.time
    w = event.weight

    # Counts decay lazily on read; updating touches only the event's own
    # keys (O(1) per event instead of O(number of nonzero counts))
    state.current_time = t

    # Update dyad count
    dyad = (s, r)
    _lazy_add!(state, state.dyad_counts, dyad, w, t)

    # Update undirected count
    _lazy_add!(state, state.undirected_counts, minmax(s, r), w, t)

    # Update degrees
    _lazy_add!(state, state.out_degree, s, w, t)
    _lazy_add!(state, state.in_degree, r, w, t)

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
    apply_decay!(state::EventNetworkState, new_time)

Materialize the exponential decay of all counts up to `new_time` (each
entry is rewritten as its decayed value stamped at `new_time`).

Since counts decay lazily on read, calling this is never required for
correctness; it exists for backward compatibility and does not change the
values returned by the accessors.
"""
function apply_decay!(state::EventNetworkState{T}, new_time::T) where T
    if state.decay <= 0
        return state
    end

    for dict in (state.dyad_counts, state.undirected_counts)
        _materialize_decay!(state, dict, new_time)
    end
    for dict in (state.out_degree, state.in_degree)
        _materialize_decay!(state, dict, new_time)
    end

    return state
end

function _materialize_decay!(state::EventNetworkState{T},
                             dict::Dict{K, Tuple{Float64,T}}, new_time::T) where {T, K}
    for (k, (value, t_stored)) in dict
        if new_time > t_stored
            elapsed = _elapsed_seconds(new_time - t_stored)
            dict[k] = (value * exp(-state.decay * elapsed), new_time)
        end
    end
    return nothing
end

"""
    get_dyad_count(state::EventNetworkState, sender::Int, receiver::Int) -> Float64

Get the weighted count of events from sender to receiver.
"""
function get_dyad_count(state::EventNetworkState, sender::Int, receiver::Int)
    return _lazy_get(state, state.dyad_counts, (sender, receiver))
end

"""
    get_undirected_count(state::EventNetworkState, actor1::Int, actor2::Int) -> Float64

Get the weighted count of events between two actors (in either direction).
"""
function get_undirected_count(state::EventNetworkState, actor1::Int, actor2::Int)
    return _lazy_get(state, state.undirected_counts, minmax(actor1, actor2))
end

"""
    get_out_degree(state::EventNetworkState, actor::Int) -> Float64

Get the weighted out-degree of an actor.
"""
function get_out_degree(state::EventNetworkState, actor::Int)
    return _lazy_get(state, state.out_degree, actor)
end

"""
    get_in_degree(state::EventNetworkState, actor::Int) -> Float64

Get the weighted in-degree of an actor.
"""
function get_in_degree(state::EventNetworkState, actor::Int)
    return _lazy_get(state, state.in_degree, actor)
end

const _EMPTY_NEIGHBORS = Set{Int}()

"""
    get_common_senders(state::EventNetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have sent events to both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets.
"""
function get_common_senders(state::EventNetworkState, actor1::Int, actor2::Int)
    return intersect(get_in_neighbors(state, actor1),
                     get_in_neighbors(state, actor2))
end

"""
    get_common_receivers(state::EventNetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have received events from both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets.
"""
function get_common_receivers(state::EventNetworkState, actor1::Int, actor2::Int)
    return intersect(get_out_neighbors(state, actor1),
                     get_out_neighbors(state, actor2))
end

"""
    get_out_neighbors(state::EventNetworkState, actor::Int) -> Set{Int}

Get the set of actors to whom the given actor has sent events.
Returns the internal set — treat as read-only.
"""
function get_out_neighbors(state::EventNetworkState, actor::Int)
    return get(state.out_neighbors, actor, _EMPTY_NEIGHBORS)
end

"""
    get_in_neighbors(state::EventNetworkState, actor::Int) -> Set{Int}

Get the set of actors who have sent events to the given actor.
Returns the internal set — treat as read-only.
"""
function get_in_neighbors(state::EventNetworkState, actor::Int)
    return get(state.in_neighbors, actor, _EMPTY_NEIGHBORS)
end

"""
    has_edge(state::EventNetworkState, sender::Int, receiver::Int) -> Bool

Check if there has been at least one event from sender to receiver.

A method of the SHARED `Graphs.has_edge` generic (re-exported by Networks.jl),
not a REM-local function of the same name — asking "is there a tie from i to j"
of an accumulated event network is the same question Graphs asks of a graph.
That is why it can be exported without colliding: `using Graphs, REM` (or
`using ERGM, REM`) dispatches one generic on the state type.
"""
function has_edge(state::EventNetworkState, sender::Int, receiver::Int)
    return get_dyad_count(state, sender, receiver) > 0
end
