"""
Core types for Relational Event Models.
"""

"""
    Event{T}

Represents a single relational event (directed interaction between actors).

# Fields
- `sender::Int`: ID of the event sender/source
- `receiver::Int`: ID of the event receiver/target
- `time::T`: Timestamp of the event
- `eventtype::Symbol`: Type/category of the event (default: :event)
- `weight::Float64`: Weight/magnitude of the event (default: 1.0)
"""
struct Event{T}
    sender::Int
    receiver::Int
    time::T
    eventtype::Symbol
    weight::Float64

    function Event{T}(sender::Int, receiver::Int, time::T,
                      eventtype::Symbol, weight::Float64) where T
        # Self-loops (sender == receiver) are permitted; exclude them from
        # risk sets via CaseControlSampler(exclude_self_loops=true) instead
        new{T}(sender, receiver, time, eventtype, weight)
    end
end

# Outer constructors
function Event(sender::Int, receiver::Int, time::T;
               eventtype::Symbol=:event, weight::Float64=1.0) where T
    Event{T}(sender, receiver, time, eventtype, weight)
end

Base.show(io::IO, e::Event) = print(io, "Event($(e.sender) → $(e.receiver) @ $(e.time))")

"""
    EventSequence{T}

A sequence of relational events, sorted by time.

The actor universe should be **declared**, not inferred: relational-event
likelihoods are conditional on the risk set, so an actor universe read off the
observed event endpoints silently drops eligible nonparticipants (isolates,
receiver-only actors) and changes the estimand. Pass `actors` to declare it:

```julia
seq = EventSequence(events; actors=ActorSet([1, 2, 3, 7]))  # 7 may be an isolate
```

If `actors` is omitted the universe falls back to the observed participants
only; `fit_rem` warns once when it is asked to fit against such a sequence
without an explicit risk set.

# Fields
- `events::Vector{Event{T}}`: Vector of events sorted by time
- `actors::Set{Int}`: Set of all actor IDs (the actor universe)
- `n_actors::Int`: Number of actors in the universe
- `eventtypes::Set{Symbol}`: Set of all event types
- `actors_declared::Bool`: True if the universe was supplied by the caller
  rather than inferred from event endpoints
"""
mutable struct EventSequence{T}
    events::Vector{Event{T}}
    actors::Set{Int}
    n_actors::Int
    eventtypes::Set{Symbol}
    actors_declared::Bool

    function EventSequence{T}(; actors=nothing) where T
        declared = !isnothing(actors)
        universe = declared ? actor_ids(actors) : Set{Int}()
        new{T}(Event{T}[], universe, length(universe), Set{Symbol}(), declared)
    end

    function EventSequence(events::Vector{Event{T}}; actors=nothing) where T
        sorted_events = sort(events, by=e -> e.time)
        declared = !isnothing(actors)
        universe = declared ? actor_ids(actors) : Set{Int}()
        eventtypes = Set{Symbol}()
        for e in sorted_events
            if declared
                (e.sender in universe && e.receiver in universe) ||
                    throw(ArgumentError(
                        "Event $(e.sender) → $(e.receiver) @ $(e.time) has an endpoint " *
                        "outside the declared actor universe"))
            else
                push!(universe, e.sender)
                push!(universe, e.receiver)
            end
            push!(eventtypes, e.eventtype)
        end
        new{T}(sorted_events, universe, length(universe), eventtypes, declared)
    end
end

Base.length(seq::EventSequence) = length(seq.events)
Base.iterate(seq::EventSequence, state=1) = state > length(seq.events) ? nothing : (seq.events[state], state + 1)
Base.getindex(seq::EventSequence, i) = seq.events[i]
Base.lastindex(seq::EventSequence) = length(seq.events)

function Base.push!(seq::EventSequence{T}, e::Event{T}) where T
    # Find insertion point to maintain sorted order
    idx = searchsortedfirst(seq.events, e, by=ev -> ev.time)
    if seq.actors_declared
        (e.sender in seq.actors && e.receiver in seq.actors) ||
            throw(ArgumentError(
                "Event $(e.sender) → $(e.receiver) @ $(e.time) has an endpoint " *
                "outside the declared actor universe"))
    end
    insert!(seq.events, idx, e)
    push!(seq.actors, e.sender)
    push!(seq.actors, e.receiver)
    push!(seq.eventtypes, e.eventtype)
    seq.n_actors = length(seq.actors)
    seq
end

"""
    ActorSet

Represents a set of actors with optional ID-to-name mapping.
"""
struct ActorSet
    ids::Vector{Int}
    id_to_name::Dict{Int, String}
    name_to_id::Dict{String, Int}

    function ActorSet(ids::Vector{Int})
        allunique(ids) || throw(ArgumentError("ActorSet IDs must be unique"))
        new(ids, Dict{Int, String}(), Dict{String, Int}())
    end

    function ActorSet(names::Vector{String})
        ids = collect(1:length(names))
        id_to_name = Dict(i => n for (i, n) in enumerate(names))
        name_to_id = Dict(n => i for (i, n) in enumerate(names))
        new(ids, id_to_name, name_to_id)
    end
end

# Convenience: accept any integer collection (ranges, UnitRange, ...)
ActorSet(ids::AbstractVector{<:Integer}) = ActorSet(collect(Int, ids))
ActorSet(ids::AbstractSet{<:Integer}) = ActorSet(sort!(collect(Int, ids)))

Base.length(as::ActorSet) = length(as.ids)
Base.in(id::Int, as::ActorSet) = id in as.ids
Base.iterate(as::ActorSet, state=1) = state > length(as.ids) ? nothing : (as.ids[state], state + 1)

"""
    actor_ids(actors) -> Set{Int}

Normalize an actor-universe specification (`ActorSet`, set, or vector of IDs)
into a `Set{Int}`. Used by `EventSequence(events; actors=...)` and by the
risk-set machinery in `generate_observations`.
"""
actor_ids(as::ActorSet) = Set{Int}(as.ids)
actor_ids(ids::AbstractSet{<:Integer}) = Set{Int}(ids)
actor_ids(ids::AbstractVector{<:Integer}) = Set{Int}(ids)
actor_ids(x) = throw(ArgumentError(
    "Cannot interpret $(typeof(x)) as an actor universe; pass an ActorSet, " *
    "a Set{Int}, or a Vector{Int}"))

"""
    NodeAttribute{T}

Stores an attribute value for each actor.

# Fields
- `name::Symbol`: Name of the attribute
- `values::Dict{Int, T}`: Mapping from actor ID to attribute value
- `default::T`: Default value for actors not in the dict
"""
struct NodeAttribute{T}
    name::Symbol
    values::Dict{Int, T}
    default::T

    function NodeAttribute(name::Symbol, values::Dict{Int, T}, default::T) where T
        new{T}(name, values, default)
    end

    function NodeAttribute(name::Symbol, default::T) where T
        new{T}(name, Dict{Int, T}(), default)
    end
end

Base.getindex(attr::NodeAttribute{T}, id::Int) where T = get(attr.values, id, attr.default)
Base.setindex!(attr::NodeAttribute{T}, val::T, id::Int) where T = attr.values[id] = val

"""
    RiskSet

Represents the risk set for a given event - the set of potential dyads that could
have experienced an event at a given time.

# Fields
- `event_index::Int`: Index of the focal event in the sequence
- `potential_senders::Vector{Int}`: Actors who could be senders
- `potential_receivers::Vector{Int}`: Actors who could be receivers
- `exclude_self_loops::Bool`: Whether to exclude self-loops from risk set
"""
struct RiskSet
    event_index::Int
    potential_senders::Vector{Int}
    potential_receivers::Vector{Int}
    exclude_self_loops::Bool

    function RiskSet(event_index::Int, senders::Vector{Int}, receivers::Vector{Int};
                     exclude_self_loops::Bool=true)
        new(event_index, senders, receivers, exclude_self_loops)
    end
end

"""
    n_dyads(rs::RiskSet)

Return the number of dyads in the risk set.
"""
function n_dyads(rs::RiskSet)
    n = length(rs.potential_senders) * length(rs.potential_receivers)
    if rs.exclude_self_loops
        # Subtract self-loops (only if sender can also be receiver)
        common = length(intersect(Set(rs.potential_senders), Set(rs.potential_receivers)))
        n -= common
    end
    return n
end
