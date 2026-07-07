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

# Fields
- `events::Vector{Event{T}}`: Vector of events sorted by time
- `actors::Set{Int}`: Set of all actor IDs
- `n_actors::Int`: Number of unique actors
- `eventtypes::Set{Symbol}`: Set of all event types
"""
mutable struct EventSequence{T}
    events::Vector{Event{T}}
    actors::Set{Int}
    n_actors::Int
    eventtypes::Set{Symbol}

    function EventSequence{T}() where T
        new{T}(Event{T}[], Set{Int}(), 0, Set{Symbol}())
    end

    function EventSequence(events::Vector{Event{T}}) where T
        sorted_events = sort(events, by=e -> e.time)
        actors = Set{Int}()
        eventtypes = Set{Symbol}()
        for e in sorted_events
            push!(actors, e.sender)
            push!(actors, e.receiver)
            push!(eventtypes, e.eventtype)
        end
        new{T}(sorted_events, actors, length(actors), eventtypes)
    end
end

Base.length(seq::EventSequence) = length(seq.events)
Base.iterate(seq::EventSequence, state=1) = state > length(seq.events) ? nothing : (seq.events[state], state + 1)
Base.getindex(seq::EventSequence, i) = seq.events[i]
Base.lastindex(seq::EventSequence) = length(seq.events)

function Base.push!(seq::EventSequence{T}, e::Event{T}) where T
    # Find insertion point to maintain sorted order
    idx = searchsortedfirst(seq.events, e, by=ev -> ev.time)
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
        new(ids, Dict{Int, String}(), Dict{String, Int}())
    end

    function ActorSet(names::Vector{String})
        ids = collect(1:length(names))
        id_to_name = Dict(i => n for (i, n) in enumerate(names))
        name_to_id = Dict(n => i for (i, n) in enumerate(names))
        new(ids, id_to_name, name_to_id)
    end
end

Base.length(as::ActorSet) = length(as.ids)
Base.in(id::Int, as::ActorSet) = id in as.ids

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
