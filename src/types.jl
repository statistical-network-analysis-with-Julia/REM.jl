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

`T` is the clock: `Float64`/`Int` for an abstract or ordinal time, `Date` or
`DateTime` for calendar time (elapsed time is then measured in **seconds** by
the decay and window machinery). A self-loop (`sender == receiver`) is
permitted at construction; exclude it from the risk set with
`CaseControlSampler(exclude_self_loops=true)` (the default).

# Example
```julia
using REM, Dates
e = Event(1, 2, 3.5)                                   # 1 → 2 at t = 3.5
e.sender, e.receiver, e.time                           # (1, 2, 3.5)
Event(1, 2, 3.5; eventtype=:email, weight=2.0).weight   # 2.0
Event(1, 2, DateTime(2024, 1, 15, 10, 30))              # calendar clock
```
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
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
seq = EventSequence(events; actors=ActorSet([1, 2, 3, 7]))  # 7 may be an isolate
length(seq), seq.n_actors, seq.actors_declared               # (3, 4, true)
seq[1], seq[end].time                                        # (Event(1 → 2 @ 1.0), 3.0)
[e.receiver for e in seq]                                    # [2, 1, 3]
push!(seq, Event(7, 1, 2.5))                                 # inserted in time order
[e.time for e in seq]                                        # [1.0, 2.0, 2.5, 3.0]
inferred = EventSequence(events)
inferred.actors_declared                                     # false — inferred
EventSequence(collect(inferred); actors=ActorSet(1:7)).actors_declared   # true
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

`show(seq)` is one line — `EventSequence{Float64}(481 events, 37 actors,
declared universe)` — not the event vector.
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
Base.firstindex(seq::EventSequence) = 1
Base.lastindex(seq::EventSequence) = length(seq.events)
Base.eachindex(seq::EventSequence) = 1:length(seq.events)
# So that `collect(seq)` is a `Vector{Event{T}}` (and can be handed back to
# `EventSequence(events; actors=...)` to declare a universe), not `Vector{Any}`
Base.eltype(::Type{EventSequence{T}}) where T = Event{T}

# One line: size, universe and whether the universe was declared (the estimand
# question), never the event vector
function Base.show(io::IO, seq::EventSequence{T}) where T
    n = length(seq)
    print(io, "EventSequence{", T, "}(", n, " event", n == 1 ? "" : "s", ", ",
          seq.n_actors, " actor", seq.n_actors == 1 ? "" : "s", ", ",
          seq.actors_declared ? "declared" : "inferred", " universe")
    length(seq.eventtypes) > 1 && print(io, ", ", length(seq.eventtypes), " event types")
    print(io, ")")
end

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
    ActorSet(ids::AbstractVector{<:Integer})
    ActorSet(ids::AbstractSet{<:Integer})
    ActorSet(names::Vector{String})

The actor universe: the set of actors eligible to send and receive, with an
optional ID-to-name mapping. Pass it as `actors=` to [`EventSequence`](@ref)
to declare the risk set the likelihood is conditional on — isolates and
noncontiguous IDs included. Names are assigned the IDs `1:length(names)`.

# Fields
- `ids::Vector{Int}`: the actor IDs (unique)
- `id_to_name::Dict{Int,String}`, `name_to_id::Dict{String,Int}`: the name
  mapping (empty when the set was built from IDs)

# Example
```julia
using REM
actors = ActorSet(1:37)                      # a range: 37 eligible actors
length(actors), 40 in actors                 # (37, false)
named = ActorSet(["Alice", "Bob", "Carol"])
named.name_to_id["Carol"], named.id_to_name[1]   # (3, "Alice")
seq = EventSequence([Event(1, 2, 1.0)]; actors=actors)   # actors 3…37 are isolates
seq.n_actors                                 # 37
```
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
function Base.show(io::IO, as::ActorSet)
    n = length(as)
    print(io, "ActorSet(", n, isempty(as.id_to_name) ? "" : " named", " actor",
          n == 1 ? "" : "s", ")")
end
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

# Sentinel for "no default value" in a NodeAttribute (a value of type T cannot
# be conjured for an arbitrary T, and `nothing` could itself be a value)
struct _NoDefault end
const _NO_DEFAULT = _NoDefault()
Base.show(io::IO, ::_NoDefault) = print(io, "no default")

# The message every entry point gives a `Vector{Event}` passed where an
# `EventSequence` is expected: what is missing is the actor universe, which
# is the estimand of every relational-event likelihood.
function _event_vector_hint(fname::AbstractString)
    return "$fname expects an EventSequence, not a Vector of Events. Wrap the " *
           "events and DECLARE the actor universe — the risk set the likelihood " *
           "is conditional on — with `EventSequence(events; actors=ActorSet(ids))` " *
           "(isolates included; `actors=` may be an ActorSet, a Set{Int}, a " *
           "Vector{Int} or a range)."
end

# The message for `f(stats, seq)` — the two positional arguments in the
# other order (an easy slip coming from `rem.dyad(edgelist, n, effects=...)`).
function _swapped_arguments_hint(fname::AbstractString)
    return "$fname takes the EventSequence FIRST and the statistics second: " *
           "`$fname(seq, [Repetition(), Reciprocity()]; ...)` (a single statistic " *
           "may be passed bare, `$fname(seq, Repetition(); ...)`)."
end

"""
    NodeAttribute{T}

Stores an attribute value for each actor.

    NodeAttribute(name, values::Dict{Int,T})            # no default: a missing actor is an error
    NodeAttribute(name, values::Dict{Int,T}, default)   # explicit default for missing actors
    NodeAttribute(name, default)                        # empty, fill with `attr[id] = value`

Reading `attr[id]` for an actor the attribute has **no value for** throws an
`ArgumentError` naming the actor unless a `default` was given explicitly.
Silently filling a missing value (with `0.0`, say) is the classic covariate
trap — a zero that looks like data — so the fill is opt-in: pass `default` to
say that missing actors *should* read as that value.

# Fields
- `name::Symbol`: Name of the attribute
- `values::Dict{Int, T}`: Mapping from actor ID to attribute value
- `default::Union{_NoDefault,T}`: Default value for actors not in the dict,
  or the `_NoDefault` sentinel when none was given (see [`has_default`](@ref))

`show` prints one line — `NodeAttribute{Float64}(:age, 2 actors, no default)`
— never the values `Dict`, so a vector of statistics wrapping an attribute
stays readable in the REPL.

# Example
```julia
using REM
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))          # no default
age[1]                                   # 25.0
try age[3] catch err; err isa ArgumentError end   # true — actor 3 has no age
filled = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0), 0.0)  # explicit default
filled[3]                                # 0.0, deliberately
sprint(show, filled)                     # "NodeAttribute{Float64}(:age, 2 actors, default = 0.0)"
```
"""
struct NodeAttribute{T}
    name::Symbol
    values::Dict{Int, T}
    default::Union{_NoDefault, T}

    function NodeAttribute(name::Symbol, values::Dict{Int, T}, default::T) where T
        new{T}(name, values, default)
    end

    function NodeAttribute(name::Symbol, values::Dict{Int, T}) where T
        new{T}(name, values, _NO_DEFAULT)
    end

    function NodeAttribute(name::Symbol, default::T) where T
        new{T}(name, Dict{Int, T}(), default)
    end
end

"""
    has_default(attr::NodeAttribute) -> Bool

Whether `attr` fills a missing actor with a default value (`true` only when
one was passed to the constructor). Without one, reading an actor the
attribute lacks is an `ArgumentError` — the deliberate alternative to a
silent zero-fill.

# Example
```julia
using REM
strict = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
filled = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0), 0.0)
has_default(strict), has_default(filled)     # (false, true)
```
"""
has_default(attr::NodeAttribute) = !(attr.default isa _NoDefault)

# One informative line (criterion 5), never the values Dict: the statistics
# that wrap an attribute (`NodeSum(icr)`, ...) inherit it through the default
# struct `show`
function Base.show(io::IO, a::NodeAttribute{T}) where T
    n = length(a.values)
    print(io, "NodeAttribute{", T, "}(:", a.name, ", ", n, " actor", n == 1 ? "" : "s",
          has_default(a) ? ", default = " * repr(a.default) : ", no default", ")")
end

function Base.getindex(attr::NodeAttribute{T}, id::Int) where T
    v = get(attr.values, id, _NO_DEFAULT)
    v isa _NoDefault || return v
    d = attr.default
    d isa _NoDefault && throw(ArgumentError(
        "NodeAttribute :$(attr.name) has no value for actor $id and no default. " *
        "Either give every actor in the risk set a value, or construct the " *
        "attribute with an explicit default — `NodeAttribute(:$(attr.name), " *
        "values, default)` — to fill missing actors deliberately (a silent " *
        "zero-fill would enter the model as data)."))
    return d
end
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

A `RiskSet` is what `at_risk=` accepts when senders and receivers differ, or
per event; [`n_dyads`](@ref) counts its dyads. The likelihood is conditional
on it, so every case must belong to its own risk set (checked before fitting).

# Example
```julia
using REM
rs = RiskSet(5, [1, 2, 3], [1, 2, 3, 4])     # focal event 5; 4 may only receive
n_dyads(rs)                                  # 9 = 3·4 − 3 self-loops
RiskSet(5, [1, 2], [3, 4]; exclude_self_loops=false)   # disjoint sides: 4 dyads
```
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

function Base.show(io::IO, rs::RiskSet)
    print(io, "RiskSet(event ", rs.event_index, ": ", length(rs.potential_senders),
          " senders × ", length(rs.potential_receivers), " receivers, ", n_dyads(rs),
          " dyads, self-loops ", rs.exclude_self_loops ? "excluded" : "included", ")")
end

"""
    n_dyads(rs::RiskSet) -> Int

Return the number of dyads in the risk set: `|senders| × |receivers|`, minus the
actors that appear on both sides when `exclude_self_loops` is set.

Allocation-free: when both actor vectors are sorted (which every `RiskSet`
built by `generate_observations` is) the common actors are counted by a
single sorted merge; unsorted or duplicated vectors are counted through sets
instead, so a hand-built `RiskSet` still gets the right answer.

# Example
```julia
using REM
n_dyads(RiskSet(1, [1, 2, 3], [1, 2, 3]))                            # 6
n_dyads(RiskSet(1, [1, 2, 3], [1, 2, 3]; exclude_self_loops=false))  # 9
n_dyads(RiskSet(1, [1, 2], [3, 4, 5]))                               # 6 (disjoint)
```
"""
function n_dyads(rs::RiskSet)
    senders = rs.potential_senders
    receivers = rs.potential_receivers
    n = length(senders) * length(receivers)
    rs.exclude_self_loops || return n
    # Subtract self-loops (only if sender can also be receiver)
    if _strictly_sorted(senders) && _strictly_sorted(receivers)
        return n - _count_common_sorted(senders, receivers)
    end
    return n - length(intersect(Set(senders), Set(receivers)))
end

# Strictly increasing (sorted and duplicate-free), without allocating
function _strictly_sorted(v::AbstractVector{Int})
    @inbounds for i in 2:length(v)
        v[i] > v[i - 1] || return false
    end
    return true
end

# Size of the intersection of two strictly increasing vectors by a sorted merge
# — O(|a| + |b|), zero allocation
function _count_common_sorted(a::AbstractVector{Int}, b::AbstractVector{Int})
    i, j, c = 1, 1, 0
    na, nb = length(a), length(b)
    @inbounds while i <= na && j <= nb
        if a[i] == b[j]
            c += 1; i += 1; j += 1
        elseif a[i] < b[j]
            i += 1
        else
            j += 1
        end
    end
    return c
end
