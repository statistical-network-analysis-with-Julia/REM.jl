"""
Statistics that read one event type, and exogenous covariates that change
over time.
"""

"""
    OfType(stat, eventtype::Symbol; name="<name(stat)>.<eventtype>")

`stat` computed on the events of **one event type** only: `OfType(Repetition(),
:fight)` is the count of past *fights* from sender to receiver, whatever else
the dyad has done. It wraps any statistic that reads the network state
(dyad, degree, triadic and four-cycle statistics, recency), under either
memory model.

This is how event types enter a REM.jl model (eventnet's type-filtered
attributes; remstats' `consider_type`): the candidate events of all types
compete in one risk set, and typed statistics let the history of each type
have its own effect. Fit a multi-type sequence with `eventtypes=:pool` (every
event is a case; untyped statistics count all types) or with
`cases_of=:fight` (only the fights are cases, every event still builds the
history) — see [`fit_rem`](@ref).

The typed counts live in a sub-state of the [`EventNetworkState`](@ref)
(`state.by_type[eventtype]`), which `fit_rem`, `generate_observations` and
`compute_statistics` create for the types their statistics name; a hand-built
state needs `EventNetworkState(seq; types=[...])`.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0; eventtype=:fight), Event(1, 2, 2.0; eventtype=:help),
                     Event(2, 1, 3.0; eventtype=:help), Event(1, 2, 4.0; eventtype=:fight)];
                    actors=ActorSet(1:3))
stats = [OfType(Repetition(), :fight), OfType(Reciprocity(), :help), Repetition()]
df = compute_statistics(seq, stats; eventtypes=:pool)
df[!, "repetition.fight"]     # [0.0, 1.0, 0.0, 1.0] — past fights 1→2 (resp. 2→1)
df[!, "reciprocity.help"]     # [0.0, 0.0, 1.0, 1.0] — past helps the other way
df.repetition                 # [0.0, 1.0, 0.0, 2.0] — all types together
```
"""
struct OfType{S<:AbstractStatistic} <: AbstractStatistic
    stat::S
    eventtype::Symbol
    stat_name::String

    function OfType(stat::S, eventtype::Symbol; name::String="") where {S<:AbstractStatistic}
        stat isa OfType && throw(ArgumentError(
            "OfType cannot wrap another OfType: an event has one type"))
        new{S}(stat, eventtype,
               isempty(name) ? string(REM.name(stat), ".", eventtype) : name)
    end
end

name(x::OfType) = x.stat_name
needs_history(x::OfType) = needs_history(x.stat)

function compute(x::OfType, state::EventNetworkState, sender::Int, receiver::Int)
    sub = get(state.by_type, x.eventtype, nothing)
    sub === nothing && _no_typed_state(x)
    # The sub-state reads its counts relative to the parent's clock (lazy
    # decay, window expiry), which the design loop moves without an event
    sub.current_time = state.current_time
    return compute(x.stat, sub, sender, receiver)
end

@noinline _no_typed_state(x::OfType) = throw(ArgumentError(
    "$(x.stat_name) reads the events of type :$(x.eventtype), but this " *
    "EventNetworkState keeps no typed counts for it: build the state with " *
    "`EventNetworkState(seq; types=[:$(x.eventtype)])` (`fit_rem`, " *
    "`generate_observations` and `compute_statistics` do this themselves)."))

# The event types whose sub-states a statistic set needs
_typed_needs(ss) = unique!(Symbol[s.eventtype for s in ss if s isa OfType])

# A typed statistic naming a type the sequence does not have is a typo, not
# a column of zeros
function _check_typed_needs(types::Vector{Symbol}, seq::EventSequence, context)
    for t in types
        t in seq.eventtypes || throw(ArgumentError(
            "$context: a statistic reads events of type :$t, but the sequence " *
            "has no event of that type (its types: " *
            join((":" * string(u) for u in sort!(collect(seq.eventtypes))), ", ") * ")"))
    end
    return types
end

"""
    TimeVaryingCovariate(f; name="covariate")

An **exogenous covariate that changes over time**: `f(sender, receiver, time)`
is evaluated for every candidate dyad at the time of the event being
explained (the state's clock), and must return a number. A policy that
starts at a known date, an actor's rank over time, a dyad's distance that
changes when someone moves — anything known from outside the event stream.

`f` is called once per row of the design, so keep it cheap and
allocation-free (index into arrays captured in a `let` block rather than
searching). It must not depend on the events themselves: that is what the
endogenous statistics are for.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0), Event(3, 1, 4.0)];
                    actors=ActorSet(1:3))
# actor 1 becomes a coordinator at t = 2.5: its calls count from then on
coordinator = TimeVaryingCovariate((s, r, t) -> s == 1 && t >= 2.5; name="sender_coordinator")
compute_statistics(seq, [coordinator]).sender_coordinator    # [0.0, 0.0, 1.0, 0.0]
```
"""
struct TimeVaryingCovariate{F} <: AbstractStatistic
    f::F
    stat_name::String

    TimeVaryingCovariate(f::F; name::String="covariate") where F = new{F}(f, name)
end

name(x::TimeVaryingCovariate) = x.stat_name
needs_history(::TimeVaryingCovariate) = false

compute(x::TimeVaryingCovariate, state::EventNetworkState, sender::Int, receiver::Int) =
    Float64(x.f(sender, receiver, state.current_time))
