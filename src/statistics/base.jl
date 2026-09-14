"""
Base types and interface for REM statistics.
"""

"""
    AbstractStatistic

Abstract base type for all REM statistics.

All statistics must implement:
- `compute(stat::AbstractStatistic, state::EventNetworkState, sender::Int, receiver::Int) -> Float64`
- `name(stat::AbstractStatistic) -> String`

Both are the shared generics of Networks.jl (`import REM: compute, name`, or
`import Networks: compute, name`, before adding methods — never a local
function of the same name). A statistic that reads the state's event log
should also say so with [`needs_history`](@ref) (the default for a foreign
statistic is `true`).

# Example
```julia
using REM
import REM: compute, name
# "Did the receiver send the previous event?" — reads the state's event log
struct ReplyToLast <: AbstractStatistic end
name(::ReplyToLast) = "reply_to_last"
compute(::ReplyToLast, state::EventNetworkState, s::Int, r::Int) =
    isempty(state.event_history) ? 0.0 : Float64(state.event_history[end][1] == r)

seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)];
                    actors=ActorSet(1:3))
compute_statistics(seq, [ReplyToLast(), Repetition()]).reply_to_last   # [0.0, 1.0, 1.0]
```
"""
abstract type AbstractStatistic end

"""
    DyadStatistic <: AbstractStatistic

Statistics that depend on the history of events between the focal dyad (sender, receiver).
Examples: repetition, reciprocity, inertia.

# Example
```julia
using REM
Repetition() isa DyadStatistic, Reciprocity() isa DyadStatistic   # (true, true)
needs_history(Repetition())                                       # false
```
"""
abstract type DyadStatistic <: AbstractStatistic end

"""
    DegreeStatistic <: AbstractStatistic

Statistics that depend on the degree (activity/popularity) of actors.
Examples: sender activity, receiver popularity.

# Example
```julia
using REM
SenderActivity() isa DegreeStatistic, LogDegree() isa DegreeStatistic   # (true, true)
```
"""
abstract type DegreeStatistic <: AbstractStatistic end

"""
    TriangleStatistic <: AbstractStatistic

Statistics that measure triadic closure effects.
Examples: transitive closure, cyclic closure, shared partners.

# Example
```julia
using REM
TransitiveClosure() isa TriangleStatistic     # true
GeometricWeightedTriads() isa TriangleStatistic   # true
```
"""
abstract type TriangleStatistic <: AbstractStatistic end

"""
    FourCycleStatistic <: AbstractStatistic

Statistics that measure four-cycle (local clustering) effects.

# Example
```julia
using REM
FourCycle() isa FourCycleStatistic, GeometricWeightedFourCycles() isa FourCycleStatistic
```
"""
abstract type FourCycleStatistic <: AbstractStatistic end

"""
    NodeStatistic <: AbstractStatistic

Statistics based on node-level attributes.
Examples: homophily, attribute matching.

# Example
```julia
using REM
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"))
AttributeMatch(gender) isa NodeStatistic     # true
```
"""
abstract type NodeStatistic <: AbstractStatistic end

"""
    InteractionStatistic <: AbstractStatistic

Statistics that capture interaction effects between attributes. REM ships no
concrete subtype; it is the slot for a user-defined product of two attribute
effects (`needs_history` is `false` for it, as for every REM family).

# Example
```julia
using REM
import REM: compute, name
struct SameTeamSenior <: InteractionStatistic
    team::NodeAttribute{String}
    age::NodeAttribute{Float64}
end
name(::SameTeamSenior) = "same_team_senior"
compute(x::SameTeamSenior, ::EventNetworkState, s::Int, r::Int) =
    Float64(x.team[s] == x.team[r]) * (x.age[s] - x.age[r])
needs_history(SameTeamSenior(NodeAttribute(:t, "a"), NodeAttribute(:a, 0.0)))   # false
```
"""
abstract type InteractionStatistic <: AbstractStatistic end

"""
    compute(stat::AbstractStatistic, state::EventNetworkState, sender::Int, receiver::Int) -> Float64

Compute the statistic value for a potential event from sender to receiver, read
off `state` as it stands **now** (before the candidate event). This is the main
interface that all statistics must implement — as a method of the shared
`Networks.compute` generic (the same `compute` ERGM.jl's terms answer to, with
a different signature), which is why `using ERGM, REM` leaves it usable.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(Repetition(), state, 1, 2)          # 2.0
compute(Reciprocity(), state, 2, 1)         # 2.0
compute(TransitiveClosure(), state, 1, 3)   # 1.0 — min(w(1,2), w(2,3)) = min(2, 1)
```
"""
function compute(stat::AbstractStatistic, state::EventNetworkState, sender::Int, receiver::Int)
    error("compute() not implemented for $(typeof(stat))")
end

"""
    name(stat::AbstractStatistic) -> String

Return the statistic's name — the column it gets in the observations frame and
the row it gets in the coefficient table. Every REM statistic accepts a
`name=` keyword to override its default (two variants of one statistic in one
model need distinct names). A method of the shared `Networks.name` generic.

# Example
```julia
using REM
name(Repetition())                              # "repetition"
name(Repetition(directed=false))                # "undirected_repetition"
name(TransitiveClosure(name="closure_min"))     # "closure_min"
```
"""
function name(stat::AbstractStatistic)
    return string(typeof(stat))
end

"""
    needs_history(stat::AbstractStatistic) -> Bool

Whether `stat` reads the per-event log `state.event_history` of an
[`EventNetworkState`](@ref). `generate_observations`, `compute_statistics` and
`fit_rem` build their state with `keep_history = any(needs_history, stats)`, so
the O(events) log is retained only when a statistic in the set consumes it.

**The default is `true`** — a statistic REM does not know about (one defined in
Relevent.jl or by a user) is assumed to need the log until it says otherwise —
and every statistic REM ships returns `false` (they read the incremental
counts, degrees, last-event times and adjacency). A package whose statistics
never touch the log can declare it: `REM.needs_history(::MyStat) = false`
(`import REM: needs_history` first).

# Example
```julia
using REM
needs_history(Repetition())          # false — reads the dyad count
needs_history(TransitiveClosure())   # false — reads the adjacency sets

struct LastSenderRepeats <: AbstractStatistic end   # reads the log: default true
needs_history(LastSenderRepeats())   # true
```
"""
needs_history(::AbstractStatistic) = true
needs_history(::DyadStatistic) = false
needs_history(::DegreeStatistic) = false
needs_history(::TriangleStatistic) = false
needs_history(::FourCycleStatistic) = false
needs_history(::NodeStatistic) = false
needs_history(::InteractionStatistic) = false

"""
    StatisticSet

A collection of statistics to compute together.

Statistics are stored as a tuple so that `compute_all` compiles to
statically dispatched calls per statistic instead of dynamic dispatch
through an abstractly-typed vector (which would dominate the
observation-generation/likelihood inner loop). Construct from a tuple or
a vector of statistics; `generate_observations`, `compute_statistics`,
and `fit_rem` convert vectors to a `StatisticSet` internally.

# Example
```julia
using REM
ss = StatisticSet([Repetition(), Reciprocity(), TransitiveClosure()])
length(ss), ss.names        # (3, ["repetition", "reciprocity", "transitive_closure"])
ss[1]                       # Repetition()
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0)]; actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute_all(ss, state, 1, 2)   # [1.0, 1.0, 0.0]
```
"""
struct StatisticSet{T<:Tuple}
    statistics::T
    names::Vector{String}

    function StatisticSet(stats::T) where {T<:Tuple}
        all(s -> s isa AbstractStatistic, stats) ||
            throw(ArgumentError("all elements must be AbstractStatistics"))
        names = [name(s) for s in stats]
        new{T}(stats, names)
    end
end

StatisticSet(stats::Vector{<:AbstractStatistic}) = StatisticSet(Tuple(stats))

Base.length(ss::StatisticSet) = length(ss.statistics)
# Does any statistic in the set read the event log? (Decides `keep_history`.)
_keeps_history(ss::StatisticSet) = any(needs_history, ss.statistics)
Base.iterate(ss::StatisticSet, state=1) = state > length(ss) ? nothing : (ss.statistics[state], state + 1)
Base.getindex(ss::StatisticSet, i) = ss.statistics[i]

# The names, not the tuple's type parameters
function Base.show(io::IO, ss::StatisticSet)
    n = length(ss)
    shown = n > 8 ? vcat(ss.names[1:7], "… ($(n - 7) more)") : ss.names
    print(io, "StatisticSet(", n, " statistic", n == 1 ? "" : "s", ": ",
          join(shown, ", "), ")")
end

"""
    compute_all(ss::StatisticSet, state::EventNetworkState, sender::Int, receiver::Int) -> Vector{Float64}

Compute all statistics in the set for a potential event, in the set's order
(one fresh `Vector{Float64}`; see [`compute_all!`](@ref) for the in-place
form). A method of the shared `Networks.compute_all` generic.

# Example
```julia
using REM
ss = StatisticSet([Repetition(), Reciprocity()])
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)];
                    actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute_all(ss, state, 1, 2)     # [2.0, 1.0]
compute_all([Repetition(), Reciprocity()], state, 2, 1)   # [1.0, 2.0] (vector form)
```
"""
function compute_all(ss::StatisticSet, state::EventNetworkState, sender::Int, receiver::Int)
    return compute_all!(Vector{Float64}(undef, length(ss)), ss, state, sender, receiver)
end

"""
    compute_all!(dest, ss::StatisticSet, state::EventNetworkState, sender::Int, receiver::Int) -> dest

In-place version of [`compute_all`](@ref) for use in sampling loops. The
per-statistic calls are unrolled at compile time (a generated function over the
tuple), so the loop is statically dispatched and allocation-free for a set of
any length — `map` over a tuple longer than 32 falls back to a generic path
that allocates.

# Example
```julia
using REM
ss = StatisticSet([Repetition(), Reciprocity()])
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
dest = Vector{Float64}(undef, length(ss))
compute_all!(dest, ss, state, 1, 2)     # [1.0, 1.0], written into dest
```
"""
@generated function compute_all!(dest::AbstractVector{Float64}, ss::StatisticSet{T},
                                 state::EventNetworkState, sender::Int, receiver::Int) where {T<:Tuple}
    n = length(T.parameters)
    body = Expr(:block)
    push!(body.args, :(length(dest) >= $n || throw(DimensionMismatch(
        "compute_all!: dest has length $(length(dest)) for $($n) statistics"))))
    for k in 1:n
        push!(body.args, :(@inbounds dest[$k] = compute(ss.statistics[$k], state, sender, receiver)))
    end
    push!(body.args, :(return dest))
    return body
end

"""
    compute_all(stats::Vector{<:AbstractStatistic}, state::EventNetworkState, sender::Int, receiver::Int) -> Vector{Float64}

Compute all statistics for a potential event.
"""
function compute_all(stats::Vector{<:AbstractStatistic}, state::EventNetworkState, sender::Int, receiver::Int)
    return [compute(s, state, sender, receiver) for s in stats]
end
