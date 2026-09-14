"""
Network state tracking for computing statistics efficiently.
"""

# What the `window=` keyword accepts, everywhere it appears (`EventNetworkState`,
# `generate_observations`, `compute_statistics`, `fit_rem`, `control_draw_cov`):
# `nothing`/`Inf` for no window, a `Real` in the clock's units, or — on a
# `Date`/`DateTime` clock, whose unit is the second — a `Dates.Period`
# (`Day(2)`), so the calendar idiom works instead of failing with a TypeError
# that never mentions seconds.
const _WindowSpec = Union{Nothing, Real, Dates.Period, Dates.CompoundPeriod}

# Convert a time difference into seconds, handling both numeric types and Dates periods.
function _elapsed_seconds(diff)
    if diff isa Real
        return Float64(diff)
    elseif diff isa Dates.CompoundPeriod
        # A sum of periods (`Day(1) + Hour(12)`): seconds of each part
        return sum(_elapsed_seconds(p) for p in diff.periods; init=0.0)
    elseif diff isa Dates.Period
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

Memory is O(actors + active dyads) — independent of the number of events
absorbed — unless the event log is kept: `keep_history` (default `true` for a
hand-built state) retains one `(sender, receiver, time, weight)` tuple per
absorbed event in `event_history`, the **public, read-only** log that
history-reading statistics consume (Relevent.jl's `PShift`, `Momentum`,
`PriorInteraction`, ...). None of REM's own statistics reads it, so
[`generate_observations`](@ref) and [`compute_statistics`](@ref) build their
state with `keep_history = any(needs_history, stats)` (see
[`needs_history`](@ref)): a fit on REM's statistics retains no per-event
memory, and a statistic that needs the log gets it.

# Two memory models: halflife decay or a sliding window

- `decay` (eventnet's model, and the only one eventnet offers): every count
  is stored as a `(value, last_update_time)` pair and decayed *lazily* on
  read relative to `current_time`, so updating the state with an event is
  O(1) instead of O(number of nonzero counts). Adjacency ("ever had an
  event") never expires under decay — counts fade, structure does not.
- `window` (the fixed-memory alternative found in other REM software, e.g.
  `remstats`' `memory = "window"`): an event older than
  `current_time − window` **stops counting** — in the dyad counts, the
  undirected counts, the degrees *and* the adjacency sets (a dyad whose
  events have all expired has no edge). Implemented with a FIFO of absorbed
  events and an expiry cursor advanced when the clock moves, amortized O(1)
  per event. `window` is in the units of the clock for numeric time and in
  **seconds** for `Date`/`DateTime` time — on a calendar clock pass a
  `Dates.Period` (`window = Day(2)`, converted to seconds) rather than
  counting seconds by hand; an event exactly `window` old still counts.
  `window = Inf` (or `nothing`, the default) is no window.

The two are **mutually exclusive**: `decay > 0` together with a finite
`window` is an `ArgumentError`. `last_event_time` (what
[`RecencyStatistic`](@ref) reads) and `event_history` are not windowed —
the time since a dyad's last event is a fact about the clock, not a count.

# Fields
- `n_actors::Int`: Number of actors
- `dyad_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}`: `(weighted count, last update)` per directed dyad
- `undirected_counts::Dict{Tuple{Int,Int}, Tuple{Float64,T}}`: `(weighted count, last update)` per undirected dyad (min,max sorted)
- `out_degree::Dict{Int, Tuple{Float64,T}}`: `(weighted out-degree, last update)` per actor
- `in_degree::Dict{Int, Tuple{Float64,T}}`: `(weighted in-degree, last update)` per actor
- `last_event_time::Dict{Tuple{Int,Int}, T}`: Time of last event for each dyad
- `decay::Float64`: Exponential decay rate (0 = no decay)
- `window::Float64`: Sliding-window length (`Inf` = no window)
- `current_time::T`: Current time in the event sequence
- `event_history::Vector{Tuple{Int,Int,T,Float64}}`: The absorbed events as
  `(sender, receiver, time, weight)`, in order — public and read-only; empty
  unless `keep_history` is set
- `keep_history::Bool`: Whether `update!` appends to `event_history`
- `n_events::Int`: Number of events absorbed by `update!` since the last
  `reset!` (what `show` reports; kept whether or not the log is)

`show(state)` is one line — `EventNetworkState{Float64}(3 actors, 2 events
absorbed, no memory decay, current_time = 2.0)` — naming the memory model
(`halflife h` for decay, `window w` for a window).

# Example
```julia
using REM
events = [Event(1, 2, 1.0), Event(2, 1, 2.0)]
seq = EventSequence(events; actors=ActorSet(1:3))
state = EventNetworkState(seq)                    # keep_history = true
for e in seq; update!(state, e); end
length(state.event_history)                       # 2
get_dyad_count(state, 1, 2)                       # 1.0
lean = EventNetworkState(seq; keep_history=false) # O(actors + dyads) memory

win = EventNetworkState(seq; window=1.5)          # events older than 1.5 expire
for e in seq; update!(win, e); end
win.current_time = 3.0                            # event 1 (t = 1.0) is now 2.0 old
get_dyad_count(win, 1, 2)                         # 0.0 — expired
get_dyad_count(win, 2, 1)                         # 1.0 — 1.0 old, still counts
```
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
    # The event log, (sender, receiver, time, weight) per absorbed event —
    # PUBLIC, read-only: it is what history-reading statistics (Relevent.jl's
    # PShift, Momentum, PriorInteraction, ...) consume. Kept only while
    # `keep_history` is set: none of REM's own statistics reads it (they read
    # the incremental counts, degrees, last-event times and adjacency), and
    # `generate_observations` / `compute_statistics` turn it off unless a
    # statistic declares `needs_history` — otherwise it would be O(events)
    # memory retained by every state for nothing.
    event_history::Vector{Tuple{Int,Int,T,Float64}}
    keep_history::Bool
    # Events absorbed since the last reset! — the log above may be off, and
    # `window_log` holds only the live events, so this is the one count of
    # what the state has seen (reported by `show`)
    n_events::Int
    # Incremental adjacency ("ever had an event"), so neighbor queries are
    # O(degree) instead of O(|event_history|). Note membership never
    # expires under decay: counts decay continuously, but structure counts
    # any past event (documented eventnet behavior). Under a WINDOW it does
    # expire: a dyad whose live events have all left the window is removed.
    out_neighbors::Dict{Int, Set{Int}}
    in_neighbors::Dict{Int, Set{Int}}
    # Sliding window (Inf = none). `window_log` is the FIFO of the events
    # still inside the window, `window_head` the index of its oldest live
    # entry (the prefix before it is expired and compacted away once it is
    # more than half the vector); the `live_*` dicts count the live events
    # per key so that a key whose events have all expired is DELETED (an
    # exact zero and no adjacency), not left at a floating-point residue.
    window::Float64
    window_log::Vector{Tuple{Int,Int,T,Float64}}
    window_head::Int
    live_dyad::Dict{Tuple{Int,Int}, Int}
    live_undirected::Dict{Tuple{Int,Int}, Int}
    live_out::Dict{Int, Int}
    live_in::Dict{Int, Int}

    function EventNetworkState{T}(; n_actors::Int=0, decay::Float64=0.0,
                                  keep_history::Bool=true,
                                  window::_WindowSpec=nothing) where T
        decay >= 0 || throw(ArgumentError("decay must be non-negative (got $decay)"))
        # A calendar period is a length of time on a calendar clock only: on
        # a numeric clock (event index, minutes, whatever the data count in)
        # converting `Day(2)` to 172800 clock units would silently be no
        # window at all
        (window isa Union{Dates.Period, Dates.CompoundPeriod} && !(T <: Dates.TimeType)) &&
            throw(ArgumentError(
                "window=$window is a calendar period but this sequence's clock is " *
                "$T (its own units, not seconds): pass a number in clock units, " *
                "or build the sequence on Date/DateTime timestamps"))
        w = _window_length(window)
        (decay > 0 && w < Inf) && throw(ArgumentError(
            "`decay` and `window` are two different memory models and cannot be " *
            "combined: pass either `decay` (eventnet's halflife decay — eventnet " *
            "offers no window) or `window` (events older than current_time − " *
            "window stop counting), not both (got decay = $decay, window = $window)"))
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
            keep_history,
            0,
            Dict{Int, Set{Int}}(),
            Dict{Int, Set{Int}}(),
            w,
            Tuple{Int,Int,T,Float64}[],
            1,
            Dict{Tuple{Int,Int}, Int}(),
            Dict{Tuple{Int,Int}, Int}(),
            Dict{Int, Int}(),
            Dict{Int, Int}()
        )
    end
end

# Normalize the `window` keyword: `nothing`/`Inf` mean no window; a finite
# window must be positive; a period is converted to seconds.
_window_length(::Nothing) = Inf
function _window_length(window::Real)
    w = Float64(window)
    (isnan(w) || w <= 0) && throw(ArgumentError(
        "window must be a positive length of time (or `nothing`/`Inf` for no " *
        "window); got $window"))
    return w
end
function _window_length(window::Union{Dates.Period, Dates.CompoundPeriod})
    w = _elapsed_seconds(window)
    w > 0 || throw(ArgumentError(
        "window must be a positive length of time; got $window " *
        "(= $w seconds — periods are converted to the seconds a Date/DateTime " *
        "clock counts in)"))
    return w
end

# A `Vector{Event}` is not an `EventSequence`: say so, and say what to do.
function EventNetworkState(events::AbstractVector{<:Event}; kwargs...)
    throw(ArgumentError(_event_vector_hint("EventNetworkState")))
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
    EventNetworkState(seq::EventSequence; decay=0.0, keep_history=true, window=nothing)

Create a EventNetworkState from an EventSequence without processing any events.
`keep_history=false` drops the per-event log; `window=` makes events older than
`current_time − window` stop counting (mutually exclusive with `decay > 0`; see
[`EventNetworkState`](@ref)).
"""
function EventNetworkState(seq::EventSequence{T}; decay::Float64=0.0,
                           keep_history::Bool=true,
                           window::_WindowSpec=nothing) where T
    state = EventNetworkState{T}(n_actors=seq.n_actors, decay=decay,
                                 keep_history=keep_history, window=window)
    state.actors = copy(seq.actors)
    return state
end

# One line, the memory model named: the Dict internals a default `show` would
# dump are the wrong thing to see in a REPL or a log
function Base.show(io::IO, state::EventNetworkState{T}) where T
    memory = state.window < Inf ? "window $(state.window)" :
             state.decay > 0 ? "halflife $(round(decay_to_halflife(state.decay); sigdigits=4)) (decay $(round(state.decay; sigdigits=4)))" :
             "no memory decay"
    n = state.n_events
    print(io, "EventNetworkState{", T, "}(", state.n_actors, " actor",
          state.n_actors == 1 ? "" : "s", ", ", n, " event", n == 1 ? "" : "s",
          " absorbed, ", memory, ", current_time = ", state.current_time, ")")
end

"""
    reset!(state::EventNetworkState) -> state

Reset the network state to empty: every count, degree, last-event time,
adjacency set, the event log and the window are cleared and the clock goes back
to `zero(T)`. The actor universe, `decay`, `window` and `keep_history` are kept,
so the same state can replay another sequence.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0)]; actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
get_dyad_count(state, 1, 2)      # 1.0
reset!(state)
get_dyad_count(state, 1, 2)      # 0.0
state.n_events, state.current_time   # (0, 0.0)
```
"""
function reset!(state::EventNetworkState{T}) where T
    state.n_events = 0
    empty!(state.dyad_counts)
    empty!(state.undirected_counts)
    empty!(state.out_degree)
    empty!(state.in_degree)
    empty!(state.last_event_time)
    empty!(state.event_history)
    empty!(state.out_neighbors)
    empty!(state.in_neighbors)
    empty!(state.window_log)
    state.window_head = 1
    empty!(state.live_dyad)
    empty!(state.live_undirected)
    empty!(state.live_out)
    empty!(state.live_in)
    state.current_time = zero(T)
    return state
end

"""
    update!(state::EventNetworkState, event::Event) -> state

Absorb `event` into the network state: advance the clock to `event.time`, add
`event.weight` to the sender→receiver dyad count, the undirected count and the
two degrees, record the dyad's last-event time, add the adjacency, append to
the event log (when `keep_history`) and to the window FIFO (when windowed).
O(1) per event — counts decay lazily on read, so nothing else is touched —
and **0 bytes** on a warmed state whose dyad has been seen before (the log
and the FIFO grow amortized when they are kept).

Statistics are meant to be read **before** the event is absorbed (that is what
[`generate_observations`](@ref) and [`compute_statistics`](@ref) do), so the
usual loop computes first and updates second.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
values = Float64[]
for e in seq
    push!(values, compute(Repetition(), state, e.sender, e.receiver))   # before
    update!(state, e)                                                    # after
end
values                           # [0.0, 0.0, 1.0]
get_dyad_count(state, 1, 2)      # 2.0
state.n_events                   # 3
```
"""
function update!(state::EventNetworkState{T}, event::Event{T}) where T
    s, r = event.sender, event.receiver
    t = event.time
    w = event.weight
    state.n_events += 1

    # Counts decay lazily on read; updating touches only the event's own
    # keys (O(1) per event instead of O(number of nonzero counts))
    state.current_time = t
    # Under a window: retire the events that have just left it
    _expire!(state)

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

    # The event log, only for the statistics that read it
    state.keep_history && push!(state.event_history, (s, r, t, w))

    # Update incremental adjacency. The lazy `get!(f, dict, key)` form: the
    # eager `get!(dict, key, Set{Int}())` evaluates (allocates) the empty set
    # on EVERY call, including the common one where the key exists — 80 B per
    # line, 160 B per event on a warmed state (pinned at 0 B by the
    # "update! is allocation-free" testset and benchmark/regression_tests.jl)
    push!(get!(() -> Set{Int}(), state.out_neighbors, s), r)
    push!(get!(() -> Set{Int}(), state.in_neighbors, r), s)

    # Under a window: remember the event so it can be retired later, and
    # count it as live on every key it touched
    if state.window < Inf
        push!(state.window_log, (s, r, t, w))
        state.live_dyad[dyad] = get(state.live_dyad, dyad, 0) + 1
        und = minmax(s, r)
        state.live_undirected[und] = get(state.live_undirected, und, 0) + 1
        state.live_out[s] = get(state.live_out, s, 0) + 1
        state.live_in[r] = get(state.live_in, r, 0) + 1
    end

    # Add actors if new
    push!(state.actors, s)
    push!(state.actors, r)
    state.n_actors = length(state.actors)

    return state
end

# ----------------------------------------------------------------------------
# Sliding window: expiry of the events that have left it
# ----------------------------------------------------------------------------
#
# The accessors below all call `_expire!` first, because `current_time` is set
# directly (`state.current_time = event.time`) by `generate_observations` and
# `compute_statistics` before the statistics of a stratum are read: the clock
# moves, and the events that fell out of the window must stop counting before
# the first read at the new time. The check is one Float64 compare when there
# is no window, and the cursor walk is amortized O(1) per absorbed event.

@inline function _expire!(state::EventNetworkState)
    state.window < Inf || return nothing
    return _expire_window!(state)
end

@noinline function _expire_window!(state::EventNetworkState{T}) where T
    log = state.window_log
    head = state.window_head
    n = length(log)
    now = state.current_time
    w = state.window
    @inbounds while head <= n
        s, r, t, wt = log[head]
        # An event exactly `window` old still counts; older ones expire. (An
        # event "in the future" of the clock — a hand-built state whose clock
        # was moved back — never expires.)
        (now > t && _elapsed_seconds(now - t) > w) || break
        _forget!(state, s, r, wt)
        head += 1
    end
    # Compact the expired prefix once it outweighs the live suffix, so the
    # FIFO holds O(live events) and the deletion cost is amortized O(1)
    if head > 1 && (head > n || 2 * (head - 1) > n)
        deleteat!(log, 1:(head - 1))
        head = 1
    end
    state.window_head = head
    return nothing
end

# Retire one event from every count it entered. A key whose live events have
# all expired is deleted outright (exact zero, no adjacency), otherwise the
# event's weight is subtracted.
function _forget!(state::EventNetworkState{T}, s::Int, r::Int, wt::Float64) where T
    dyad = (s, r)
    if _forget_count!(state.dyad_counts, state.live_dyad, dyad, wt)
        out_s = get(state.out_neighbors, s, nothing)
        out_s === nothing || delete!(out_s, r)
        in_r = get(state.in_neighbors, r, nothing)
        in_r === nothing || delete!(in_r, s)
    end
    _forget_count!(state.undirected_counts, state.live_undirected, minmax(s, r), wt)
    _forget_count!(state.out_degree, state.live_out, s, wt)
    _forget_count!(state.in_degree, state.live_in, r, wt)
    return nothing
end

# Returns `true` when the key's last live event was retired (key deleted)
function _forget_count!(counts::Dict{K, Tuple{Float64,T}}, live::Dict{K, Int},
                        key::K, wt::Float64) where {K, T}
    c = live[key] - 1
    if c == 0
        delete!(live, key)
        delete!(counts, key)
        return true
    end
    live[key] = c
    value, t_stored = counts[key]
    counts[key] = (value - wt, t_stored)
    return false
end


"""
    get_dyad_count(state::EventNetworkState, sender::Int, receiver::Int) -> Float64

The (decayed, event-weighted) count of events from `sender` to `receiver`;
`0.0` when there has been none, or when every one of them has left the window.
This is what [`Repetition`](@ref) reads (and [`Reciprocity`](@ref), with the
roles swapped).

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 0.0), Event(1, 2, 10.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
get_dyad_count(state, 1, 2)      # 2.0
get_dyad_count(state, 2, 1)      # 0.0

decayed = EventNetworkState(seq; decay=halflife_to_decay(10.0))
for e in seq; update!(decayed, e); end
get_dyad_count(decayed, 1, 2)    # 1.5 — the event at t = 0 is one halflife old
```
"""
function get_dyad_count(state::EventNetworkState, sender::Int, receiver::Int)
    _expire!(state)
    return _lazy_get(state, state.dyad_counts, (sender, receiver))
end

"""
    get_undirected_count(state::EventNetworkState, actor1::Int, actor2::Int) -> Float64

The (decayed, event-weighted) count of events between the two actors in
**either** direction — what `Repetition(directed=false)` and
[`CommonNeighbors`](@ref) read. Symmetric in its arguments.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)];
                    actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
get_undirected_count(state, 1, 2)    # 3.0
get_undirected_count(state, 2, 1)    # 3.0
get_dyad_count(state, 2, 1)          # 1.0 — one direction only
```
"""
function get_undirected_count(state::EventNetworkState, actor1::Int, actor2::Int)
    _expire!(state)
    return _lazy_get(state, state.undirected_counts, minmax(actor1, actor2))
end

"""
    get_out_degree(state::EventNetworkState, actor::Int) -> Float64

The (decayed, event-weighted) number of events `actor` has sent — what
[`SenderActivity`](@ref) reads. `0.0` for an actor that has sent nothing.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
get_out_degree(state, 1)    # 2.0
get_out_degree(state, 3)    # 0.0
```
"""
function get_out_degree(state::EventNetworkState, actor::Int)
    _expire!(state)
    return _lazy_get(state, state.out_degree, actor)
end

"""
    get_in_degree(state::EventNetworkState, actor::Int) -> Float64

The (decayed, event-weighted) number of events `actor` has received — what
[`ReceiverPopularity`](@ref) reads. `0.0` for an actor that has received nothing.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
get_in_degree(state, 1)     # 1.0
get_in_degree(state, 2)     # 1.0
get_in_degree(state, 3)     # 1.0
```
"""
function get_in_degree(state::EventNetworkState, actor::Int)
    _expire!(state)
    return _lazy_get(state, state.in_degree, actor)
end

"""
    _EmptyNeighbors <: AbstractSet{Int}

The **immutable, allocation-free** empty neighbour set returned by
[`get_out_neighbors`](@ref)/[`get_in_neighbors`](@ref) for an actor with no
neighbours (panel 2026-09, item 26). It is a singleton (`_EMPTY_NEIGHBORS`),
shared by every state and every actor; being immutable, it cannot be corrupted
by a caller who `push!`es into what the docstring says is read-only — the old
shared *mutable* `Set{Int}()` could be. `copy` gives a fresh `Set{Int}()`.
"""
struct _EmptyNeighbors <: AbstractSet{Int} end
const _EMPTY_NEIGHBORS = _EmptyNeighbors()
Base.length(::_EmptyNeighbors) = 0
Base.isempty(::_EmptyNeighbors) = true
Base.iterate(::_EmptyNeighbors, state=nothing) = nothing
Base.in(::Any, ::_EmptyNeighbors) = false
Base.copy(::_EmptyNeighbors) = Set{Int}()
Base.emptymutable(::_EmptyNeighbors, ::Type{U}=Int) where U = Set{U}()
Base.push!(::_EmptyNeighbors, x...) = throw(ArgumentError(
    "the neighbour sets returned by get_out_neighbors/get_in_neighbors are " *
    "read-only (this one is the shared empty set); `copy` it first"))
Base.delete!(::_EmptyNeighbors, x) = throw(ArgumentError(
    "the neighbour sets returned by get_out_neighbors/get_in_neighbors are " *
    "read-only (this one is the shared empty set); `copy` it first"))

# What the neighbour accessors return: the state's own set, or the singleton
const _NeighborSet = Union{Set{Int}, _EmptyNeighbors}

# ----------------------------------------------------------------------------
# Allocation-free intersections (panel 2026-09, item 26)
# ----------------------------------------------------------------------------
#
# The triadic statistics need |a ∩ b| and Σ_{k ∈ a ∩ b} f(k) with the focal
# sender and receiver excluded. `intersect(a, b)` + `delete!` materialised a
# Set per evaluation (320 B for TransitiveClosure, 1488 B for CommonNeighbors);
# these iterate the SMALLER set and test membership in the larger — O(min
# degree), zero bytes.

"""
    _count_common(a, b, x1, x2) -> Int

Number of elements of `a ∩ b` other than `x1` and `x2`, without allocating.
"""
@inline _count_common(::_EmptyNeighbors, ::_NeighborSet, ::Int, ::Int) = 0
@inline _count_common(::Set{Int}, ::_EmptyNeighbors, ::Int, ::Int) = 0
function _count_common(a::Set{Int}, b::Set{Int}, x1::Int, x2::Int)
    length(a) > length(b) && return _count_common(b, a, x1, x2)
    c = 0
    for k in a
        (k == x1 || k == x2) && continue
        k in b && (c += 1)
    end
    return c
end

"""
    _sum_common(f, a, b, x1, x2) -> Float64

`Σ f(k)` over the elements of `a ∩ b` other than `x1` and `x2`, without
allocating. `f` must be symmetric in the two sets' roles (it is called once per
common element, whichever set is iterated).
"""
@inline _sum_common(f::F, ::_EmptyNeighbors, ::_NeighborSet, ::Int, ::Int) where F = 0.0
@inline _sum_common(f::F, ::Set{Int}, ::_EmptyNeighbors, ::Int, ::Int) where F = 0.0
function _sum_common(f::F, a::Set{Int}, b::Set{Int}, x1::Int, x2::Int) where F
    length(a) > length(b) && return _sum_common(f, b, a, x1, x2)
    total = 0.0
    for k in a
        (k == x1 || k == x2) && continue
        k in b && (total += f(k))
    end
    return total
end

"""
    get_common_senders(state::EventNetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have sent events to both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets. (Allocates the
result; the statistics use the allocation-free `_count_common`/`_sum_common`.)
"""
function get_common_senders(state::EventNetworkState, actor1::Int, actor2::Int)
    return intersect(get_in_neighbors(state, actor1),
                     get_in_neighbors(state, actor2))
end

"""
    get_common_receivers(state::EventNetworkState, actor1::Int, actor2::Int) -> Set{Int}

Get the set of actors who have received events from both actor1 and actor2.
O(min degree) via the incrementally maintained adjacency sets. (Allocates the
result; the statistics use the allocation-free `_count_common`/`_sum_common`.)
"""
function get_common_receivers(state::EventNetworkState, actor1::Int, actor2::Int)
    return intersect(get_out_neighbors(state, actor1),
                     get_out_neighbors(state, actor2))
end

"""
    get_out_neighbors(state::EventNetworkState, actor::Int) -> AbstractSet{Int}

Get the set of actors to whom the given actor has sent events (under a
`window`, events still inside it).
Returns the internal set — treat as read-only. An actor with no neighbours
gets the shared immutable empty set (`copy` it to get a `Set{Int}`).

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
Set(get_out_neighbors(state, 1))    # Set([2, 3])
isempty(get_out_neighbors(state, 3))   # true — read-only shared empty set
mine = copy(get_out_neighbors(state, 3)); push!(mine, 9)   # a Set{Int} of your own
```
"""
@inline function get_out_neighbors(state::EventNetworkState, actor::Int)::_NeighborSet
    _expire!(state)
    return get(state.out_neighbors, actor, _EMPTY_NEIGHBORS)
end

"""
    get_in_neighbors(state::EventNetworkState, actor::Int) -> AbstractSet{Int}

Get the set of actors who have sent events to the given actor (under a
`window`, events still inside it).
Returns the internal set — treat as read-only. An actor with no neighbours
gets the shared immutable empty set (`copy` it to get a `Set{Int}`).

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
Set(get_in_neighbors(state, 1))    # Set([2])
Set(get_in_neighbors(state, 3))    # Set([1])
```
"""
@inline function get_in_neighbors(state::EventNetworkState, actor::Int)::_NeighborSet
    _expire!(state)
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

Adjacency never expires under `decay` (a count fades, the edge stays); under
a `window` it expires with the dyad's last live event.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 3, 2.0)]; actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
has_edge(state, 1, 2)    # true
has_edge(state, 2, 1)    # false — direction matters
has_edge(state, 1, 3)    # false
```
"""
function has_edge(state::EventNetworkState, sender::Int, receiver::Int)
    return get_dyad_count(state, sender, receiver) > 0
end
