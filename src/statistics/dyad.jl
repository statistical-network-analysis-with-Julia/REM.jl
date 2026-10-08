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

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(Repetition(), state, 1, 2)                  # 2.0
compute(Repetition(directed=false), state, 1, 2)    # 3.0
```
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

# Example
```julia
using REM
seq = EventSequence([Event(2, 1, 1.0), Event(2, 1, 2.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(Reciprocity(), state, 1, 2)    # 2.0 — two past 2→1 events
```
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

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 1, 3.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(InertiaStatistic(), state, 1, 2)                          # 2 + 1 = 3.0
compute(InertiaStatistic(reciprocity_weight=0.5), state, 1, 2)    # 2 + 0.5 = 2.5
```
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

Recency: how recently the last event occurred on the focal dyad, as a
**decreasing** transform of the elapsed time `Δ = current_time − t_last`;
`0.0` when the dyad has no prior event. Elapsed time is in the clock's units
for numeric time and in **seconds** for `Date`/`DateTime` time.

At `Δ = 0` — an event tied with the last one on the dyad (`ties=:ordered`
lets a dyad act twice at one timestamp), or a hand-built state read at its
last event's time — the transforms differ by design: `:exp_decay` is
well-defined there and returns `exp(0) = 1`, "just happened" (not `0`,
"never happened"); `:inverse` and `:inverse_log` are singular at `0` and
return `0.0`, a deliberate cap: an event tied with the last one on the dyad
is treated as having no usable recency. Pass `transform=:exp_decay` if tied
same-dyad events are a feature of your data.

    RecencyStatistic(; directed=true, transform=:inverse, decay=1.0, name="recency_<transform>")

# Fields
- `directed::Bool`: If true, the last `s → r` event; if false, the last event
  in either direction between `s` and `r` (the later of the two).
- `transform::Symbol`: the transform of `Δ`:
    - `:inverse` — `1 / Δ`
    - `:inverse_log` — `1 / log(1 + Δ)` (called `:log` before 0.2.0, which
      misdescribed it: it is not a log transform, and `:log` is now refused
      with a pointer here)
    - `:exp_decay` — `exp(−decay · Δ)`
- `decay::Float64`: Decay parameter for the `:exp_decay` transform.
- `stat_name::String`: Name for this statistic.

Unlike the state-level `decay`/`window` (which weight every *count*), this is
a statistic of its own: it reads the dyad's last-event time, which no window
expires.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 3.0)]; actors=ActorSet(1:2))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
state.current_time = 5.0
compute(RecencyStatistic(), state, 1, 2)                           # 1 / 4 = 0.25
compute(RecencyStatistic(directed=false), state, 1, 2)             # last in either direction: 1 / 2
compute(RecencyStatistic(transform=:inverse_log), state, 1, 2)     # 1 / log(5)
compute(RecencyStatistic(transform=:exp_decay, decay=0.5), state, 1, 2)  # exp(-2)
```
"""
struct RecencyStatistic <: DyadStatistic
    directed::Bool
    transform::Symbol
    decay::Float64
    stat_name::String

    function RecencyStatistic(; directed::Bool=true, transform::Symbol=:inverse,
                               decay::Float64=1.0, name::String="")
        transform === :log && throw(ArgumentError(
            "RecencyStatistic: transform=:log was `1 / log(1 + elapsed)`, which is " *
            "not a log transform; it is now called `:inverse_log` (the same " *
            "formula). Pass transform=:inverse_log, :inverse or :exp_decay."))
        transform in (:inverse, :inverse_log, :exp_decay) || throw(ArgumentError(
            "RecencyStatistic: transform must be :inverse (1/Δ), :inverse_log " *
            "(1/log(1+Δ)) or :exp_decay (exp(−decay·Δ)); got :$transform"))
        decay > 0 || throw(ArgumentError("RecencyStatistic: decay must be positive (got $decay)"))
        stat_name = isempty(name) ? "recency_$(transform)" : name
        new(directed, transform, decay, stat_name)
    end
end

function compute(stat::RecencyStatistic, state::EventNetworkState{T}, sender::Int, receiver::Int) where T
    lt = state.last_event_time
    t_fwd = get(lt, (sender, receiver), nothing)
    if stat.directed
        t_fwd === nothing && return 0.0
        last_time = t_fwd
    else
        # The last event in EITHER direction: the later of the two dyads'
        # last-event times (before 0.2.0 this looked up the (min, max) dyad
        # only, i.e. one direction)
        t_rev = get(lt, (receiver, sender), nothing)
        if t_fwd === nothing
            t_rev === nothing && return 0.0
            last_time = t_rev
        elseif t_rev === nothing
            last_time = t_fwd
        else
            last_time = max(t_fwd, t_rev)
        end
    end

    elapsed = _elapsed_seconds(state.current_time - last_time)
    # A prior event exists: `exp_decay` is defined at Δ = 0 (and gives 1,
    # "just happened"); the two singular transforms are capped at 0.0 there
    # (a documented convention, not a value of the transform)
    if stat.transform === :exp_decay
        return elapsed < 0 ? 0.0 : exp(-stat.decay * elapsed)
    end
    elapsed <= 0 && return 0.0
    if stat.transform === :inverse
        return 1.0 / elapsed
    else  # :inverse_log
        return 1.0 / log1p(elapsed)
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

# Example
```julia
using REM
dist = DyadCovariate(Dict((1, 2) => 10.0, (2, 1) => 12.0); default=100.0, name="distance")
state = EventNetworkState{Float64}()
compute(dist, state, 1, 2)    # 10.0
compute(dist, state, 3, 1)    # 100.0 (the default)
```
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
