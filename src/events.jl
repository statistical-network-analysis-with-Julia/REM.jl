"""
Event loading and parsing utilities.
"""

"""
    load_events(filepath::String; kwargs...) -> EventSequence

Load events from a CSV file.

# Arguments
- `filepath`: Path to the CSV file

# Keyword Arguments
- `sender_col::Symbol=:sender`: Column name for sender IDs
- `receiver_col::Symbol=:receiver`: Column name for receiver IDs
- `time_col::Symbol=:time`: Column name for timestamps
- `type_col::Union{Symbol,Nothing}=nothing`: Column name for event types
- `weight_col::Union{Symbol,Nothing}=nothing`: Column name for event weights
- `time_type::Type=Float64`: Type to parse timestamps as
- `actor_names::Bool=false`: If true, treat sender/receiver as names and assign numeric IDs

# Returns
- `EventSequence`: Sequence of loaded events. The actor universe is **inferred**
  from the event endpoints (`actors_declared == false`): declare it before
  fitting — `EventSequence(collect(seq); actors=ActorSet(ids))` — or pass
  `at_risk=` to `fit_rem`, which otherwise warns.

# Example
```julia
using REM, CSV, DataFrames
path = joinpath(mktempdir(), "events.csv")
CSV.write(path, DataFrame(sender=[1, 2, 1], receiver=[2, 1, 3], time=[1.0, 2.0, 3.0]))
seq = load_events(path)
length(seq), seq.n_actors            # (3, 3)
named = joinpath(mktempdir(), "named.csv")
CSV.write(named, DataFrame(from=["Ann", "Bob"], to=["Bob", "Ann"], t=[1.0, 2.0]))
load_events(named; sender_col=:from, receiver_col=:to, time_col=:t, actor_names=true)
```
"""
function load_events(filepath::String;
                     sender_col::Symbol=:sender,
                     receiver_col::Symbol=:receiver,
                     time_col::Symbol=:time,
                     type_col::Union{Symbol,Nothing}=nothing,
                     weight_col::Union{Symbol,Nothing}=nothing,
                     time_type::Type{T}=Float64,
                     actor_names::Bool=false) where T
    df = CSV.read(filepath, DataFrame)
    load_events(df; sender_col, receiver_col, time_col, type_col, weight_col,
                time_type, actor_names)
end

"""
    load_events(df::DataFrame; kwargs...) -> EventSequence

Load events from a DataFrame (same keywords as the file method: `sender_col`,
`receiver_col`, `time_col`, `type_col`, `weight_col`, `time_type`,
`actor_names`). String timestamps are parsed as `time_type` (`Float64`, `Int`,
`DateTime` or `Date`).

# Example
```julia
using REM, DataFrames, Dates
df = DataFrame(sender=[1, 2, 1], receiver=[2, 1, 3], time=[1.0, 2.0, 3.0],
               kind=[:email, :email, :call], w=[1.0, 2.0, 1.0])
seq = load_events(df; type_col=:kind, weight_col=:w)
seq[2].weight, seq.eventtypes          # (2.0, Set([:email, :call]))
stamped = DataFrame(sender=[1, 2], receiver=[2, 1],
                    time=["2024-01-01T10:00:00", "2024-01-01T11:00:00"])
load_events(stamped; time_type=DateTime)[1].time    # DateTime("2024-01-01T10:00:00")
```
"""
function load_events(df::DataFrame;
                     sender_col::Symbol=:sender,
                     receiver_col::Symbol=:receiver,
                     time_col::Symbol=:time,
                     type_col::Union{Symbol,Nothing}=nothing,
                     weight_col::Union{Symbol,Nothing}=nothing,
                     time_type::Type{T}=Float64,
                     actor_names::Bool=false) where T
    # Build actor name mapping if needed
    name_to_id = Dict{Any, Int}()
    if actor_names
        all_actors = unique(vcat(df[!, sender_col], df[!, receiver_col]))
        for (i, name) in enumerate(all_actors)
            name_to_id[name] = i
        end
    end

    events = Event{T}[]
    sizehint!(events, nrow(df))

    for row in eachrow(df)
        # Get sender and receiver IDs
        if actor_names
            sender = name_to_id[row[sender_col]]
            receiver = name_to_id[row[receiver_col]]
        else
            sender = Int(row[sender_col])
            receiver = Int(row[receiver_col])
        end

        # Parse timestamp
        time_val = parse_time(row[time_col], T)

        # Get optional fields
        eventtype = isnothing(type_col) ? :event : Symbol(row[type_col])
        weight = isnothing(weight_col) ? 1.0 : Float64(row[weight_col])

        push!(events, Event(sender, receiver, time_val; eventtype, weight))
    end

    return EventSequence(events)
end

"""
    load_events!(seq::EventSequence, filepath::String; kwargs...)

Load events from a CSV file and add them to an existing EventSequence (inserted
in time order). Same keywords as [`load_events`](@ref) except `time_type`, which
is the sequence's clock.

# Example
```julia
using REM, CSV, DataFrames
path = joinpath(mktempdir(), "more.csv")
CSV.write(path, DataFrame(sender=[3, 1], receiver=[1, 3], time=[2.5, 4.0]))
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 3.0)]; actors=ActorSet(1:3))
load_events!(seq, path)
[e.time for e in seq]           # [1.0, 2.5, 3.0, 4.0]
```
"""
function load_events!(seq::EventSequence{T}, filepath::String;
                      sender_col::Symbol=:sender,
                      receiver_col::Symbol=:receiver,
                      time_col::Symbol=:time,
                      type_col::Union{Symbol,Nothing}=nothing,
                      weight_col::Union{Symbol,Nothing}=nothing,
                      actor_names::Bool=false) where T
    df = CSV.read(filepath, DataFrame)
    load_events!(seq, df; sender_col, receiver_col, time_col, type_col,
                 weight_col, actor_names)
end

"""
    load_events!(seq::EventSequence, df::DataFrame; kwargs...)

Load events from a DataFrame and add them to an existing EventSequence
(inserted in time order). With `actor_names=true`, pass the same `name_to_id`
dictionary across calls to keep the name → ID assignment stable.

# Example
```julia
using REM, DataFrames
seq = EventSequence{Float64}(; actors=ActorSet(1:3))
load_events!(seq, DataFrame(sender=[1, 2], receiver=[2, 1], time=[1.0, 2.0]))
load_events!(seq, DataFrame(sender=[3], receiver=[1], time=[1.5]))
[(e.sender, e.receiver) for e in seq]     # [(1, 2), (3, 1), (2, 1)]
```
"""
function load_events!(seq::EventSequence{T}, df::DataFrame;
                      sender_col::Symbol=:sender,
                      receiver_col::Symbol=:receiver,
                      time_col::Symbol=:time,
                      type_col::Union{Symbol,Nothing}=nothing,
                      weight_col::Union{Symbol,Nothing}=nothing,
                      actor_names::Bool=false,
                      name_to_id::Dict{Any,Int}=Dict{Any,Int}()) where T
    for row in eachrow(df)
        if actor_names
            sender = get!(name_to_id, row[sender_col], length(name_to_id) + 1)
            receiver = get!(name_to_id, row[receiver_col], length(name_to_id) + 1)
        else
            sender = Int(row[sender_col])
            receiver = Int(row[receiver_col])
        end

        time_val = parse_time(row[time_col], T)
        eventtype = isnothing(type_col) ? :event : Symbol(row[type_col])
        weight = isnothing(weight_col) ? 1.0 : Float64(row[weight_col])

        push!(seq, Event(sender, receiver, time_val; eventtype, weight))
    end

    return seq
end

"""
    parse_time(val, ::Type{T}) where T

Parse a time value to the specified type.
"""
parse_time(val::T, ::Type{T}) where T = val
parse_time(val::Number, ::Type{T}) where T<:Number = T(val)
parse_time(val::AbstractString, ::Type{Float64}) = parse(Float64, val)
parse_time(val::AbstractString, ::Type{Int}) = parse(Int, val)
parse_time(val::AbstractString, ::Type{DateTime}) = DateTime(val)
parse_time(val::AbstractString, ::Type{Date}) = Date(val)

# Handle Unix timestamps
function parse_time(val::Number, ::Type{DateTime})
    # Assume Unix timestamp in seconds
    DateTime(Dates.unix2datetime(val))
end

"""
    halflife_to_decay(halflife::Real) -> Float64
    halflife_to_decay(halflife::Dates.Period) -> Float64

Convert a halflife to an exponential decay rate: `λ = log(2) / halflife`, so
that an event's weight `exp(−λ·Δ)` is `0.5` when it is `halflife` old. This is
eventnet's halflife parameterisation; pass the result as `decay=` to
[`fit_rem`](@ref), [`generate_observations`](@ref) or [`EventNetworkState`](@ref).
The halflife is in clock units — **seconds** for `Date`/`DateTime` clocks, so
on a calendar clock pass a `Dates.Period` (`halflife_to_decay(Day(7))`, which
is `halflife_to_decay(7 * 86400)`) rather than counting seconds by hand.

# Example
```julia
using REM, Dates
λ = halflife_to_decay(10.0)          # 0.0693…
compute_decay_weight(10.0, λ)        # 0.5
decay_to_halflife(λ)                 # 10.0
halflife_to_decay(Day(7)) == halflife_to_decay(7 * 86400)   # true — seconds
```
"""
function halflife_to_decay(halflife::Real)
    halflife > 0 || throw(ArgumentError("halflife must be positive"))
    return log(2) / halflife
end
function halflife_to_decay(halflife::Union{Dates.Period, Dates.CompoundPeriod})
    s = _elapsed_seconds(halflife)
    s > 0 || throw(ArgumentError(
        "halflife must be positive; got $halflife (= $s seconds — periods are " *
        "converted to the seconds a Date/DateTime clock counts in)"))
    return log(2) / s
end

"""
    decay_to_halflife(decay::Real) -> Float64

Convert an exponential decay rate back to a halflife: `log(2) / decay`, the
inverse of [`halflife_to_decay`](@ref).

# Example
```julia
using REM
decay_to_halflife(log(2) / 24)      # 24.0
decay_to_halflife(halflife_to_decay(7.0))   # 7.0
```
"""
function decay_to_halflife(decay::Real)
    decay > 0 || throw(ArgumentError("decay rate must be positive"))
    return log(2) / decay
end

"""
    compute_decay_weight(elapsed_time::Real, decay::Real) -> Float64

The weight `exp(−decay · elapsed_time)` an event carries `elapsed_time` after
it happened — what every decayed count in an [`EventNetworkState`](@ref)
applies (lazily, on read). Note the argument order: elapsed time first, decay
rate second. `elapsed_time` must be non-negative.

# Example
```julia
using REM
λ = halflife_to_decay(10.0)
compute_decay_weight(0.0, λ)         # 1.0 — a fresh event
compute_decay_weight(10.0, λ)        # 0.5 — one halflife old
compute_decay_weight(20.0, λ)        # 0.25
```
"""
function compute_decay_weight(elapsed_time::Real, decay::Real)
    elapsed_time >= 0 || throw(ArgumentError("elapsed_time must be non-negative"))
    return exp(-decay * elapsed_time)
end
