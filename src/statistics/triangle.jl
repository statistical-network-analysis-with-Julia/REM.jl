"""
Triangle (triadic closure) statistics for REM.

These statistics capture effects based on triadic structures in the network,
measuring various forms of closure and transitivity.

# eventnet's definition (the default since 0.2.0)

For a candidate event `s → r`, eventnet evaluates every two-path through a
third actor `k ∉ {s, r}`, combines the (halflife-decayed, event-weighted)
counts of its two dyads with an **aggregation function** — `min` by default,
or `max`, `sum`, `product` — and *adds the values of the parallel two-paths*:

    Σ_{k ≠ s,r}  aggregation(w(first dyad of the two-path), w(second dyad))

`weighted = true, aggregation = :min` is that definition, and the default.
`weighted = false` is the pre-0.2 definition: the **number of distinct third
parties** `k` closing the pattern, read off the adjacency ("ever had an
event"), which under decay never fades and under a window expires with the
last event of a dyad.
"""

# The combining functions eventnet offers for the dyad weights of a two-path
# (and of the three-path of a four-cycle)
const _AGGREGATIONS = (:min, :max, :sum, :product)

function _check_aggregation(aggregation::Symbol)
    aggregation in _AGGREGATIONS || throw(ArgumentError(
        "aggregation must be one of :min, :max, :sum or :product (eventnet's " *
        "combining functions for the weights of a two-path), got :$aggregation"))
    return aggregation
end

@inline function _aggregate(aggregation::Symbol, a::Float64, b::Float64)
    aggregation === :min && return min(a, b)
    aggregation === :max && return max(a, b)
    aggregation === :sum && return a + b
    return a * b
end

@inline function _aggregate(aggregation::Symbol, a::Float64, b::Float64, c::Float64)
    aggregation === :min && return min(a, b, c)
    aggregation === :max && return max(a, b, c)
    aggregation === :sum && return a + b + c
    return a * b * c
end

# One docstring paragraph shared by the four directed closure statistics
const _TRIAD_FIELDS_DOC = """
# Fields
- `weighted::Bool` (default `true`): eventnet's weighted form — sum over the
  closing third parties `k` of `aggregation` applied to the two dyad weights
  of the two-path (decayed counts, event weights included). `false` counts the
  distinct third parties instead (adjacency only; the pre-0.2 definition).
- `aggregation::Symbol` (default `:min`): `:min`, `:max`, `:sum` or `:product`.
  Ignored when `weighted = false`.
- `stat_name::String`: Name for this statistic.
"""

"""
    TransitiveClosure <: TriangleStatistic

Transitive closure: tendency for `s → r` when there is a `k` with `s → k → r`.

    TransitiveClosure(; weighted=true, aggregation=:min, name="transitive_closure")

eventnet's definition (the default): `Σ_{k ≠ s,r} aggregation(w(s,k), w(k,r))`,
with `w` the decayed, event-weighted dyad counts. `weighted=false` returns the
number of distinct `k` with `s → k` and `k → r` ever observed.

$_TRIAD_FIELDS_DOC
# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(TransitiveClosure(), state, 1, 3)                       # min(2, 1) = 1.0
compute(TransitiveClosure(aggregation=:sum), state, 1, 3)       # 2 + 1 = 3.0
compute(TransitiveClosure(weighted=false), state, 1, 3)         # one third party: 1.0
```
"""
struct TransitiveClosure <: TriangleStatistic
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function TransitiveClosure(; weighted::Bool=true, aggregation::Symbol=:min,
                               name::String="transitive_closure")
        new(weighted, _check_aggregation(aggregation), name)
    end
end

function compute(stat::TransitiveClosure, state::EventNetworkState, sender::Int, receiver::Int)
    # k with s→k and k→r: out-neighbors of the sender who are in-neighbors of
    # the receiver
    out_s = get_out_neighbors(state, sender)
    in_r = get_in_neighbors(state, receiver)
    stat.weighted || return Float64(_count_common(out_s, in_r, sender, receiver))
    aggregation = stat.aggregation
    return _sum_common(out_s, in_r, sender, receiver) do k
        _aggregate(aggregation, get_dyad_count(state, sender, k),
                   get_dyad_count(state, k, receiver))
    end
end

name(stat::TransitiveClosure) = stat.stat_name

"""
    CyclicClosure <: TriangleStatistic

Cyclic closure: tendency for `s → r` when there is a `k` with `r → k → s`.

    CyclicClosure(; weighted=true, aggregation=:min, name="cyclic_closure")

eventnet's definition (the default): `Σ_{k ≠ s,r} aggregation(w(r,k), w(k,s))`.
`weighted=false` returns the number of distinct `k` with `r → k` and `k → s`.

$_TRIAD_FIELDS_DOC
# Example
```julia
using REM
seq = EventSequence([Event(3, 2, 1.0), Event(2, 1, 2.0), Event(2, 1, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(CyclicClosure(), state, 1, 3)                 # 3→2→1 closes 1→3: min(1, 2) = 1.0
compute(CyclicClosure(aggregation=:max), state, 1, 3) # 2.0
```
"""
struct CyclicClosure <: TriangleStatistic
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function CyclicClosure(; weighted::Bool=true, aggregation::Symbol=:min,
                           name::String="cyclic_closure")
        new(weighted, _check_aggregation(aggregation), name)
    end
end

function compute(stat::CyclicClosure, state::EventNetworkState, sender::Int, receiver::Int)
    # k with r→k and k→s
    out_r = get_out_neighbors(state, receiver)
    in_s = get_in_neighbors(state, sender)
    stat.weighted || return Float64(_count_common(out_r, in_s, sender, receiver))
    aggregation = stat.aggregation
    return _sum_common(out_r, in_s, sender, receiver) do k
        _aggregate(aggregation, get_dyad_count(state, receiver, k),
                   get_dyad_count(state, k, sender))
    end
end

name(stat::CyclicClosure) = stat.stat_name

"""
    SharedSender <: TriangleStatistic

Shared sender: tendency for `s → r` when there is a `k` with `k → s` and `k → r`
(an in-star; eventnet's "shared sender" / two-in-star statistic).

    SharedSender(; weighted=true, aggregation=:min, name="shared_sender")

eventnet's definition (the default): `Σ_{k ≠ s,r} aggregation(w(k,s), w(k,r))`.
`weighted=false` returns the number of distinct `k` who sent to both.

$_TRIAD_FIELDS_DOC
# Example
```julia
using REM
seq = EventSequence([Event(3, 1, 1.0), Event(3, 2, 2.0), Event(3, 2, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(SharedSender(), state, 1, 2)                     # min(1, 2) = 1.0
compute(SharedSender(aggregation=:product), state, 1, 2) # 2.0
```
"""
struct SharedSender <: TriangleStatistic
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function SharedSender(; weighted::Bool=true, aggregation::Symbol=:min,
                          name::String="shared_sender")
        new(weighted, _check_aggregation(aggregation), name)
    end
end

function compute(stat::SharedSender, state::EventNetworkState, sender::Int, receiver::Int)
    # k with k→s and k→r
    in_s = get_in_neighbors(state, sender)
    in_r = get_in_neighbors(state, receiver)
    stat.weighted || return Float64(_count_common(in_s, in_r, sender, receiver))
    aggregation = stat.aggregation
    return _sum_common(in_s, in_r, sender, receiver) do k
        _aggregate(aggregation, get_dyad_count(state, k, sender),
                   get_dyad_count(state, k, receiver))
    end
end

name(stat::SharedSender) = stat.stat_name

"""
    SharedReceiver <: TriangleStatistic

Shared receiver: tendency for `s → r` when there is a `k` with `s → k` and
`r → k` (an out-star; eventnet's "shared receiver" / two-out-star statistic).

    SharedReceiver(; weighted=true, aggregation=:min, name="shared_receiver")

eventnet's definition (the default): `Σ_{k ≠ s,r} aggregation(w(s,k), w(r,k))`.
`weighted=false` returns the number of distinct `k` who received from both.

$_TRIAD_FIELDS_DOC
# Example
```julia
using REM
seq = EventSequence([Event(1, 3, 1.0), Event(2, 3, 2.0), Event(2, 3, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(SharedReceiver(), state, 1, 2)                   # min(1, 2) = 1.0
compute(SharedReceiver(weighted=false), state, 1, 2)     # 1.0 (one shared receiver)
```
"""
struct SharedReceiver <: TriangleStatistic
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function SharedReceiver(; weighted::Bool=true, aggregation::Symbol=:min,
                            name::String="shared_receiver")
        new(weighted, _check_aggregation(aggregation), name)
    end
end

function compute(stat::SharedReceiver, state::EventNetworkState, sender::Int, receiver::Int)
    # k with s→k and r→k
    out_s = get_out_neighbors(state, sender)
    out_r = get_out_neighbors(state, receiver)
    stat.weighted || return Float64(_count_common(out_s, out_r, sender, receiver))
    aggregation = stat.aggregation
    return _sum_common(out_s, out_r, sender, receiver) do k
        _aggregate(aggregation, get_dyad_count(state, sender, k),
                   get_dyad_count(state, receiver, k))
    end
end

name(stat::SharedReceiver) = stat.stat_name

"""
    CommonNeighbors <: TriangleStatistic

Common neighbours regardless of direction: the undirected (SYM) form of the
closure statistics — `k ∉ {s, r}` is a common neighbour when it has had an event
with `s` in either direction and one with `r` in either direction.

    CommonNeighbors(; weighted=true, aggregation=:min, name="common_neighbors")

eventnet's weighted form (the default): `Σ_k aggregation(u(s,k), u(k,r))` with
`u` the **undirected** decayed, event-weighted counts (events in either
direction added together — what [`get_undirected_count`](@ref) returns).
`weighted=false` returns the number of distinct common neighbours.

# Fields
- `weighted::Bool` (default `true`): the weighted form above; `false` counts
  distinct common neighbours (adjacency only).
- `aggregation::Symbol` (default `:min`): `:min`, `:max`, `:sum` or `:product`.
- `stat_name::String`: Name for this statistic.

# Example
```julia
using REM
seq = EventSequence([Event(1, 3, 1.0), Event(3, 1, 2.0), Event(3, 2, 3.0)];
                    actors=ActorSet(1:3))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(CommonNeighbors(), state, 1, 2)                 # min(u(1,3) = 2, u(3,2) = 1) = 1.0
compute(CommonNeighbors(aggregation=:sum), state, 1, 2) # 3.0
compute(CommonNeighbors(weighted=false), state, 1, 2)   # 1.0
```
"""
struct CommonNeighbors <: TriangleStatistic
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function CommonNeighbors(; weighted::Bool=true, aggregation::Symbol=:min,
                             name::String="common_neighbors")
        new(weighted, _check_aggregation(aggregation), name)
    end
end

function compute(stat::CommonNeighbors, state::EventNetworkState, sender::Int, receiver::Int)
    out_s = get_out_neighbors(state, sender)
    in_s = get_in_neighbors(state, sender)
    out_r = get_out_neighbors(state, receiver)
    in_r = get_in_neighbors(state, receiver)
    # The two loops are written out here (not delegated to a helper taking the
    # four neighbour sets) so that each `in` test is dispatched on ONE small
    # union — four union-typed arguments in one signature exceed Julia's
    # union-splitting limit and fall back to dynamic dispatch, which allocates
    total = 0.0
    for k in out_s
        (k == sender || k == receiver) && continue
        (k in out_r || k in in_r) && (total += _common_neighbor_value(stat, state, sender, receiver, k))
    end
    for k in in_s
        (k == sender || k == receiver) && continue
        k in out_s && continue                   # already counted above
        (k in out_r || k in in_r) && (total += _common_neighbor_value(stat, state, sender, receiver, k))
    end
    return total
end

# One common neighbour's contribution: 1 for the count, eventnet's aggregation
# of the two undirected counts for the weighted form
@inline function _common_neighbor_value(stat::CommonNeighbors, state::EventNetworkState,
                                        sender::Int, receiver::Int, k::Int)
    stat.weighted || return 1.0
    return _aggregate(stat.aggregation, get_undirected_count(state, sender, k),
                      get_undirected_count(state, k, receiver))
end

name(stat::CommonNeighbors) = stat.stat_name

"""
    GeometricWeightedTriads <: TriangleStatistic

Geometrically weighted shared-partner statistic (the relational-event analogue
of ERGM's GWESP): with `n` the **number of distinct** third parties closing the
chosen pattern,

    exp(α) · (1 − (1 − exp(−α))^n)

so each additional shared partner adds less than the one before (`0` for
`n = 0`, `1` for `n = 1`, `1 + (1 − e^{−α})` for `n = 2`, ...). `n` is the count
of third parties — the `weighted=false` value of the corresponding closure
statistic — not a weighted sum; use the closure statistics with
`aggregation=` for eventnet's weighted forms.

# Fields
- `closure_type::Symbol`: `:transitive` (`s→k→r`), `:cyclic` (`r→k→s`),
  `:shared_sender` (`k→s, k→r`) or `:shared_receiver` (`s→k, r→k`).
- `alpha::Float64`: Decay parameter (higher = less down-weighting).
- `stat_name::String`: Name for this statistic.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 4, 2.0), Event(1, 3, 3.0), Event(3, 4, 4.0)];
                    actors=ActorSet(1:4))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
gw = GeometricWeightedTriads(closure_type=:transitive, alpha=0.5)
compute(gw, state, 1, 4)     # two two-paths (via 2 and 3): e^0.5 (1 − (1 − e^-0.5)^2) ≈ 1.393
```
"""
struct GeometricWeightedTriads <: TriangleStatistic
    closure_type::Symbol
    alpha::Float64
    stat_name::String

    function GeometricWeightedTriads(; closure_type::Symbol=:transitive, alpha::Float64=0.5,
                                      name::String="")
        stat_name = isempty(name) ? "gw_$(closure_type)" : name
        closure_type in (:transitive, :cyclic, :shared_sender, :shared_receiver) ||
            throw(ArgumentError("closure_type must be :transitive, :cyclic, :shared_sender, or :shared_receiver"))
        alpha > 0 || throw(ArgumentError("alpha must be positive"))
        new(closure_type, alpha, stat_name)
    end
end

function compute(stat::GeometricWeightedTriads, state::EventNetworkState, sender::Int, receiver::Int)
    # Number of distinct third parties for the chosen closure pattern
    n = if stat.closure_type == :transitive
        _count_common(get_out_neighbors(state, sender), get_in_neighbors(state, receiver),
                      sender, receiver)
    elseif stat.closure_type == :cyclic
        _count_common(get_out_neighbors(state, receiver), get_in_neighbors(state, sender),
                      sender, receiver)
    elseif stat.closure_type == :shared_sender
        _count_common(get_in_neighbors(state, sender), get_in_neighbors(state, receiver),
                      sender, receiver)
    else  # :shared_receiver
        _count_common(get_out_neighbors(state, sender), get_out_neighbors(state, receiver),
                      sender, receiver)
    end
    n == 0 && return 0.0
    # Geometrically weighted sum: exp(alpha) * (1 - (1 - exp(-alpha))^n)
    return exp(stat.alpha) * (1 - (1 - exp(-stat.alpha))^n)
end

name(stat::GeometricWeightedTriads) = stat.stat_name
