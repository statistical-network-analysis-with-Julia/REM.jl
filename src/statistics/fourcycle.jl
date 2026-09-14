"""
Four-cycle statistics for REM.

These statistics capture local clustering effects based on four-node structures,
measuring the closure of three-paths through two intermediate nodes. As for the
triadic statistics, the default is eventnet's weighted form: the three dyad
weights of each three-path are combined with `aggregation` (`:min` by default)
and the values of the parallel three-paths are added.
"""

"""
    FourCycle <: FourCycleStatistic

Measures four-cycle closure: tendency for s→r when there exist j, k such that
s→j, k→j, and k→r (or variants).

    FourCycle(; cycle_type=:out_out, weighted=true, aggregation=:min, name="")

This captures local clustering where sender and receiver share connections
to a common pair of intermediaries. Both intermediaries are distinct from each
other and from `s` and `r`.

# Fields
- `cycle_type::Symbol`: Type of four-cycle configuration.
    - `:out_out`: s→j←k→r (shared out-neighbor pattern)
    - `:in_in`: s←j→k←r (shared in-neighbor pattern)
    - `:out_in`: s→j→k→r (two-path through intermediaries)
    - `:in_out`: s←j←k←r (reverse chain)
    - `:mixed`: the sum of the four
- `weighted::Bool` (default `true`): sum over the closing three-paths `(j, k)`
  of `aggregation` applied to the three dyad weights (decayed counts, event
  weights included). `false` counts the distinct three-paths instead
  (adjacency only; the pre-0.2 definition).
- `aggregation::Symbol` (default `:min`): `:min`, `:max`, `:sum` or `:product`
  over the three weights. Ignored when `weighted = false`.
- `stat_name::String`: Name for this statistic.

# Example
```julia
using REM
# s = 1 → j = 2 ← k = 3 → r = 4, with 3→2 observed twice
seq = EventSequence([Event(1, 2, 1.0), Event(3, 2, 2.0), Event(3, 2, 3.0), Event(3, 4, 4.0)];
                    actors=ActorSet(1:4))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
compute(FourCycle(cycle_type=:out_out), state, 1, 4)                    # min(1, 2, 1) = 1.0
compute(FourCycle(cycle_type=:out_out, aggregation=:sum), state, 1, 4)  # 4.0
compute(FourCycle(cycle_type=:out_out, weighted=false), state, 1, 4)    # one three-path: 1.0
```
"""
struct FourCycle <: FourCycleStatistic
    cycle_type::Symbol
    weighted::Bool
    aggregation::Symbol
    stat_name::String

    function FourCycle(; cycle_type::Symbol=:out_out, weighted::Bool=true,
                       aggregation::Symbol=:min, name::String="")
        stat_name = isempty(name) ? "four_cycle_$(cycle_type)" : name
        cycle_type in (:out_out, :in_in, :out_in, :in_out, :mixed) ||
            throw(ArgumentError("cycle_type must be :out_out, :in_in, :out_in, :in_out, or :mixed"))
        new(cycle_type, weighted, _check_aggregation(aggregation), stat_name)
    end
end

function compute(stat::FourCycle, state::EventNetworkState, sender::Int, receiver::Int)
    if stat.cycle_type == :out_out
        return _compute_out_out(stat, state, sender, receiver)
    elseif stat.cycle_type == :in_in
        return _compute_in_in(stat, state, sender, receiver)
    elseif stat.cycle_type == :out_in
        return _compute_out_in(stat, state, sender, receiver)
    elseif stat.cycle_type == :in_out
        return _compute_in_out(stat, state, sender, receiver)
    else  # :mixed
        return (_compute_out_out(stat, state, sender, receiver) +
                _compute_in_in(stat, state, sender, receiver) +
                _compute_out_in(stat, state, sender, receiver) +
                _compute_in_out(stat, state, sender, receiver))
    end
end

# The contribution of one closing three-path with dyad weights (a, b, c)
@inline function _three_path_value(stat::FourCycle, a::Float64, b::Float64, c::Float64)
    stat.weighted || return 1.0
    return _aggregate(stat.aggregation, a, b, c)
end

# s→j←k→r: sender and k both send to j, k sends to receiver
function _compute_out_out(stat::FourCycle, state::EventNetworkState, sender::Int, receiver::Int)
    count = 0.0
    out_neighbors_s = get_out_neighbors(state, sender)
    in_neighbors_r = get_in_neighbors(state, receiver)

    for j in out_neighbors_s
        j == sender && continue
        j == receiver && continue

        # Find k who also sends to j and sends to r
        in_neighbors_j = get_in_neighbors(state, j)
        for k in in_neighbors_j
            k == sender && continue
            k == receiver && continue
            k == j && continue

            if k in in_neighbors_r
                count += _three_path_value(stat,
                                           get_dyad_count(state, sender, j),
                                           get_dyad_count(state, k, j),
                                           get_dyad_count(state, k, receiver))
            end
        end
    end

    return count
end

# s←j→k←r: j sends to both sender and k, receiver sends to k
function _compute_in_in(stat::FourCycle, state::EventNetworkState, sender::Int, receiver::Int)
    count = 0.0
    in_neighbors_s = get_in_neighbors(state, sender)
    out_neighbors_r = get_out_neighbors(state, receiver)

    for j in in_neighbors_s
        j == sender && continue
        j == receiver && continue

        # Find k who receives from j and receives from r
        out_neighbors_j = get_out_neighbors(state, j)
        for k in out_neighbors_j
            k == sender && continue
            k == receiver && continue
            k == j && continue

            if k in out_neighbors_r
                count += _three_path_value(stat,
                                           get_dyad_count(state, j, sender),
                                           get_dyad_count(state, j, k),
                                           get_dyad_count(state, receiver, k))
            end
        end
    end

    return count
end

# s→j→k→r: chain from sender through j and k to receiver
function _compute_out_in(stat::FourCycle, state::EventNetworkState, sender::Int, receiver::Int)
    count = 0.0
    out_neighbors_s = get_out_neighbors(state, sender)
    in_neighbors_r = get_in_neighbors(state, receiver)

    for j in out_neighbors_s
        j == sender && continue
        j == receiver && continue

        out_neighbors_j = get_out_neighbors(state, j)
        for k in out_neighbors_j
            k == sender && continue
            k == receiver && continue
            k == j && continue

            if k in in_neighbors_r
                count += _three_path_value(stat,
                                           get_dyad_count(state, sender, j),
                                           get_dyad_count(state, j, k),
                                           get_dyad_count(state, k, receiver))
            end
        end
    end

    return count
end

# s←j←k←r: reverse chain from receiver through k and j to sender
function _compute_in_out(stat::FourCycle, state::EventNetworkState, sender::Int, receiver::Int)
    count = 0.0
    in_neighbors_s = get_in_neighbors(state, sender)
    out_neighbors_r = get_out_neighbors(state, receiver)

    for j in in_neighbors_s
        j == sender && continue
        j == receiver && continue

        in_neighbors_j = get_in_neighbors(state, j)
        for k in in_neighbors_j
            k == sender && continue
            k == receiver && continue
            k == j && continue

            if k in out_neighbors_r
                count += _three_path_value(stat,
                                           get_dyad_count(state, j, sender),
                                           get_dyad_count(state, k, j),
                                           get_dyad_count(state, receiver, k))
            end
        end
    end

    return count
end

name(stat::FourCycle) = stat.stat_name

"""
    GeometricWeightedFourCycles <: FourCycleStatistic

Geometrically weighted four-cycle statistic: with `n` the **number of
distinct** closing three-paths (the `weighted=false` value of the
corresponding [`FourCycle`](@ref)),

    exp(α) · (1 − (1 − exp(−α))^n)

so each additional four-cycle adds less than the one before.

# Fields
- `cycle_type::Symbol`: Type of four-cycle configuration (as for `FourCycle`).
- `alpha::Float64`: Decay parameter (higher = less down-weighting).
- `stat_name::String`: Name for this statistic.

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(3, 2, 2.0), Event(3, 4, 3.0)];
                    actors=ActorSet(1:4))
state = EventNetworkState(seq)
for e in seq; update!(state, e); end
gw = GeometricWeightedFourCycles(cycle_type=:out_out, alpha=0.5)
compute(gw, state, 1, 4)    # one three-path (1→2←3→4): 1.0
```
"""
struct GeometricWeightedFourCycles <: FourCycleStatistic
    cycle_type::Symbol
    alpha::Float64
    stat_name::String
    # The unweighted counter, built once here rather than per evaluation
    base::FourCycle

    function GeometricWeightedFourCycles(; cycle_type::Symbol=:out_out, alpha::Float64=0.5,
                                          name::String="")
        stat_name = isempty(name) ? "gw_four_cycle_$(cycle_type)" : name
        cycle_type in (:out_out, :in_in, :out_in, :in_out, :mixed) ||
            throw(ArgumentError("cycle_type must be :out_out, :in_in, :out_in, :in_out, or :mixed"))
        alpha > 0 || throw(ArgumentError("alpha must be positive"))
        new(cycle_type, alpha, stat_name, FourCycle(cycle_type=cycle_type, weighted=false))
    end
end

function compute(stat::GeometricWeightedFourCycles, state::EventNetworkState, sender::Int, receiver::Int)
    # Unweighted count of closing three-paths
    n = compute(stat.base, state, sender, receiver)
    n == 0 && return 0.0
    # Apply geometric weighting
    return exp(stat.alpha) * (1 - (1 - exp(-stat.alpha))^n)
end

name(stat::GeometricWeightedFourCycles) = stat.stat_name
