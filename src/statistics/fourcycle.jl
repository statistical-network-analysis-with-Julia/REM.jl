"""
Four-cycle statistics for REM.

These statistics capture local clustering effects based on four-node structures,
measuring the closure of three-paths through two intermediate nodes.
"""

"""
    FourCycle <: FourCycleStatistic

Measures four-cycle closure: tendency for s→r when there exist j, k such that
s→j, k→j, and k→r (or variants).

This captures local clustering where sender and receiver share connections
to a common pair of intermediaries.

# Fields
- `cycle_type::Symbol`: Type of four-cycle configuration.
    - :out_out: s→j←k→r (shared out-neighbor pattern)
    - :in_in: s←j→k←r (shared in-neighbor pattern)
    - :out_in: s→j→k→r (two-path through intermediaries)
    - :mixed: any configuration
- `weighted::Bool`: If true, weight by edge weights.
- `stat_name::String`: Name for this statistic.
"""
struct FourCycle <: FourCycleStatistic
    cycle_type::Symbol
    weighted::Bool
    stat_name::String

    function FourCycle(; cycle_type::Symbol=:out_out, weighted::Bool=false, name::String="")
        stat_name = isempty(name) ? "four_cycle_$(cycle_type)" : name
        cycle_type in (:out_out, :in_in, :out_in, :in_out, :mixed) ||
            throw(ArgumentError("cycle_type must be :out_out, :in_in, :out_in, :in_out, or :mixed"))
        new(cycle_type, weighted, stat_name)
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
                if !stat.weighted
                    count += 1.0
                else
                    w_sj = get_dyad_count(state, sender, j)
                    w_kj = get_dyad_count(state, k, j)
                    w_kr = get_dyad_count(state, k, receiver)
                    count += min(w_sj, w_kj, w_kr)
                end
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
                if !stat.weighted
                    count += 1.0
                else
                    w_js = get_dyad_count(state, j, sender)
                    w_jk = get_dyad_count(state, j, k)
                    w_rk = get_dyad_count(state, receiver, k)
                    count += min(w_js, w_jk, w_rk)
                end
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
                if !stat.weighted
                    count += 1.0
                else
                    w_sj = get_dyad_count(state, sender, j)
                    w_jk = get_dyad_count(state, j, k)
                    w_kr = get_dyad_count(state, k, receiver)
                    count += min(w_sj, w_jk, w_kr)
                end
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
                if !stat.weighted
                    count += 1.0
                else
                    w_js = get_dyad_count(state, j, sender)
                    w_kj = get_dyad_count(state, k, j)
                    w_rk = get_dyad_count(state, receiver, k)
                    count += min(w_js, w_kj, w_rk)
                end
            end
        end
    end

    return count
end

name(stat::FourCycle) = stat.stat_name

"""
    GeometricWeightedFourCycles <: FourCycleStatistic

Geometrically weighted four-cycle statistic.
Down-weights the contribution of additional four-cycles.

# Fields
- `cycle_type::Symbol`: Type of four-cycle configuration.
- `alpha::Float64`: Decay parameter (higher = less down-weighting).
- `stat_name::String`: Name for this statistic.
"""
struct GeometricWeightedFourCycles <: FourCycleStatistic
    cycle_type::Symbol
    alpha::Float64
    stat_name::String

    function GeometricWeightedFourCycles(; cycle_type::Symbol=:out_out, alpha::Float64=0.5,
                                          name::String="")
        stat_name = isempty(name) ? "gw_four_cycle_$(cycle_type)" : name
        cycle_type in (:out_out, :in_in, :out_in, :in_out, :mixed) ||
            throw(ArgumentError("Invalid cycle_type"))
        alpha > 0 || throw(ArgumentError("alpha must be positive"))
        new(cycle_type, alpha, stat_name)
    end
end

function compute(stat::GeometricWeightedFourCycles, state::EventNetworkState, sender::Int, receiver::Int)
    # Get unweighted count first
    base_stat = FourCycle(cycle_type=stat.cycle_type, weighted=false)
    n = compute(base_stat, state, sender, receiver)

    if n == 0
        return 0.0
    end

    # Apply geometric weighting
    return exp(stat.alpha) * (1 - (1 - exp(-stat.alpha))^n)
end

name(stat::GeometricWeightedFourCycles) = stat.stat_name
