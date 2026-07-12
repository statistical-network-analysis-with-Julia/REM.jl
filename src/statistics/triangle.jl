"""
Triangle (triadic closure) statistics for REM.

These statistics capture effects based on triadic structures in the network,
measuring various forms of closure and transitivity.
"""

"""
    TransitiveClosure <: TriangleStatistic

Measures transitive closure: tendency for s→r when there exists k such that s→k→r.
Returns the count of actors k who have received from s and sent to r.

# Fields
- `weighted::Bool`: If true, weight by edge weights.
- `stat_name::String`: Name for this statistic.
"""
struct TransitiveClosure <: TriangleStatistic
    weighted::Bool
    stat_name::String

    TransitiveClosure(; weighted::Bool=false, name::String="transitive_closure") = new(weighted, name)
end

function compute(stat::TransitiveClosure, state::EventNetworkState, sender::Int, receiver::Int)
    # Find actors k where: s→k and k→r
    # These are out-neighbors of sender who are in-neighbors of receiver
    out_neighbors = get_out_neighbors(state, sender)
    in_neighbors = get_in_neighbors(state, receiver)

    common = intersect(out_neighbors, in_neighbors)
    # Exclude sender and receiver themselves
    delete!(common, sender)
    delete!(common, receiver)

    if !stat.weighted
        return Float64(length(common))
    else
        # Weight by the minimum edge weight in each two-path
        total = 0.0
        for k in common
            w_sk = get_dyad_count(state, sender, k)
            w_kr = get_dyad_count(state, k, receiver)
            total += min(w_sk, w_kr)
        end
        return total
    end
end

name(stat::TransitiveClosure) = stat.stat_name

"""
    CyclicClosure <: TriangleStatistic

Measures cyclic closure: tendency for s→r when there exists k such that r→k→s.
Returns the count of actors k who have received from r and sent to s.

# Fields
- `weighted::Bool`: If true, weight by edge weights.
- `stat_name::String`: Name for this statistic.
"""
struct CyclicClosure <: TriangleStatistic
    weighted::Bool
    stat_name::String

    CyclicClosure(; weighted::Bool=false, name::String="cyclic_closure") = new(weighted, name)
end

function compute(stat::CyclicClosure, state::EventNetworkState, sender::Int, receiver::Int)
    # Find actors k where: r→k and k→s
    out_neighbors_r = get_out_neighbors(state, receiver)
    in_neighbors_s = get_in_neighbors(state, sender)

    common = intersect(out_neighbors_r, in_neighbors_s)
    delete!(common, sender)
    delete!(common, receiver)

    if !stat.weighted
        return Float64(length(common))
    else
        total = 0.0
        for k in common
            w_rk = get_dyad_count(state, receiver, k)
            w_ks = get_dyad_count(state, k, sender)
            total += min(w_rk, w_ks)
        end
        return total
    end
end

name(stat::CyclicClosure) = stat.stat_name

"""
    SharedSender <: TriangleStatistic

Measures shared sender effect: tendency for s→r when there exists k such that k→s and k→r.
Returns the count of actors k who have sent to both s and r.

# Fields
- `weighted::Bool`: If true, weight by edge weights.
- `stat_name::String`: Name for this statistic.
"""
struct SharedSender <: TriangleStatistic
    weighted::Bool
    stat_name::String

    SharedSender(; weighted::Bool=false, name::String="shared_sender") = new(weighted, name)
end

function compute(stat::SharedSender, state::EventNetworkState, sender::Int, receiver::Int)
    # Find actors k where: k→s and k→r
    in_neighbors_s = get_in_neighbors(state, sender)
    in_neighbors_r = get_in_neighbors(state, receiver)

    common = intersect(in_neighbors_s, in_neighbors_r)
    delete!(common, sender)
    delete!(common, receiver)

    if !stat.weighted
        return Float64(length(common))
    else
        total = 0.0
        for k in common
            w_ks = get_dyad_count(state, k, sender)
            w_kr = get_dyad_count(state, k, receiver)
            total += min(w_ks, w_kr)
        end
        return total
    end
end

name(stat::SharedSender) = stat.stat_name

"""
    SharedReceiver <: TriangleStatistic

Measures shared receiver effect: tendency for s→r when there exists k such that s→k and r→k.
Returns the count of actors k who have received from both s and r.

# Fields
- `weighted::Bool`: If true, weight by edge weights.
- `stat_name::String`: Name for this statistic.
"""
struct SharedReceiver <: TriangleStatistic
    weighted::Bool
    stat_name::String

    SharedReceiver(; weighted::Bool=false, name::String="shared_receiver") = new(weighted, name)
end

function compute(stat::SharedReceiver, state::EventNetworkState, sender::Int, receiver::Int)
    # Find actors k where: s→k and r→k
    out_neighbors_s = get_out_neighbors(state, sender)
    out_neighbors_r = get_out_neighbors(state, receiver)

    common = intersect(out_neighbors_s, out_neighbors_r)
    delete!(common, sender)
    delete!(common, receiver)

    if !stat.weighted
        return Float64(length(common))
    else
        total = 0.0
        for k in common
            w_sk = get_dyad_count(state, sender, k)
            w_rk = get_dyad_count(state, receiver, k)
            total += min(w_sk, w_rk)
        end
        return total
    end
end

name(stat::SharedReceiver) = stat.stat_name

"""
    CommonNeighbors <: TriangleStatistic

Measures the number of common neighbors (undirected) between sender and receiver.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct CommonNeighbors <: TriangleStatistic
    stat_name::String

    CommonNeighbors(; name::String="common_neighbors") = new(name)
end

function compute(stat::CommonNeighbors, state::EventNetworkState, sender::Int, receiver::Int)
    # Get all neighbors (in or out) for both actors
    neighbors_s = union(get_out_neighbors(state, sender), get_in_neighbors(state, sender))
    neighbors_r = union(get_out_neighbors(state, receiver), get_in_neighbors(state, receiver))

    common = intersect(neighbors_s, neighbors_r)
    delete!(common, sender)
    delete!(common, receiver)

    return Float64(length(common))
end

name(stat::CommonNeighbors) = stat.stat_name

"""
    GeometricWeightedTriads <: TriangleStatistic

Geometrically weighted shared partner statistic.
Down-weights the contribution of additional shared partners.

# Fields
- `closure_type::Symbol`: Type of closure (:transitive, :cyclic, :shared_sender, :shared_receiver).
- `alpha::Float64`: Decay parameter (higher = less down-weighting).
- `stat_name::String`: Name for this statistic.
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
    # Get the appropriate common set based on closure type
    if stat.closure_type == :transitive
        out_neighbors = get_out_neighbors(state, sender)
        in_neighbors = get_in_neighbors(state, receiver)
        common = intersect(out_neighbors, in_neighbors)
    elseif stat.closure_type == :cyclic
        out_neighbors_r = get_out_neighbors(state, receiver)
        in_neighbors_s = get_in_neighbors(state, sender)
        common = intersect(out_neighbors_r, in_neighbors_s)
    elseif stat.closure_type == :shared_sender
        in_neighbors_s = get_in_neighbors(state, sender)
        in_neighbors_r = get_in_neighbors(state, receiver)
        common = intersect(in_neighbors_s, in_neighbors_r)
    else  # :shared_receiver
        out_neighbors_s = get_out_neighbors(state, sender)
        out_neighbors_r = get_out_neighbors(state, receiver)
        common = intersect(out_neighbors_s, out_neighbors_r)
    end

    delete!(common, sender)
    delete!(common, receiver)

    n = length(common)
    if n == 0
        return 0.0
    end

    # Geometrically weighted sum: exp(alpha) * (1 - (1 - exp(-alpha))^n)
    return exp(stat.alpha) * (1 - (1 - exp(-stat.alpha))^n)
end

name(stat::GeometricWeightedTriads) = stat.stat_name
