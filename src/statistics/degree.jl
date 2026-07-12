"""
Degree-based statistics for REM.

These statistics capture effects based on the activity and popularity of actors.
"""

"""
    SenderActivity <: DegreeStatistic

Measures the sender's past activity (out-degree).
Returns the (weighted) number of past events sent by the sender.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct SenderActivity <: DegreeStatistic
    stat_name::String

    SenderActivity(; name::String="sender_activity") = new(name)
end

function compute(stat::SenderActivity, state::EventNetworkState, sender::Int, receiver::Int)
    return get_out_degree(state, sender)
end

name(stat::SenderActivity) = stat.stat_name

"""
    ReceiverActivity <: DegreeStatistic

Measures the receiver's past activity (out-degree).
Returns the (weighted) number of past events sent by the receiver.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct ReceiverActivity <: DegreeStatistic
    stat_name::String

    ReceiverActivity(; name::String="receiver_activity") = new(name)
end

function compute(stat::ReceiverActivity, state::EventNetworkState, sender::Int, receiver::Int)
    return get_out_degree(state, receiver)
end

name(stat::ReceiverActivity) = stat.stat_name

"""
    SenderPopularity <: DegreeStatistic

Measures the sender's past popularity (in-degree).
Returns the (weighted) number of past events received by the sender.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct SenderPopularity <: DegreeStatistic
    stat_name::String

    SenderPopularity(; name::String="sender_popularity") = new(name)
end

function compute(stat::SenderPopularity, state::EventNetworkState, sender::Int, receiver::Int)
    return get_in_degree(state, sender)
end

name(stat::SenderPopularity) = stat.stat_name

"""
    ReceiverPopularity <: DegreeStatistic

Measures the receiver's past popularity (in-degree).
Returns the (weighted) number of past events received by the receiver.

# Fields
- `stat_name::String`: Name for this statistic.
"""
struct ReceiverPopularity <: DegreeStatistic
    stat_name::String

    ReceiverPopularity(; name::String="receiver_popularity") = new(name)
end

function compute(stat::ReceiverPopularity, state::EventNetworkState, sender::Int, receiver::Int)
    return get_in_degree(state, receiver)
end

name(stat::ReceiverPopularity) = stat.stat_name

"""
    TotalDegree <: DegreeStatistic

Measures the total degree (in + out) of an actor.

# Fields
- `role::Symbol`: Which actor's degree to compute (:sender or :receiver).
- `stat_name::String`: Name for this statistic.
"""
struct TotalDegree <: DegreeStatistic
    role::Symbol
    stat_name::String

    function TotalDegree(; role::Symbol=:sender, name::String="")
        stat_name = isempty(name) ? "$(role)_total_degree" : name
        role in (:sender, :receiver) || throw(ArgumentError("role must be :sender or :receiver"))
        new(role, stat_name)
    end
end

function compute(stat::TotalDegree, state::EventNetworkState, sender::Int, receiver::Int)
    actor = stat.role == :sender ? sender : receiver
    return get_out_degree(state, actor) + get_in_degree(state, actor)
end

name(stat::TotalDegree) = stat.stat_name

"""
    DegreeDifference <: DegreeStatistic

Measures the difference in degree between sender and receiver.

# Fields
- `degree_type::Symbol`: Type of degree to compare (:out, :in, or :total).
- `absolute::Bool`: If true, return absolute difference.
- `stat_name::String`: Name for this statistic.
"""
struct DegreeDifference <: DegreeStatistic
    degree_type::Symbol
    absolute::Bool
    stat_name::String

    function DegreeDifference(; degree_type::Symbol=:out, absolute::Bool=false, name::String="")
        stat_name = isempty(name) ? "degree_diff_$(degree_type)$(absolute ? "_abs" : "")" : name
        degree_type in (:out, :in, :total) || throw(ArgumentError("degree_type must be :out, :in, or :total"))
        new(degree_type, absolute, stat_name)
    end
end

function compute(stat::DegreeDifference, state::EventNetworkState, sender::Int, receiver::Int)
    if stat.degree_type == :out
        sender_deg = get_out_degree(state, sender)
        receiver_deg = get_out_degree(state, receiver)
    elseif stat.degree_type == :in
        sender_deg = get_in_degree(state, sender)
        receiver_deg = get_in_degree(state, receiver)
    else  # :total
        sender_deg = get_out_degree(state, sender) + get_in_degree(state, sender)
        receiver_deg = get_out_degree(state, receiver) + get_in_degree(state, receiver)
    end

    diff = sender_deg - receiver_deg
    return stat.absolute ? abs(diff) : diff
end

name(stat::DegreeDifference) = stat.stat_name

"""
    LogDegree <: DegreeStatistic

Measures the log-transformed degree of an actor.
Uses log(1 + degree) to handle zero degrees.

# Fields
- `role::Symbol`: Which actor's degree to compute (:sender or :receiver).
- `degree_type::Symbol`: Type of degree (:out, :in, or :total).
- `stat_name::String`: Name for this statistic.
"""
struct LogDegree <: DegreeStatistic
    role::Symbol
    degree_type::Symbol
    stat_name::String

    function LogDegree(; role::Symbol=:sender, degree_type::Symbol=:out, name::String="")
        stat_name = isempty(name) ? "log_$(role)_$(degree_type)_degree" : name
        role in (:sender, :receiver) || throw(ArgumentError("role must be :sender or :receiver"))
        degree_type in (:out, :in, :total) || throw(ArgumentError("degree_type must be :out, :in, or :total"))
        new(role, degree_type, stat_name)
    end
end

function compute(stat::LogDegree, state::EventNetworkState, sender::Int, receiver::Int)
    actor = stat.role == :sender ? sender : receiver

    if stat.degree_type == :out
        deg = get_out_degree(state, actor)
    elseif stat.degree_type == :in
        deg = get_in_degree(state, actor)
    else  # :total
        deg = get_out_degree(state, actor) + get_in_degree(state, actor)
    end

    return log1p(deg)
end

name(stat::LogDegree) = stat.stat_name
