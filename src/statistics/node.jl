"""
Node attribute statistics for REM.

These statistics capture effects based on actor-level attributes,
including homophily, attribute matching, and covariate effects.
"""

"""
    AttributeMatch <: NodeStatistic

Measures homophily: tendency for events between actors with matching attributes.
Returns 1.0 if sender and receiver have the same attribute value, 0.0 otherwise.

# Fields
- `attribute::NodeAttribute`: The attribute to match on.
- `stat_name::String`: Name for this statistic.
"""
struct AttributeMatch{T} <: NodeStatistic
    attribute::NodeAttribute{T}
    stat_name::String

    function AttributeMatch(attribute::NodeAttribute{T}; name::String="") where T
        stat_name = isempty(name) ? "match_$(attribute.name)" : name
        new{T}(attribute, stat_name)
    end
end

function compute(stat::AttributeMatch, state::EventNetworkState, sender::Int, receiver::Int)
    return stat.attribute[sender] == stat.attribute[receiver] ? 1.0 : 0.0
end

name(stat::AttributeMatch) = stat.stat_name

"""
    NodeMix <: NodeStatistic

Measures mixing patterns: indicator for specific sender-receiver attribute combinations.
Returns 1.0 if sender has value `sender_value` and receiver has `receiver_value`.

# Fields
- `attribute::NodeAttribute`: The attribute to check.
- `sender_value`: Required sender attribute value.
- `receiver_value`: Required receiver attribute value.
- `stat_name::String`: Name for this statistic.
"""
struct NodeMix{T} <: NodeStatistic
    attribute::NodeAttribute{T}
    sender_value::T
    receiver_value::T
    stat_name::String

    function NodeMix(attribute::NodeAttribute{T}, sender_value::T, receiver_value::T;
                     name::String="") where T
        stat_name = isempty(name) ? "mix_$(attribute.name)_$(sender_value)_$(receiver_value)" : name
        new{T}(attribute, sender_value, receiver_value, stat_name)
    end
end

function compute(stat::NodeMix, state::EventNetworkState, sender::Int, receiver::Int)
    sender_matches = stat.attribute[sender] == stat.sender_value
    receiver_matches = stat.attribute[receiver] == stat.receiver_value
    return (sender_matches && receiver_matches) ? 1.0 : 0.0
end

name(stat::NodeMix) = stat.stat_name

"""
    NodeDifference <: NodeStatistic

Measures the difference in a numeric attribute between sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `absolute::Bool`: If true, return absolute difference.
- `stat_name::String`: Name for this statistic.
"""
struct NodeDifference{T<:Number} <: NodeStatistic
    attribute::NodeAttribute{T}
    absolute::Bool
    stat_name::String

    function NodeDifference(attribute::NodeAttribute{T}; absolute::Bool=false,
                            name::String="") where T<:Number
        stat_name = isempty(name) ? "diff_$(attribute.name)$(absolute ? "_abs" : "")" : name
        new{T}(attribute, absolute, stat_name)
    end
end

function compute(stat::NodeDifference, state::EventNetworkState, sender::Int, receiver::Int)
    diff = Float64(stat.attribute[sender]) - Float64(stat.attribute[receiver])
    return stat.absolute ? abs(diff) : diff
end

name(stat::NodeDifference) = stat.stat_name

"""
    NodeSum <: NodeStatistic

Measures the sum of a numeric attribute for sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
"""
struct NodeSum{T<:Number} <: NodeStatistic
    attribute::NodeAttribute{T}
    stat_name::String

    function NodeSum(attribute::NodeAttribute{T}; name::String="") where T<:Number
        stat_name = isempty(name) ? "sum_$(attribute.name)" : name
        new{T}(attribute, stat_name)
    end
end

function compute(stat::NodeSum, state::EventNetworkState, sender::Int, receiver::Int)
    return Float64(stat.attribute[sender]) + Float64(stat.attribute[receiver])
end

name(stat::NodeSum) = stat.stat_name

"""
    NodeProduct <: NodeStatistic

Measures the product of a numeric attribute for sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
"""
struct NodeProduct{T<:Number} <: NodeStatistic
    attribute::NodeAttribute{T}
    stat_name::String

    function NodeProduct(attribute::NodeAttribute{T}; name::String="") where T<:Number
        stat_name = isempty(name) ? "product_$(attribute.name)" : name
        new{T}(attribute, stat_name)
    end
end

function compute(stat::NodeProduct, state::EventNetworkState, sender::Int, receiver::Int)
    return Float64(stat.attribute[sender]) * Float64(stat.attribute[receiver])
end

name(stat::NodeProduct) = stat.stat_name

"""
    SenderAttribute <: NodeStatistic

Returns the sender's attribute value (as a main effect).

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
"""
struct SenderAttribute{T<:Number} <: NodeStatistic
    attribute::NodeAttribute{T}
    stat_name::String

    function SenderAttribute(attribute::NodeAttribute{T}; name::String="") where T<:Number
        stat_name = isempty(name) ? "sender_$(attribute.name)" : name
        new{T}(attribute, stat_name)
    end
end

function compute(stat::SenderAttribute, state::EventNetworkState, sender::Int, receiver::Int)
    return Float64(stat.attribute[sender])
end

name(stat::SenderAttribute) = stat.stat_name

"""
    ReceiverAttribute <: NodeStatistic

Returns the receiver's attribute value (as a main effect).

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
"""
struct ReceiverAttribute{T<:Number} <: NodeStatistic
    attribute::NodeAttribute{T}
    stat_name::String

    function ReceiverAttribute(attribute::NodeAttribute{T}; name::String="") where T<:Number
        stat_name = isempty(name) ? "receiver_$(attribute.name)" : name
        new{T}(attribute, stat_name)
    end
end

function compute(stat::ReceiverAttribute, state::EventNetworkState, sender::Int, receiver::Int)
    return Float64(stat.attribute[receiver])
end

name(stat::ReceiverAttribute) = stat.stat_name

"""
    SenderCategorical <: NodeStatistic

Returns 1.0 if sender has a specific categorical attribute value.

# Fields
- `attribute::NodeAttribute`: The categorical attribute.
- `value`: The value to match.
- `stat_name::String`: Name for this statistic.
"""
struct SenderCategorical{T} <: NodeStatistic
    attribute::NodeAttribute{T}
    value::T
    stat_name::String

    function SenderCategorical(attribute::NodeAttribute{T}, value::T;
                               name::String="") where T
        stat_name = isempty(name) ? "sender_$(attribute.name)_$(value)" : name
        new{T}(attribute, value, stat_name)
    end
end

function compute(stat::SenderCategorical, state::EventNetworkState, sender::Int, receiver::Int)
    return stat.attribute[sender] == stat.value ? 1.0 : 0.0
end

name(stat::SenderCategorical) = stat.stat_name

"""
    ReceiverCategorical <: NodeStatistic

Returns 1.0 if receiver has a specific categorical attribute value.

# Fields
- `attribute::NodeAttribute`: The categorical attribute.
- `value`: The value to match.
- `stat_name::String`: Name for this statistic.
"""
struct ReceiverCategorical{T} <: NodeStatistic
    attribute::NodeAttribute{T}
    value::T
    stat_name::String

    function ReceiverCategorical(attribute::NodeAttribute{T}, value::T;
                                 name::String="") where T
        stat_name = isempty(name) ? "receiver_$(attribute.name)_$(value)" : name
        new{T}(attribute, value, stat_name)
    end
end

function compute(stat::ReceiverCategorical, state::EventNetworkState, sender::Int, receiver::Int)
    return stat.attribute[receiver] == stat.value ? 1.0 : 0.0
end

name(stat::ReceiverCategorical) = stat.stat_name
