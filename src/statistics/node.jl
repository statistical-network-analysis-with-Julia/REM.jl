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
# Example
```julia
using REM
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "M", 3 => "F"))
state = EventNetworkState{Float64}()
compute(AttributeMatch(gender), state, 1, 2)    # 1.0
compute(AttributeMatch(gender), state, 1, 3)    # 0.0
```
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
    ActorMix <: NodeStatistic

Measures mixing patterns: indicator for specific sender-receiver attribute combinations.
Returns 1.0 if sender has value `sender_value` and receiver has `receiver_value`.

Called `NodeMix` before v0.2. The name moved because ERGM.jl exports a *different*
`NodeMix` (a cross-sectional mixing-matrix term), and two distinct exported types
sharing a name make the binding ambiguous — undefined, in fact — in any session
that loads both packages. `REM.NodeMix` survives as a deprecated, non-exported
alias.

# Fields
- `attribute::NodeAttribute`: The attribute to check.
- `sender_value`: Required sender attribute value.
- `receiver_value`: Required receiver attribute value.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"))
state = EventNetworkState{Float64}()
compute(ActorMix(gender, "M", "F"), state, 1, 2)    # 1.0
compute(ActorMix(gender, "M", "F"), state, 2, 1)    # 0.0
```
"""
struct ActorMix{T} <: NodeStatistic
    attribute::NodeAttribute{T}
    sender_value::T
    receiver_value::T
    stat_name::String

    function ActorMix(attribute::NodeAttribute{T}, sender_value::T, receiver_value::T;
                      name::String="") where T
        stat_name = isempty(name) ? "mix_$(attribute.name)_$(sender_value)_$(receiver_value)" : name
        new{T}(attribute, sender_value, receiver_value, stat_name)
    end
end

# Deprecated alias for the pre-v0.2 name. NOT exported: exporting it would
# recreate the very collision with `ERGM.NodeMix` that the rename removes.
# `REM.NodeMix(...)` still constructs an `ActorMix`, with a deprecation warning.
Base.@deprecate_binding NodeMix ActorMix false

function compute(stat::ActorMix, state::EventNetworkState, sender::Int, receiver::Int)
    sender_matches = stat.attribute[sender] == stat.sender_value
    receiver_matches = stat.attribute[receiver] == stat.receiver_value
    return (sender_matches && receiver_matches) ? 1.0 : 0.0
end

name(stat::ActorMix) = stat.stat_name

# A value of the wrong type for the attribute: coerce when that is exact,
# otherwise say what the attribute holds (a bare MethodError would not)
function ActorMix(attribute::NodeAttribute{T}, sender_value, receiver_value;
                  name::String="") where T
    return ActorMix(attribute,
                    _coerce_attribute_value(attribute, sender_value, "ActorMix"),
                    _coerce_attribute_value(attribute, receiver_value, "ActorMix");
                    name=name)
end

function _coerce_attribute_value(attr::NodeAttribute{T}, v, context::String) where T
    v isa T && return v
    coerced = try
        convert(T, v)
    catch
        nothing
    end
    coerced isa T && return coerced
    example = isempty(attr.values) ? "." :
              " (the attribute's values include $(repr(first(values(attr.values)))))."
    throw(ArgumentError(
        "$context: attribute :$(attr.name) holds $T values, but $(repr(v)) is a " *
        "$(typeof(v)). Pass a value of type $T" * example))
end

# The numeric statistics on a non-numeric attribute: say which statistics
# handle a categorical attribute
function _require_numeric_attribute(attr::NodeAttribute{T}, context::String) where T
    throw(ArgumentError(
        "$context needs a numeric attribute, but :$(attr.name) holds $T values. " *
        "For a categorical attribute use AttributeMatch (homophily), ActorMix " *
        "(a sender/receiver value pair) or SenderCategorical/ReceiverCategorical " *
        "(an indicator for one value)."))
end

"""
    NodeDifference <: NodeStatistic

Measures the difference in a numeric attribute between sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `absolute::Bool`: If true, return absolute difference.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
state = EventNetworkState{Float64}()
compute(NodeDifference(age), state, 1, 2)                  # -5.0
compute(NodeDifference(age; absolute=true), state, 1, 2)   # 5.0
```
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

NodeDifference(attribute::NodeAttribute; kwargs...) =
    _require_numeric_attribute(attribute, "NodeDifference")

"""
    NodeSum <: NodeStatistic

Measures the sum of a numeric attribute for sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
compute(NodeSum(age), EventNetworkState{Float64}(), 1, 2)    # 55.0
```
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

NodeSum(attribute::NodeAttribute; kwargs...) = _require_numeric_attribute(attribute, "NodeSum")

"""
    NodeProduct <: NodeStatistic

Measures the product of a numeric attribute for sender and receiver.

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
icr = NodeAttribute(:icr, Dict(1 => 1.0, 2 => 0.0, 3 => 1.0))
compute(NodeProduct(icr), EventNetworkState{Float64}(), 1, 3)    # 1.0 (both have it)
compute(NodeProduct(icr), EventNetworkState{Float64}(), 1, 2)    # 0.0
```
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

NodeProduct(attribute::NodeAttribute; kwargs...) = _require_numeric_attribute(attribute, "NodeProduct")

"""
    SenderAttribute <: NodeStatistic

Returns the sender's attribute value (as a main effect).

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
compute(SenderAttribute(age), EventNetworkState{Float64}(), 1, 2)    # 25.0
```
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

SenderAttribute(attribute::NodeAttribute; kwargs...) =
    _require_numeric_attribute(attribute, "SenderAttribute")

"""
    ReceiverAttribute <: NodeStatistic

Returns the receiver's attribute value (as a main effect).

# Fields
- `attribute::NodeAttribute{T}`: The numeric attribute.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
compute(ReceiverAttribute(age), EventNetworkState{Float64}(), 1, 2)    # 30.0
```
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

ReceiverAttribute(attribute::NodeAttribute; kwargs...) =
    _require_numeric_attribute(attribute, "ReceiverAttribute")

"""
    SenderCategorical <: NodeStatistic

Returns 1.0 if sender has a specific categorical attribute value.

# Fields
- `attribute::NodeAttribute`: The categorical attribute.
- `value`: The value to match.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"))
compute(SenderCategorical(gender, "M"), EventNetworkState{Float64}(), 1, 2)    # 1.0
compute(SenderCategorical(gender, "M"), EventNetworkState{Float64}(), 2, 1)    # 0.0
```
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

SenderCategorical(attribute::NodeAttribute{T}, value; name::String="") where T =
    SenderCategorical(attribute, _coerce_attribute_value(attribute, value, "SenderCategorical");
                      name=name)

"""
    ReceiverCategorical <: NodeStatistic

Returns 1.0 if receiver has a specific categorical attribute value.

# Fields
- `attribute::NodeAttribute`: The categorical attribute.
- `value`: The value to match.
- `stat_name::String`: Name for this statistic.
# Example
```julia
using REM
gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"))
compute(ReceiverCategorical(gender, "F"), EventNetworkState{Float64}(), 1, 2)    # 1.0
```
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

ReceiverCategorical(attribute::NodeAttribute{T}, value; name::String="") where T =
    ReceiverCategorical(attribute, _coerce_attribute_value(attribute, value, "ReceiverCategorical");
                        name=name)
