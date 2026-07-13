"""
Observation generation for REM estimation.

Implements case-control sampling and observation generation for fitting
relational event models using survival analysis methods.
"""

"""
    Observation

A single observation for model estimation, consisting of:
- Statistics computed for a potential event (sender→receiver)
- Whether the event actually occurred (case) or not (control)

# Fields
- `event_index::Int`: Index of the focal event in the sequence
- `sender::Int`: Sender ID
- `receiver::Int`: Receiver ID
- `statistics::Vector{Float64}`: Computed statistic values
- `is_event::Bool`: True if this dyad actually had an event (case)
- `stratum::Int`: Stratum ID for stratified analysis (events in same stratum share risk set)
- `risk_set_size::Int`: Number of dyads in this stratum's risk set (case included)
- `sampling_prob::Float64`: Probability with which each non-case dyad of the
  risk set was sampled as a control (1.0 when the full risk set is enumerated)
- `tie_weight::Float64`: The weight with which this row enters its stratum's
  **denominator** (the numerator is always the case's own `exp(η)`). `1.0`
  except under the Efron tie correction, where the `d` cases tied at one
  timestamp enter the *j*-th of their strata with weight `1 − (j−1)/d`; see
  `ties=` in [`generate_observations`](@ref)
"""
struct Observation
    event_index::Int
    sender::Int
    receiver::Int
    statistics::Vector{Float64}
    is_event::Bool
    stratum::Int
    risk_set_size::Int
    sampling_prob::Float64
    tie_weight::Float64
end

# Backwards-compatible constructors (no risk-set bookkeeping / no tie weight:
# an unweighted row is one with denominator weight 1)
Observation(event_index::Int, sender::Int, receiver::Int,
            statistics::Vector{Float64}, is_event::Bool, stratum::Int) =
    Observation(event_index, sender, receiver, statistics, is_event, stratum, 0, NaN,
                1.0)

Observation(event_index::Int, sender::Int, receiver::Int,
            statistics::Vector{Float64}, is_event::Bool, stratum::Int,
            risk_set_size::Int, sampling_prob::Float64) =
    Observation(event_index, sender, receiver, statistics, is_event, stratum,
                risk_set_size, sampling_prob, 1.0)

"""
    CaseControlSampler

Generates observations using case-control sampling.
For each observed event (case), samples a specified number of non-events (controls)
from the risk set.

# Fields
- `n_controls::Int`: Number of control samples per case
- `exclude_self_loops::Bool`: Whether to exclude self-loops from sampling
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
"""
struct CaseControlSampler
    n_controls::Int
    exclude_self_loops::Bool
    seed::Union{Int, Nothing}

    function CaseControlSampler(; n_controls::Int=100, exclude_self_loops::Bool=true,
                                 seed::Union{Int, Nothing}=nothing)
        n_controls > 0 || throw(ArgumentError("n_controls must be positive"))
        new(n_controls, exclude_self_loops, seed)
    end
end

"""
    _normalize_riskset(spec, event_idx::Int, exclude_self_loops::Bool) -> RiskSet

Turn a user-supplied risk-set specification for a single event into a `RiskSet`
with sorted, deduplicated sender/receiver vectors.
"""
function _normalize_riskset(rs::RiskSet, event_idx::Int, exclude_self_loops::Bool)
    senders = sort!(unique(rs.potential_senders))
    receivers = sort!(unique(rs.potential_receivers))
    return RiskSet(event_idx, senders, receivers;
                   exclude_self_loops=rs.exclude_self_loops)
end

function _normalize_riskset(spec, event_idx::Int, exclude_self_loops::Bool)
    ids = sort!(collect(Int, actor_ids(spec)))
    return RiskSet(event_idx, ids, copy(ids); exclude_self_loops=exclude_self_loops)
end

# Internal: resolve the `at_risk` keyword into a provider
# `(event_idx, state) -> RiskSet`.
#
# Supported forms:
#   nothing                      -> the sequence's actor universe (static)
#   ActorSet / Set / Vector{Int} -> a static actor universe
#   Vector of specs              -> one risk set per event (indexed by event index)
#   RiskSet                      -> a static risk set (possibly asymmetric)
#   callable                     -> `(event_index, state) -> RiskSet`, evaluated
#                                   at each event against the current network state
function _riskset_provider(at_risk, seq::EventSequence, sampler::CaseControlSampler)
    esl = sampler.exclude_self_loops

    if isnothing(at_risk)
        ids = sort!(collect(Int, seq.actors))
        return (event_idx, state) -> RiskSet(event_idx, ids, ids; exclude_self_loops=esl)
    elseif at_risk isa RiskSet
        static = _normalize_riskset(at_risk, 0, esl)
        return (event_idx, state) -> RiskSet(event_idx, static.potential_senders,
                                             static.potential_receivers;
                                             exclude_self_loops=static.exclude_self_loops)
    elseif at_risk isa ActorSet || at_risk isa AbstractSet{<:Integer} ||
           at_risk isa AbstractVector{<:Integer}
        static = _normalize_riskset(at_risk, 0, esl)
        return (event_idx, state) -> RiskSet(event_idx, static.potential_senders,
                                             static.potential_receivers;
                                             exclude_self_loops=esl)
    elseif at_risk isa AbstractVector
        # Per-event risk sets, one entry per event in the sequence
        length(at_risk) == length(seq) || throw(ArgumentError(
            "Per-event risk sets must have one entry per event: got " *
            "$(length(at_risk)) entries for $(length(seq)) events"))
        cache = Dict{Int, RiskSet}()
        return function (event_idx, state)
            get!(cache, event_idx) do
                _normalize_riskset(at_risk[event_idx], event_idx, esl)
            end
        end
    elseif at_risk isa Function
        # Callback risk set: evaluated per event against the current state
        return function (event_idx, state)
            applicable(at_risk, event_idx, state) || throw(ArgumentError(
                "Risk-set callback must be callable as " *
                "(event_index::Int, state::EventNetworkState)"))
            _normalize_riskset(at_risk(event_idx, state), event_idx, esl)
        end
    else
        throw(ArgumentError(
            "Unsupported `at_risk` specification of type $(typeof(at_risk)); pass " *
            "nothing, an ActorSet/Set{Int}/Vector{Int}, a RiskSet, a vector of " *
            "per-event risk sets, or a callback (event_index, state) -> RiskSet"))
    end
end

# Internal: the case dyad must belong to its own risk set, and the risk set must
# leave at least one valid control. Both are checked before any fitting happens.
function _validate_case!(rs::RiskSet, event::Event, event_idx::Int)
    in_rs = insorted(event.sender, rs.potential_senders) &&
            insorted(event.receiver, rs.potential_receivers) &&
            !(rs.exclude_self_loops && event.sender == event.receiver)
    in_rs || throw(ArgumentError(
        "Event $event_idx ($(event.sender) → $(event.receiver)) is not a member of " *
        "its own risk set; the case and its controls would come from different " *
        "actor universes. Declare the actor universe with " *
        "`EventSequence(events; actors=...)` or fix the `at_risk` specification."))

    n_rs = n_dyads(rs)
    n_rs >= 2 || throw(ArgumentError(
        "Risk set for event $event_idx contains $n_rs dyad(s); at least one valid " *
        "control (besides the case) is required"))
    return n_rs
end

# ============================================================================
# Tied event times (issue REM#2, review finding 12)
# ============================================================================
#
# The conditional-logit partial likelihood is a Cox partial likelihood with one
# stratum per event, so tied timestamps are exactly the classical Cox tie
# problem and the classical vocabulary applies. What the tie does *here* is
# specific, though, and worth stating: the statistics are read off the network
# state as it stands BEFORE the focal event, and the state absorbs each event as
# it is passed. Ordering a tie therefore does more than fix a sort order — it
# lets the event placed first ENTER THE STATISTICS of the event placed second
# (its Repetition, its Reciprocity, its degrees). That is the information the
# arbitrary sort invents.
#
# The policies (the shared `Networks.TIE_POLICIES` vocabulary):
#   :error    (default) refuse — the fit would depend on an arbitrary sort
#   :ordered  the legacy behaviour: sequence order, no correction
#   :breslow  one risk set per tie block (state frozen across it), each tied
#             event its own stratum with the same denominator
#   :efron    as :breslow, plus the Efron denominator weights 1 − (j−1)/d on the
#             tied cases
#   :batch    rejected: with the state frozen, a "simultaneous batch" IS Breslow

# Supported here, and why `:batch` is not.
const _REM_TIES_SUPPORTED = (:error, :ordered, :breslow, :efron)
const _REM_TIES_MODEL =
    "`fit_rem` / `generate_observations` (conditional-logit partial likelihood)"
const _REM_TIES_REASONS = Dict(
    :batch => "there is no exposure interval in an ordinal partial likelihood " *
              "for a batch to consume; holding the risk set fixed across the " *
              "tied events and giving each its own stratum IS the Breslow " *
              "correction, so pass `ties=:breslow` (or `:efron`) instead")

# Maximal runs of equal event time. `EventSequence` is time-sorted, so a tie is
# a run of length > 1 and this is one pass.
function _tie_blocks(seq::EventSequence, start_index::Int, end_index::Int)
    blocks = UnitRange{Int}[]
    i = start_index
    while i <= end_index
        j = i
        while j < end_index && seq[j + 1].time == seq[i].time
            j += 1
        end
        push!(blocks, i:j)
        i = j + 1
    end
    return blocks
end

# `ties=:error`: name the tie, do not fit.
function _reject_ties(seq::EventSequence, blocks::Vector{UnitRange{Int}})
    tied = filter(b -> length(b) > 1, blocks)
    isempty(tied) && return nothing
    b = first(tied)
    t = seq[first(b)].time
    n_tied_events = sum(length, tied)
    throw(ArgumentError(
        "Event sequence contains tied timestamps: events $(first(b))–$(last(b)) " *
        "($(length(b)) of them) all occur at t = $t" *
        (length(tied) > 1 ?
         ", and $(length(tied)) timestamps carry ties in all ($n_tied_events events)" :
         "") *
        ". The conditional-logit likelihood is a likelihood over the ORDER of " *
        "events: ordering these arbitrarily would let whichever event is placed " *
        "first enter the statistics of the ones placed after it, so the estimate " *
        "would depend on an arbitrary sort. Choose a policy explicitly: " *
        "`ties=:efron` (the Efron correction, the best approximation and what " *
        "`survival::coxph` defaults to), `ties=:breslow` (the Breslow " *
        "correction), or `ties=:ordered` (the legacy behaviour — sequence order, " *
        "no correction)."))
end

# Efron's weights re-weight each tied CASE's contribution to one shared risk-set
# denominator, which presumes the tied cases are distinct members of that risk
# set. One dyad acting twice at one timestamp is not a case classical survival
# analysis has (a subject dies once), and the fractional weight of a doubled
# member is not defined by anything — it is invented, and can even go negative.
# Refuse it rather than pick. (Breslow has no such problem: its denominator is
# the plain risk-set sum, whatever the cases are.)
function _require_distinct_tied_cases(seq::EventSequence, block::UnitRange{Int})
    dyads = [(seq[k].sender, seq[k].receiver) for k in block]
    allunique(dyads) || throw(ArgumentError(
        "ties=:efron requires the events tied at one timestamp to be distinct " *
        "dyads, but dyad $(first(d for d in dyads if count(==(d), dyads) > 1)) " *
        "acts twice at t = $(seq[first(block)].time) (events " *
        "$(first(block))–$(last(block))). Efron's correction re-weights each " *
        "tied case's contribution to ONE risk-set denominator, and a risk-set " *
        "member that is its own competitor has no such weight. Use " *
        "`ties=:breslow` (whose denominator is the plain risk-set sum and is " *
        "well defined here) or `ties=:ordered`."))
    return nothing
end

# Breslow and Efron are defined against ONE risk set shared by the tied events.
# A per-event `at_risk` that changes inside a tie block has no such thing.
function _require_shared_riskset(rs::RiskSet, rs0::RiskSet, block::UnitRange{Int},
                                 ties::Symbol)
    (rs.potential_senders == rs0.potential_senders &&
     rs.potential_receivers == rs0.potential_receivers &&
     rs.exclude_self_loops == rs0.exclude_self_loops) || throw(ArgumentError(
        "ties=:$ties requires the events tied at one timestamp (events " *
        "$(first(block))–$(last(block))) to share ONE risk set — the correction " *
        "is defined as a re-weighting of a single risk set's denominator — but " *
        "the `at_risk` specification gives them different ones. Use a risk set " *
        "that is constant across each tie block, or `ties=:ordered`."))
    return nothing
end

"""
    generate_observations(seq::EventSequence, stats, sampler::CaseControlSampler;
                          kwargs...) -> DataFrame

Generate observations for model estimation using case-control sampling.

# Arguments
- `seq::EventSequence`: The event sequence to analyze
- `stats`: Statistics to compute (a `StatisticSet` or a vector of statistics;
  vectors are converted to a tuple-backed `StatisticSet` internally so the
  inner loop is dispatch-free)
- `sampler::CaseControlSampler`: Sampling configuration

# Keyword Arguments
- `start_index::Int=1`: Index of first event to include
- `end_index::Int=length(seq)`: Index of last event to include
- `decay::Float64=0.0`: Exponential decay rate for network state
- `at_risk=nothing`: The risk set. One of
  - `nothing`: the sequence's actor universe (`seq.actors`)
  - an `ActorSet`, `Set{Int}` or `Vector{Int}`: a static actor universe
  - a `RiskSet`: a static (possibly asymmetric) sender/receiver risk set
  - a `Vector` of such specs, one per event: per-event (time-varying) risk sets
  - a callback `(event_index::Int, state::EventNetworkState) -> RiskSet` (or an
    actor collection), evaluated at each event against the current network state
- `ties::Symbol=:error`: how to handle **tied timestamps** (the shared
  `Networks.TIE_POLICIES` vocabulary). The statistics of an event are read off
  the network state as it stands before it, so ordering a tie is not a mere
  sort: it lets the event placed first enter the statistics of the event placed
  second.
  - `:error` (default) — refuse: name the tie and throw
  - `:ordered` — sequence order, no correction (the legacy behaviour)
  - `:breslow` — the Breslow correction: the tied events share one risk set (the
    network state is frozen across the tie block and absorbs the whole block at
    once) and each is a stratum with the same denominator
  - `:efron` — the Efron correction: as `:breslow`, and the `d` tied cases enter
    the denominator of the *j*-th of their strata with weight `1 − (j−1)/d`
    (carried in the `tie_weight` column). Matches `survival::coxph(..., ties="efron")`
  - `:batch` — rejected here, with a pointer to `:breslow` (see
    `Networks.check_tie_policy`)

  On tie-free data every policy produces the identical design — a tie correction
  on untied data is a no-op, and that is tested.

Every case is validated against its own risk set, and each risk set must admit
at least one control; both throw an `ArgumentError` otherwise.

# Returns
- `DataFrame`: Observations with columns for each statistic, plus `is_event`,
  `stratum`, `risk_set_size`, `sampling_prob` and `tie_weight`. The policy that
  was actually applied is attached as the DataFrame metadata key `"tie_method"`
  (`:none` when the data had no ties), so `fit_rem(::DataFrame, ...)` can report
  the truth without being told again.
"""
function generate_observations(seq::EventSequence, stats::Vector{<:AbstractStatistic},
                               sampler::CaseControlSampler; kwargs...)
    return generate_observations(seq, StatisticSet(stats), sampler; kwargs...)
end

function generate_observations(seq::EventSequence{T}, stats::StatisticSet,
                               sampler::CaseControlSampler;
                               start_index::Int=1, end_index::Int=length(seq),
                               decay::Float64=0.0,
                               at_risk=nothing, ties::Symbol=:error) where T
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)

    # Local RNG: reproducible without mutating the global RNG
    rng = isnothing(sampler.seed) ? Random.default_rng() : Random.Xoshiro(sampler.seed)

    # Tied timestamps: a tie is a run of equal times in the (time-sorted)
    # sequence. `:error` refuses to fit; the corrections freeze the network
    # state across the run, so the tied events cannot enter each other's
    # statistics; `:ordered` keeps the legacy arbitrary sort.
    blocks = _tie_blocks(seq, start_index, end_index)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(seq, blocks)
    # What was ACTUALLY done — a correction on tie-free data corrected nothing
    tie_applied = has_ties ? ties : :none
    freeze = ties === :breslow || ties === :efron

    # Initialize network state
    state = EventNetworkState(seq; decay=decay)

    # Process events before start_index to build initial state
    for i in 1:(start_index - 1)
        update!(state, seq[i])
    end

    # Resolve the risk-set specification into a per-event provider
    riskset_for = _riskset_provider(at_risk, seq, sampler)

    # Pre-allocate observation storage
    observations = Observation[]
    stat_names = stats.names

    # Process each tie block (a block of length 1 is an untied event, which is
    # every event under `:error`, and the loop then does exactly what the
    # per-event loop always did)
    for block in blocks
        d = length(block)
        local rs0::RiskSet
        ties === :efron && d > 1 && _require_distinct_tied_cases(seq, block)

        for (j, event_idx) in enumerate(block)
            event = seq[event_idx]

            # Advance the state clock without adding the event yet (counts
            # decay lazily on read relative to current_time). Under a tie
            # correction the state is NOT updated inside the block, so every
            # tied event is evaluated against the same pre-tie state.
            state.current_time = event.time

            # Risk set for this event; the case must belong to it (otherwise the
            # case and its controls come from different actor universes)
            rs = riskset_for(event_idx, state)
            risk_set_size = _validate_case!(rs, event, event_idx)
            if freeze && d > 1
                j == 1 ? (rs0 = rs) : _require_shared_riskset(rs, rs0, block, ties)
            end

            senders = rs.potential_senders
            receivers = rs.potential_receivers
            exclude_self_loops = rs.exclude_self_loops

            # Efron: the OTHER cases tied with this one stay in the denominator,
            # down-weighted by 1 − (j−1)/d, so they are excluded from the control
            # pool and re-added as explicit rows below. Every other policy
            # excludes the case dyad alone, as always.
            tie_weight = ties === :efron ? 1.0 - (j - 1) / d : 1.0
            excluded = ties === :efron ?
                Set((seq[k].sender, seq[k].receiver) for k in block) :
                Set(((event.sender, event.receiver),))
            forced = ties === :efron ?
                [(seq[k].sender, seq[k].receiver) for k in block if k != event_idx] :
                Tuple{Int,Int}[]

            # Number of distinct dyads available as controls
            max_controls = risk_set_size - length(excluded)
            n_wanted = min(sampler.n_controls, max_controls)
            if n_wanted < sampler.n_controls
                @warn "Requested $(sampler.n_controls) controls but only $max_controls " *
                      "distinct dyads are available; using the full risk set instead" maxlog = 1
            end
            # Probability that a given non-case dyad enters the stratum as a control
            sampling_prob = max_controls == 0 ? 1.0 : n_wanted / max_controls

            # Compute statistics for the actual event (case)
            case_stats = compute_all(stats, state, event.sender, event.receiver)
            push!(observations, Observation(event_idx, event.sender, event.receiver,
                                            case_stats, true, event_idx,
                                            risk_set_size, sampling_prob, tie_weight))

            # The cases tied with this one (Efron only): denominator rows, not
            # cases of THIS stratum
            for (s, r) in forced
                push!(observations,
                      Observation(event_idx, s, r, compute_all(stats, state, s, r),
                                  false, event_idx, risk_set_size, sampling_prob,
                                  tie_weight))
            end

            # Controls are drawn WITHOUT replacement: a dyad drawn k times would
            # contribute k·exp(η) to the stratum denominator and bias estimates.
            if 2 * n_wanted >= max_controls
                # Dense request: enumerate all distinct control dyads, then take
                # a random subset (or all of them)
                all_dyads = Tuple{Int,Int}[]
                for s in senders, r in receivers
                    (exclude_self_loops && s == r) && continue
                    (s, r) in excluded && continue
                    push!(all_dyads, (s, r))
                end
                chosen = n_wanted >= length(all_dyads) ? all_dyads :
                         all_dyads[randperm(rng, length(all_dyads))[1:n_wanted]]
                for (s, r) in chosen
                    control_stats = compute_all(stats, state, s, r)
                    push!(observations,
                          Observation(event_idx, s, r, control_stats, false, event_idx,
                                      risk_set_size, sampling_prob, 1.0))
                end
            else
                # Sparse request: rejection-sample distinct dyads
                n_senders = length(senders)
                n_receivers = length(receivers)
                sampled = Set{Tuple{Int,Int}}()
                while length(sampled) < n_wanted
                    s = senders[rand(rng, 1:n_senders)]
                    r = receivers[rand(rng, 1:n_receivers)]

                    (exclude_self_loops && s == r) && continue
                    (s, r) in excluded && continue
                    (s, r) in sampled && continue

                    push!(sampled, (s, r))
                    control_stats = compute_all(stats, state, s, r)
                    push!(observations,
                          Observation(event_idx, s, r, control_stats, false, event_idx,
                                      risk_set_size, sampling_prob, 1.0))
                end
            end

            # Absorb the event — unless a tie correction is in force, in which
            # case the whole block is absorbed below, after every tied event has
            # been evaluated against the same state.
            freeze || update!(state, event)
        end

        if freeze
            for k in block
                update!(state, seq[k])
            end
        end
    end

    # Convert to DataFrame, carrying the tie policy that was actually applied
    df = observations_to_dataframe(observations, stat_names)
    metadata!(df, "tie_method", string(tie_applied); style=:note)
    return df
end

"""
    observations_to_dataframe(observations::Vector{Observation}, stat_names::Vector{String}) -> DataFrame

Convert observations to a DataFrame.
"""
function observations_to_dataframe(observations::Vector{Observation}, stat_names::Vector{String})
    n_obs = length(observations)
    n_stats = length(stat_names)

    # Create columns
    data = Dict{String, Vector}()
    data["event_index"] = [o.event_index for o in observations]
    data["sender"] = [o.sender for o in observations]
    data["receiver"] = [o.receiver for o in observations]
    data["is_event"] = [o.is_event for o in observations]
    data["stratum"] = [o.stratum for o in observations]
    # Risk-set bookkeeping: the size of the stratum's risk set and the
    # probability with which each non-case dyad was sampled as a control
    data["risk_set_size"] = [o.risk_set_size for o in observations]
    data["sampling_prob"] = [o.sampling_prob for o in observations]
    # Denominator weight (1.0 everywhere except on the Efron-corrected rows of a
    # tie block); see `ties=` in `generate_observations`
    data["tie_weight"] = [o.tie_weight for o in observations]

    # Add statistic columns
    for (i, sname) in enumerate(stat_names)
        data[sname] = [o.statistics[i] for o in observations]
    end

    return DataFrame(data)
end

"""
    compute_statistics(seq::EventSequence, stats; decay::Float64=0.0) -> DataFrame

Compute statistics for all events in a sequence (without sampling controls).
`stats` may be a `StatisticSet` or a vector of statistics.

# Returns
- `DataFrame`: One row per event with computed statistics
"""
function compute_statistics(seq::EventSequence, stats::Vector{<:AbstractStatistic};
                            decay::Float64=0.0)
    return compute_statistics(seq, StatisticSet(stats); decay=decay)
end

function compute_statistics(seq::EventSequence{T}, stats::StatisticSet;
                            decay::Float64=0.0) where T
    state = EventNetworkState(seq; decay=decay)
    stat_names = stats.names

    results = Vector{Vector{Float64}}()
    senders = Int[]
    receivers = Int[]
    times = T[]

    for (i, event) in enumerate(seq)
        # Advance the state clock (counts decay lazily on read relative
        # to current_time)
        state.current_time = event.time

        # Compute statistics
        stat_values = compute_all(stats, state, event.sender, event.receiver)
        push!(results, stat_values)
        push!(senders, event.sender)
        push!(receivers, event.receiver)
        push!(times, event.time)

        # Update state
        update!(state, event)
    end

    # Build DataFrame
    df = DataFrame(
        sender = senders,
        receiver = receivers,
        time = times
    )

    for (i, sname) in enumerate(stat_names)
        df[!, sname] = [r[i] for r in results]
    end

    return df
end
