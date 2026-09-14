"""
Observation generation for REM estimation.

Implements case-control sampling and observation generation for fitting
relational event models using survival analysis methods.
"""

# The design is the `DataFrame` that `generate_observations` builds
# column-major (`_ObsBuffer` → `_to_dataframe`): one row per case or control
# with `event_index`, `sender`, `receiver`, `is_event`, `stratum`,
# `risk_set_size`, `sampling_prob`, `tie_weight` and one column per statistic.
# A hand-built design is a DataFrame with (at least) `is_event`, `stratum` and
# the statistic columns — see `fit_rem(::DataFrame, ...)`. (The pre-0.2 row
# type `Observation` and its `observations_to_dataframe` converter are gone:
# nothing produced them and nothing consumed them.)

"""
    CaseControlSampler

Generates observations using case-control sampling.
For each observed event (case), samples a specified number of non-events (controls)
from the risk set.

# Fields
- `n_controls::Int`: Number of control samples per case (controls are drawn
  **without replacement**; when fewer distinct dyads exist the full risk set
  is enumerated instead, with a one-time warning)
- `exclude_self_loops::Bool`: Whether to exclude self-loops from sampling
- `seed::Union{Int, Nothing}`: Random seed for the control draw (a local
  `Xoshiro(seed)`; takes precedence over the `rng` keyword of
  [`generate_observations`](@ref), which is used when `seed` is `nothing`)

# Example
```julia
using REM
sampler = CaseControlSampler(n_controls=100, seed=42)
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)];
                    actors=ActorSet(1:4))
obs = generate_observations(seq, [Repetition()], sampler)   # 12 dyads: full risk set
size(obs, 1)                                                # 3 cases + 3·11 controls = 36
```
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

Base.show(io::IO, s::CaseControlSampler) =
    print(io, "CaseControlSampler(n_controls=", s.n_controls, ", exclude_self_loops=",
          s.exclude_self_loops, ", seed=", s.seed === nothing ? "nothing" : s.seed, ")")

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
# `(event_idx, state) -> (RiskSet, n_dyads)`.
#
# Supported forms:
#   nothing                      -> the sequence's actor universe (static)
#   ActorSet / Set / Vector{Int} -> a static actor universe
#   Vector of specs              -> one risk set per event (indexed by event index)
#   RiskSet                      -> a static risk set (possibly asymmetric)
#   callable                     -> `(event_index, state) -> RiskSet`, evaluated
#                                   at each event against the current network state
#
# The dyad count rides with the risk set: for a static specification it is
# computed ONCE, so the per-event cost of `generate_observations` does not
# carry an O(n_actors) term (the old `n_dyads` call per event allocated two
# Sets of the whole actor universe for every event — 139 KB/event at 2000
# actors); for a per-event or callback risk set it is the allocation-free
# sorted-merge count in `n_dyads`.
function _riskset_provider(at_risk, seq::EventSequence, sampler::CaseControlSampler)
    esl = sampler.exclude_self_loops

    if isnothing(at_risk)
        ids = sort!(collect(Int, seq.actors))
        static = RiskSet(0, ids, ids; exclude_self_loops=esl)
        n_static = n_dyads(static)
        return (event_idx, state) -> (RiskSet(event_idx, ids, ids; exclude_self_loops=esl),
                                      n_static)
    elseif at_risk isa RiskSet
        static = _normalize_riskset(at_risk, 0, esl)
        n_static = n_dyads(static)
        return (event_idx, state) -> (RiskSet(event_idx, static.potential_senders,
                                              static.potential_receivers;
                                              exclude_self_loops=static.exclude_self_loops),
                                      n_static)
    elseif at_risk isa ActorSet || at_risk isa AbstractSet{<:Integer} ||
           at_risk isa AbstractVector{<:Integer}
        static = _normalize_riskset(at_risk, 0, esl)
        n_static = n_dyads(static)
        return (event_idx, state) -> (RiskSet(event_idx, static.potential_senders,
                                              static.potential_receivers;
                                              exclude_self_loops=esl),
                                      n_static)
    elseif at_risk isa AbstractVector
        # Per-event risk sets, one entry per event in the sequence
        length(at_risk) == length(seq) || throw(ArgumentError(
            "Per-event risk sets must have one entry per event: got " *
            "$(length(at_risk)) entries for $(length(seq)) events"))
        cache = Dict{Int, Tuple{RiskSet, Int}}()
        return function (event_idx, state)
            get!(cache, event_idx) do
                rs = _normalize_riskset(at_risk[event_idx], event_idx, esl)
                (rs, n_dyads(rs))
            end
        end
    elseif at_risk isa Function
        # Callback risk set: evaluated per event against the current state
        return function (event_idx, state)
            applicable(at_risk, event_idx, state) || throw(ArgumentError(
                "Risk-set callback must be callable as " *
                "(event_index::Int, state::EventNetworkState)"))
            rs = _normalize_riskset(at_risk(event_idx, state), event_idx, esl)
            (rs, n_dyads(rs))
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
# `n_rs` is the risk set's dyad count (cached by the provider for static risk
# sets, so this is O(log n) per event and allocates nothing).
function _validate_case!(rs::RiskSet, n_rs::Int, event::Event, event_idx::Int)
    # A self-loop event under `exclude_self_loops` is the common mistake, and
    # neither of the fixes for a missing actor applies to it: name it first
    (rs.exclude_self_loops && event.sender == event.receiver) && throw(ArgumentError(
        "Event $event_idx is a self-loop ($(event.sender) → $(event.sender)) and " *
        "the risk set excludes self-loops (`exclude_self_loops=true`, the " *
        "default), so the case is not a member of its own risk set. Drop " *
        "self-loop events from the sequence (`filter(e -> e.sender != " *
        "e.receiver, events)`) or pass `exclude_self_loops=false` to admit " *
        "i → i dyads as cases and controls."))
    in_rs = insorted(event.sender, rs.potential_senders) &&
            insorted(event.receiver, rs.potential_receivers)
    in_rs || throw(ArgumentError(
        "Event $event_idx ($(event.sender) → $(event.receiver)) is not a member of " *
        "its own risk set; the case and its controls would come from different " *
        "actor universes. Declare the actor universe with " *
        "`EventSequence(events; actors=...)` or fix the `at_risk` specification."))

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
- `decay::Float64=0.0`: Exponential decay rate for network state (eventnet's
  halflife model — `halflife_to_decay(h)`; the only memory model eventnet offers)
- `window=nothing`: Sliding window — events older than `current_time −
  window` stop counting in every count, degree and adjacency set. A `Real`
  in the clock's units (seconds for `Date`/`DateTime`), or a `Dates.Period`
  (`Day(2)`) on a calendar clock. The fixed-memory alternative to decay;
  **mutually exclusive with `decay > 0`** (`ArgumentError`). `window=Inf` is
  no window. See [`EventNetworkState`](@ref).
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
- `rng::AbstractRNG=Random.default_rng()`: the source of the control draw when
  the sampler has no `seed`. A `seed` on the sampler takes precedence (a local
  `Xoshiro(seed)`, reproducible whatever `rng` is); without one every draw
  flows from `rng`, so `generate_observations(...; rng=Xoshiro(9))` is
  reproducible too — never from the global RNG behind the caller's back.

Every case is validated against its own risk set, and each risk set must admit
at least one control; both throw an `ArgumentError` otherwise.

The stream is processed in **O(events)**: the per-event cost is the statistics
of one case and `n_controls` controls, independent of the number of actors (a
static risk set's dyad count is computed once; the design is accumulated
column-major into one matrix rather than one vector per row). Memory is the
design itself.

# Returns
- `DataFrame`: Observations with columns for each statistic, plus `event_index`,
  `sender`, `receiver`, `is_event`, `stratum`, `risk_set_size`, `sampling_prob`
  and `tie_weight`. The policy that was actually applied is attached as the
  DataFrame metadata key `"tie_method"` (`:none` when the data had no ties), so
  `fit_rem(::DataFrame, ...)` can report the truth without being told again.

# Example
```julia
using REM, Random
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0), Event(3, 2, 4.0)]
seq = EventSequence(events; actors=ActorSet(1:5))
stats = [Repetition(), Reciprocity()]
obs = generate_observations(seq, stats, CaseControlSampler(n_controls=5, seed=1))
size(obs, 1)                                             # 24 = 4 cases + 4·5 controls
obs2 = generate_observations(seq, stats, CaseControlSampler(n_controls=5);
                             rng=Xoshiro(9))             # rng-driven, reproducible
```
"""
function generate_observations(seq::EventSequence, stats::Vector{<:AbstractStatistic},
                               sampler::CaseControlSampler; kwargs...)
    return generate_observations(seq, StatisticSet(stats), sampler; kwargs...)
end

# A `Vector{Event}` is not an `EventSequence`: name the missing actor universe
function generate_observations(events::AbstractVector{<:Event}, args...; kwargs...)
    throw(ArgumentError(_event_vector_hint("generate_observations")))
end

function generate_observations(seq::EventSequence{T}, stats::StatisticSet,
                               sampler::CaseControlSampler;
                               start_index::Int=1, end_index::Int=length(seq),
                               decay::Float64=0.0,
                               window::_WindowSpec=nothing,
                               at_risk=nothing, ties::Symbol=:error,
                               rng::AbstractRNG=Random.default_rng()) where T
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)

    # The control draw: `sampler.seed` pins it to a local Xoshiro (reproducible
    # without touching any other stream, whatever `rng` is); otherwise it flows
    # from the caller's `rng` — never from the global RNG behind the caller's
    # back, which is the ecosystem's rng contract.
    rng = isnothing(sampler.seed) ? rng : Random.Xoshiro(sampler.seed)

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

    # Initialize network state — with the per-event log only if a statistic
    # in the set reads it (none of REM's own does; see `needs_history`)
    state = EventNetworkState(seq; decay=decay, window=window,
                              keep_history=_keeps_history(stats))

    # Process events before start_index to build initial state
    for i in 1:(start_index - 1)
        update!(state, seq[i])
    end

    # Resolve the risk-set specification into a per-event provider
    riskset_for = _riskset_provider(at_risk, seq, sampler)

    # Column-major accumulation of the design: the statistics of each row are
    # written by `compute_all!` into ONE reused buffer and appended to a flat
    # vector that IS the p × n_rows matrix, so a row costs p Float64s — not a
    # Vector{Float64} and an `Observation` per row (O(events) allocations,
    # panel 2026-09 criterion 4)
    n_events_here = max(end_index - start_index + 1, 0)
    buf = _ObsBuffer(length(stats), n_events_here * (sampler.n_controls + 1))
    # The rejection sampler's "already drawn" set, emptied per event rather
    # than rebuilt
    sampled = Set{Tuple{Int,Int}}()
    all_dyads = Tuple{Int,Int}[]

    # Process each tie block (a block of length 1 is an untied event, which is
    # every event under `:error`, and the loop then does exactly what the
    # per-event loop always did)
    for block in blocks
        d = length(block)
        local rs0::RiskSet
        efron_block = ties === :efron && d > 1
        efron_block && _require_distinct_tied_cases(seq, block)

        for (j, event_idx) in enumerate(block)
            event = seq[event_idx]

            # Advance the state clock without adding the event yet (counts
            # decay lazily on read relative to current_time). Under a tie
            # correction the state is NOT updated inside the block, so every
            # tied event is evaluated against the same pre-tie state.
            state.current_time = event.time

            # Risk set for this event; the case must belong to it (otherwise the
            # case and its controls come from different actor universes)
            rs, n_rs = riskset_for(event_idx, state)
            risk_set_size = _validate_case!(rs, n_rs, event, event_idx)
            if freeze && d > 1
                j == 1 ? (rs0 = rs) : _require_shared_riskset(rs, rs0, block, ties)
            end

            senders = rs.potential_senders
            receivers = rs.potential_receivers
            exclude_self_loops = rs.exclude_self_loops
            case_s, case_r = event.sender, event.receiver

            # Efron: the OTHER cases tied with this one stay in the denominator,
            # down-weighted by 1 − (j−1)/d, so they are excluded from the control
            # pool and re-added as explicit rows below. Every other policy
            # excludes the case dyad alone — a single tuple compare, no Set.
            tie_weight = ties === :efron ? 1.0 - (j - 1) / d : 1.0
            excluded = efron_block ?
                Set((seq[k].sender, seq[k].receiver) for k in block) : nothing
            forced = efron_block ?
                [(seq[k].sender, seq[k].receiver) for k in block if k != event_idx] :
                _NO_FORCED
            n_excluded = excluded === nothing ? 1 : length(excluded)

            # Number of distinct dyads available as controls
            max_controls = risk_set_size - n_excluded
            n_wanted = min(sampler.n_controls, max_controls)
            if n_wanted < sampler.n_controls
                @warn "Requested $(sampler.n_controls) controls but only $max_controls " *
                      "distinct dyads are available; using the full risk set instead" maxlog = 1
            end
            # Probability that a given non-case dyad enters the stratum as a control
            sampling_prob = max_controls == 0 ? 1.0 : n_wanted / max_controls

            # Compute statistics for the actual event (case)
            _push_row!(buf, stats, state, event_idx, case_s, case_r, true,
                       risk_set_size, sampling_prob, tie_weight)

            # The cases tied with this one (Efron only): denominator rows, not
            # cases of THIS stratum
            for (s, r) in forced
                _push_row!(buf, stats, state, event_idx, s, r, false,
                           risk_set_size, sampling_prob, tie_weight)
            end

            # Controls are drawn WITHOUT replacement: a dyad drawn k times would
            # contribute k·exp(η) to the stratum denominator and bias estimates.
            if 2 * n_wanted >= max_controls
                # Dense request: enumerate all distinct control dyads, then take
                # a random subset (or all of them)
                empty!(all_dyads)
                for s in senders, r in receivers
                    (exclude_self_loops && s == r) && continue
                    _is_excluded(excluded, s, r, case_s, case_r) && continue
                    push!(all_dyads, (s, r))
                end
                if n_wanted >= length(all_dyads)
                    for (s, r) in all_dyads
                        _push_row!(buf, stats, state, event_idx, s, r, false,
                                   risk_set_size, sampling_prob, 1.0)
                    end
                else
                    for k in randperm(rng, length(all_dyads))[1:n_wanted]
                        s, r = all_dyads[k]
                        _push_row!(buf, stats, state, event_idx, s, r, false,
                                   risk_set_size, sampling_prob, 1.0)
                    end
                end
            else
                # Sparse request: rejection-sample distinct dyads
                n_senders = length(senders)
                n_receivers = length(receivers)
                empty!(sampled)
                while length(sampled) < n_wanted
                    s = senders[rand(rng, 1:n_senders)]
                    r = receivers[rand(rng, 1:n_receivers)]

                    (exclude_self_loops && s == r) && continue
                    _is_excluded(excluded, s, r, case_s, case_r) && continue
                    (s, r) in sampled && continue

                    push!(sampled, (s, r))
                    _push_row!(buf, stats, state, event_idx, s, r, false,
                               risk_set_size, sampling_prob, 1.0)
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
    df = _to_dataframe(buf, stats.names)
    metadata!(df, "tie_method", string(tie_applied); style=:note)
    return df
end

# The empty `forced` list every non-Efron event shares (no per-event allocation)
const _NO_FORCED = Tuple{Int,Int}[]

# Is (s, r) excluded from the control pool? Every policy but Efron excludes the
# case dyad alone — one tuple compare, no Set; an Efron tie block excludes all
# its tied cases (re-added as weighted denominator rows).
@inline _is_excluded(::Nothing, s::Int, r::Int, case_s::Int, case_r::Int) =
    s == case_s && r == case_r
@inline _is_excluded(excluded::Set{Tuple{Int,Int}}, s::Int, r::Int, ::Int, ::Int) =
    (s, r) in excluded

# Column-major design accumulator for `generate_observations`: the bookkeeping
# columns as typed vectors and the statistics as ONE flat vector holding the
# p × n_rows matrix (rows appended p values at a time), filled through a single
# reused `vals` buffer by `compute_all!`.
struct _ObsBuffer
    event_index::Vector{Int}
    sender::Vector{Int}
    receiver::Vector{Int}
    is_event::Vector{Bool}
    stratum::Vector{Int}
    risk_set_size::Vector{Int}
    sampling_prob::Vector{Float64}
    tie_weight::Vector{Float64}
    stats::Vector{Float64}
    vals::Vector{Float64}
    p::Int
end

function _ObsBuffer(p::Int, n_rows_hint::Int)
    buf = _ObsBuffer(Int[], Int[], Int[], Bool[], Int[], Int[], Float64[], Float64[],
                     Float64[], Vector{Float64}(undef, p), p)
    n_rows_hint > 0 || return buf
    for v in (buf.event_index, buf.sender, buf.receiver, buf.is_event, buf.stratum,
              buf.risk_set_size, buf.sampling_prob, buf.tie_weight)
        sizehint!(v, n_rows_hint)
    end
    sizehint!(buf.stats, n_rows_hint * p)
    return buf
end

function _push_row!(buf::_ObsBuffer, stats::StatisticSet, state::EventNetworkState,
                    event_idx::Int, s::Int, r::Int, is_case::Bool,
                    risk_set_size::Int, sampling_prob::Float64, tie_weight::Float64)
    compute_all!(buf.vals, stats, state, s, r)
    append!(buf.stats, buf.vals)
    push!(buf.event_index, event_idx)
    push!(buf.sender, s)
    push!(buf.receiver, r)
    push!(buf.is_event, is_case)
    push!(buf.stratum, event_idx)
    push!(buf.risk_set_size, risk_set_size)
    push!(buf.sampling_prob, sampling_prob)
    push!(buf.tie_weight, tie_weight)
    return buf
end

function _to_dataframe(buf::_ObsBuffer, stat_names::Vector{String})
    p = buf.p
    n_rows = length(buf.event_index)
    df = DataFrame(
        event_index = buf.event_index,
        sender = buf.sender,
        receiver = buf.receiver,
        is_event = buf.is_event,
        stratum = buf.stratum,
        # Risk-set bookkeeping: the size of the stratum's risk set and the
        # probability with which each non-case dyad was sampled as a control
        risk_set_size = buf.risk_set_size,
        sampling_prob = buf.sampling_prob,
        # Denominator weight (1.0 everywhere except on the Efron-corrected rows
        # of a tie block); see `ties=` in `generate_observations`
        tie_weight = buf.tie_weight;
        copycols=false,
    )
    X = reshape(buf.stats, p, n_rows)
    for (k, sname) in enumerate(stat_names)
        df[!, sname] = X[k, :]
    end
    return df
end

"""
    compute_statistics(seq::EventSequence, stats; decay=0.0, window=nothing) -> DataFrame

Compute statistics for all events in a sequence (without sampling controls):
one row per event, each statistic read off the network state as it stands
**before** that event. `stats` may be a `StatisticSet` or a vector of
statistics. `decay` (eventnet's halflife model) and `window` (events older than
`current_time − window` stop counting; a `Real` in clock units or a
`Dates.Period` on a calendar clock) are the two memory models of
[`EventNetworkState`](@ref) and are mutually exclusive.

# Returns
- `DataFrame`: One row per event with `sender`, `receiver`, `time` and the
  computed statistics

# Example
```julia
using REM
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)]; actors=ActorSet(1:3))
df = compute_statistics(seq, [Repetition(), Reciprocity()])
df.repetition           # [0.0, 0.0, 1.0]
df.reciprocity          # [0.0, 1.0, 1.0]
dfw = compute_statistics(seq, [Repetition()]; window=1.0)
dfw.repetition          # [0.0, 0.0, 0.0] — the 1→2 event at t = 1 is 2.0 old at t = 3
```
"""
function compute_statistics(events::AbstractVector{<:Event}, args...; kwargs...)
    throw(ArgumentError(_event_vector_hint("compute_statistics")))
end

function compute_statistics(seq::EventSequence, stats::Vector{<:AbstractStatistic};
                            decay::Float64=0.0, window::_WindowSpec=nothing)
    return compute_statistics(seq, StatisticSet(stats); decay=decay, window=window)
end

function compute_statistics(seq::EventSequence{T}, stats::StatisticSet;
                            decay::Float64=0.0,
                            window::_WindowSpec=nothing) where T
    state = EventNetworkState(seq; decay=decay, window=window,
                              keep_history=_keeps_history(stats))
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
