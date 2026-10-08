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

The sampler holds no random state: the control draw comes from the `rng`
keyword of [`generate_observations`](@ref) (or [`fit_rem`](@ref)), the
ecosystem's one randomness keyword.

# Example
```julia
using REM, Random
sampler = CaseControlSampler(n_controls=100)
seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)];
                    actors=ActorSet(1:4))
obs = generate_observations(seq, [Repetition()], sampler; rng=Xoshiro(42))   # 12 dyads: full risk set
size(obs, 1)                                                # 3 cases + 3·11 controls = 36
```
"""
struct CaseControlSampler
    n_controls::Int
    exclude_self_loops::Bool

    function CaseControlSampler(; n_controls::Int=100, exclude_self_loops::Bool=true)
        n_controls > 0 || throw(ArgumentError("n_controls must be positive"))
        new(n_controls, exclude_self_loops)
    end
end

function Base.show(io::IO, s::CaseControlSampler)
    print(io, "CaseControlSampler(n_controls=", s.n_controls, ", exclude_self_loops=",
          s.exclude_self_loops, ")")
end

"""
    _normalize_riskset(spec, event_idx::Int, exclude_self_loops::Bool) -> RiskSet

Turn a user-supplied risk-set specification for a single event into a `RiskSet`
with sorted, deduplicated sender/receiver vectors.
"""
function _normalize_riskset(rs::RiskSet, event_idx::Int, exclude_self_loops::Bool)
    rs.dyads === nothing ||
        return RiskSet(event_idx, rs.potential_senders, rs.potential_receivers,
                       rs.exclude_self_loops, rs.dyads)
    senders = sort!(unique(rs.potential_senders))
    receivers = sort!(unique(rs.potential_receivers))
    return RiskSet(event_idx, senders, receivers;
                   exclude_self_loops=rs.exclude_self_loops)
end

# A list of (sender, receiver) pairs: a dyad-level risk set
const _DyadList = Union{AbstractVector{<:Tuple{Integer,Integer}},
                        AbstractSet{<:Tuple{Integer,Integer}}}
_normalize_riskset(dyads::_DyadList, event_idx::Int, exclude_self_loops::Bool) =
    RiskSet(event_idx, dyads)

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
        return _static_provider(ids, ids, esl)
    elseif at_risk isa RiskSet || at_risk isa _DyadList
        rs = _normalize_riskset(at_risk, 0, esl)
        return _static_provider(rs.potential_senders, rs.potential_receivers,
                                rs.exclude_self_loops, rs.dyads)
    elseif at_risk isa ActorSet || at_risk isa AbstractSet{<:Integer} ||
           at_risk isa AbstractVector{<:Integer}
        rs = _normalize_riskset(at_risk, 0, esl)
        return _static_provider(rs.potential_senders, rs.potential_receivers, esl)
    elseif at_risk isa AbstractVector
        # Per-event risk sets, one entry per event in the sequence
        length(at_risk) == length(seq) || throw(ArgumentError(
            "Per-event risk sets must have one entry per event: got " *
            "$(length(at_risk)) entries for $(length(seq)) events"))
        return _per_event_provider(at_risk, esl)
    elseif at_risk isa Function
        return _callback_provider(at_risk, esl)
    else
        throw(ArgumentError(
            "Unsupported `at_risk` specification of type $(typeof(at_risk)); pass " *
            "nothing, an ActorSet/Set{Int}/Vector{Int}, a RiskSet, a list of " *
            "(sender, receiver) dyads, a vector of per-event risk sets, or a " *
            "callback (event_index, state) -> RiskSet"))
    end
end

# The three provider closures are built by functions of their own, so that each
# captures only its own (never reassigned) arguments: built inline in the
# branches above, the captured `static`/`n_static` were assigned in several
# branches and Julia boxed them (`Core.Box`) — a dynamically
# typed read on every event.
function _static_provider(senders::Vector{Int}, receivers::Vector{Int}, esl::Bool,
                          dyads::Union{Nothing, Vector{Tuple{Int,Int}}}=nothing)
    # The dyad count of a static risk set is computed ONCE
    n_static = n_dyads(RiskSet(0, senders, receivers, esl, dyads))
    return (event_idx, state) -> (RiskSet(event_idx, senders, receivers, esl, dyads),
                                  n_static)
end

function _per_event_provider(at_risk::AbstractVector, esl::Bool)
    cache = Dict{Int, Tuple{RiskSet, Int}}()
    return function (event_idx, state)
        get!(cache, event_idx) do
            rs = _normalize_riskset(at_risk[event_idx], event_idx, esl)
            (rs, n_dyads(rs))
        end
    end
end

function _callback_provider(at_risk::F, esl::Bool) where F
    # Callback risk set: evaluated per event against the current state
    return function (event_idx, state)
        applicable(at_risk, event_idx, state) || throw(ArgumentError(
            "Risk-set callback must be callable as " *
            "(event_index::Int, state::EventNetworkState)"))
        rs = _normalize_riskset(at_risk(event_idx, state), event_idx, esl)
        (rs, n_dyads(rs))
    end
end

# Internal: the case dyad must belong to its own risk set, and the risk set must
# leave at least one valid control. Both are checked before any fitting happens.
# `n_rs` is the risk set's dyad count (cached by the provider for static risk
# sets, so this is O(log n) per event and allocates nothing).
# Is the dyad a member of the risk set?
@inline function _in_riskset(rs::RiskSet, s::Int, r::Int)
    dy = rs.dyads
    dy === nothing || return insorted((s, r), dy)
    (rs.exclude_self_loops && s == r) && return false
    return insorted(s, rs.potential_senders) && insorted(r, rs.potential_receivers)
end

function _validate_case!(rs::RiskSet, n_rs::Int, event::Event, event_idx::Int)
    if rs.dyads !== nothing
        _in_riskset(rs, event.sender, event.receiver) || throw(ArgumentError(
            "Event $event_idx ($(event.sender) → $(event.receiver)) is not one of " *
            "the $(length(rs.dyads)) dyads of its dyad-level risk set; a case must " *
            "be at risk. Add the dyad to the list or fix the `at_risk` specification."))
        n_rs >= 2 || throw(ArgumentError(
            "Risk set for event $event_idx contains $n_rs dyad(s); at least one " *
            "valid control (besides the case) is required"))
        return n_rs
    end
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
# Tied event times
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
# The policies (the shared `NetworkCore.TIE_POLICIES` vocabulary):
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
              "correction, so pass `ties=:breslow` (or `:efron` with the full " *
              "risk set) instead")

# Maximal runs of equal event time over the events `start_index:end_index`,
# each taken to its FULL extent in the sequence: a run that straddles
# `start_index` (or `end_index`) is returned whole, so the events before the
# window that are tied with the first event of the window are recognised as a
# tie (they are not absorbed into the state before the window opens — the
# pre-0.2 code absorbed them and let them leak into the statistics of their
# simultaneous partners, bypassing `ties=:error`). `EventSequence` is time-sorted, so a tie is a run of length > 1 and this
# is one pass.
function _tie_blocks(seq::EventSequence, start_index::Int, end_index::Int)
    blocks = UnitRange{Int}[]
    start_index > end_index && return blocks
    lo, hi = start_index, end_index
    while lo > 1 && seq[lo - 1].time == seq[start_index].time
        lo -= 1
    end
    while hi < length(seq) && seq[hi + 1].time == seq[end_index].time
        hi += 1
    end
    i = lo
    while i <= hi
        j = i
        while j < hi && seq[j + 1].time == seq[i].time
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
        "`ties=:breslow` (the Breslow correction; the one to use with sampled " *
        "controls), `ties=:efron` (the Efron correction, what " *
        "`survival::coxph` defaults to — with the FULL risk set only), or " *
        "`ties=:ordered` (the legacy behaviour — sequence order, no correction)."))
end

"""
    REM.check_tie_sampling(ties, block_size, n_controls, n_eligible; context="")

The guard that keeps the **Efron tie correction** away from **sampled
controls**. It throws an `ArgumentError` when `ties === :efron`, the tie block
holds `block_size > 1` events and only `n_controls < n_eligible` of the
`n_eligible` non-case dyads of its risk set would be kept; otherwise it returns
`nothing`.

Why the combination is refused. Efron's denominator for the *j*-th of `d`
tied events is the sum over the risk set with the `d` tied cases down-weighted
by `1 − (j−1)/d`. With sampled controls the other tied cases have to stay in
every stratum of the block (with probability one), while the other dyads
enter with probability `m/(N−d)`: the tied cases — the dyads that just acted,
with high linear predictors — then dominate the sampled denominator by a
factor of about `(N−d)/m`, and the estimate is biased toward zero with
confident standard errors (truth 1.0: 0.63 at 5 controls, 0.88 at 100, with
0 % coverage). Neither Horvitz–Thompson weights on
the controls (they overcorrect: 1.47) nor nested case-control sampling with
the Efron weight on any tied case that happens to be drawn (+12 % at 5
controls) removes the bias: Efron's factors are not the conditional
probabilities that make the case-control argument work. Breslow's are — the
Breslow correction is the likelihood of the tied events drawn independently
from one frozen risk set, so each of its strata is an ordinary
nested-case-control stratum, and **`ties=:breslow` is unbiased with sampled
controls** (pinned by simulation in the test suite).

`REM.check_tie_sampling` is `public`: Revel.jl's `event_design` calls it, so
the two packages refuse the same combination with the same words.

# Example
```julia
using REM
REM.check_tie_sampling(:efron, 3, 1000, 600)      # nothing — the full risk set is kept
REM.check_tie_sampling(:breslow, 3, 5, 600)       # nothing — Breslow may be sampled
REM.check_tie_sampling(:efron, 1, 5, 600)         # nothing — an untied event
err = try REM.check_tie_sampling(:efron, 3, 5, 600) catch e e end
err isa ArgumentError                             # true
```
"""
function check_tie_sampling(ties::Symbol, block_size::Integer, n_controls::Integer,
                            n_eligible::Integer; context::AbstractString="")
    (ties === :efron && block_size > 1 && n_controls < n_eligible) || return nothing
    prefix = isempty(context) ? "" : "$context: "
    throw(ArgumentError(
        prefix * "ties=:efron cannot be combined with sampled controls: a block " *
        "of $block_size tied events would keep only $n_controls of the " *
        "$n_eligible eligible control dyads of its risk set. Efron's correction " *
        "keeps the other tied cases in every stratum of the block with " *
        "probability one while the controls are sampled, and the tied cases " *
        "then dominate the sampled denominator — the estimate is biased toward " *
        "zero with confident standard errors. Use `ties=:breslow`, which is " *
        "unbiased with sampled controls, or keep `ties=:efron` with the full " *
        "risk set (`n_controls` at least the number of eligible controls, " *
        "$n_eligible here)."))
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
     rs.exclude_self_loops == rs0.exclude_self_loops &&
     rs.dyads == rs0.dyads) || throw(ArgumentError(
        "ties=:$ties requires the events tied at one timestamp (events " *
        "$(first(block))–$(last(block))) to share ONE risk set — the correction " *
        "is defined as a re-weighting of a single risk set's denominator — but " *
        "the `at_risk` specification gives them different ones. Use a risk set " *
        "that is constant across each tie block, or `ties=:ordered`."))
    return nothing
end

# ============================================================================
# Event types
# ============================================================================
#
# An untyped statistic counts the events of all types together — a `:fight`
# repeats a `:help` — which is a modelling decision the user has to take, not
# one to take silently. `eventtypes=:error` (default) refuses a sequence with
# more than one type; the explicit choices are `eventtypes=:pool` (every event
# a case, untyped statistics pooled) and `cases_of=` (the events of some types
# are the cases; all events build the history). `OfType(stat, type)` reads the
# history of one type under either (typed sub-states of the network state).
const _EVENTTYPE_POLICIES = (:error, :pool)

function _check_eventtypes(seq::EventSequence, policy::Symbol, context::AbstractString;
                           acknowledged::Bool=false)
    policy in _EVENTTYPE_POLICIES || throw(ArgumentError(
        "$context: `eventtypes` must be :error (refuse a sequence with more than " *
        "one event type) or :pool (count the events of every type together); " *
        "got :$policy"))
    (policy === :error && !acknowledged && length(seq.eventtypes) > 1) || return nothing
    types = join((":" * string(t) for t in sort!(collect(seq.eventtypes))), ", ")
    throw(ArgumentError(
        "$context: the sequence carries $(length(seq.eventtypes)) event types " *
        "($types). Say how they enter the model: `eventtypes=:pool` makes every " *
        "event a case and lets untyped statistics count the events of all types " *
        "together; `cases_of=:x` (on `fit_rem`/`generate_observations`) models " *
        "the events of type :x only, with every event still building the " *
        "history. Either way, `OfType(stat, :x)` restricts a statistic to the " *
        "history of one type. Typed risk sets and stratified fits by type are " *
        "Revel.jl's (`fit_stratified(by = e -> e.eventtype)`)."))
end

# ============================================================================
# Undirected events
# ============================================================================
#
# `directed=false` treats every event as an unordered pair: the sequence is
# rewritten with each pair as (min, max) — so nothing can depend on the order
# in which a pair happened to be stored — and the risk set holds the
# n(n−1)/2 unordered pairs `s < r` of a SYMMETRIC actor set (plus `s == r`
# when self-loops are admitted), not the n(n−1) ordered dyads: an undirected
# sequence fitted against the ordered dyads has every non-case pair in its
# risk set twice.
_case_types(::Nothing, seq::EventSequence, context) = nothing
function _case_types(cases_of, seq::EventSequence, context)
    types = cases_of isa Symbol ? Symbol[cases_of] : collect(Symbol, cases_of)
    isempty(types) && throw(ArgumentError("$context: `cases_of` names no event type"))
    for t in types
        t in seq.eventtypes || throw(ArgumentError(
            "$context: `cases_of` names the event type :$t, but the sequence has " *
            "no event of that type (its types: " *
            join((":" * string(u) for u in sort!(collect(seq.eventtypes))), ", ") * ")"))
    end
    return types
end

function _undirected_sequence(seq::EventSequence{T}) where T
    events = [Event(minmax(e.sender, e.receiver)..., e.time;
                    eventtype=e.eventtype, weight=e.weight) for e in seq]
    return seq.actors_declared ? EventSequence(events; actors=collect(Int, seq.actors)) :
                                 EventSequence(events)
end

# The number of unordered pairs of a symmetric risk set
function _n_pairs(rs::RiskSet)
    rs.dyads === nothing || return length(rs.dyads)
    m = length(rs.potential_senders)
    return rs.exclude_self_loops ? m * (m - 1) ÷ 2 : m * (m + 1) ÷ 2
end

function _require_symmetric(rs::RiskSet, event_idx::Int)
    if rs.dyads !== nothing
        all(d -> d[1] <= d[2], rs.dyads) || throw(ArgumentError(
            "directed=false with a dyad-level risk set needs every listed pair " *
            "written as (smaller, larger) — an unordered pair is stored once — " *
            "but the risk set of event $event_idx lists a pair the other way round"))
        return nothing
    end
    (rs.potential_senders === rs.potential_receivers ||
     rs.potential_senders == rs.potential_receivers) || throw(ArgumentError(
        "directed=false needs a symmetric risk set (the same actors as senders " *
        "and as receivers: an unordered pair has no sender side), but the risk " *
        "set of event $event_idx has $(length(rs.potential_senders)) senders and " *
        "$(length(rs.potential_receivers)) different receivers"))
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
- `start_index::Int=1`: Index of first event to include as a case (the events
  before it build the network state only). A tie block that straddles
  `start_index` is treated as a tie: under `ties=:error` it is refused, under
  `:breslow`/`:efron` the earlier members of the block do not enter the state
  before the block's own strata are built (they are the block's other tied
  cases, Efron-weighted under `:efron`), and only `:ordered` absorbs them first
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
  - a list of `(sender, receiver)` pairs (or `RiskSet(i, pairs)`): a
    **dyad-level** risk set — exactly the listed dyads are at risk
  - a `Vector` of such specs, one per event: per-event (time-varying) risk sets
  - a callback `(event_index::Int, state::EventNetworkState) -> RiskSet` (or an
    actor collection), evaluated at each event against the current network state
- `directed::Bool=true`: `false` treats the events as **undirected**: each
  event is rewritten as the pair `(min, max)` and the risk set holds the
  unordered pairs `s < r` of a symmetric actor set (`n(n−1)/2` of them), not
  the `n(n−1)` ordered dyads. Use undirected statistics with it
  (`Repetition(directed=false)`, `CommonNeighbors`, …): a directed statistic
  would read the pair in ID order, which carries no meaning
- `ties::Symbol=:error`: how to handle **tied timestamps** (the shared
  `NetworkCore.TIE_POLICIES` vocabulary). The statistics of an event are read off
  the network state as it stands before it, so ordering a tie is not a mere
  sort: it lets the event placed first enter the statistics of the event placed
  second.
  - `:error` (default) — refuse: name the tie and throw
  - `:ordered` — sequence order, no correction (the legacy behaviour)
  - `:breslow` — the Breslow correction: the tied events share one risk set (the
    network state is frozen across the tie block and absorbs the whole block at
    once) and each is a stratum with the same denominator. Valid with sampled
    controls: each stratum is an ordinary nested-case-control stratum
  - `:efron` — the Efron correction: as `:breslow`, and the `d` tied cases enter
    the denominator of the *j*-th of their strata with weight `1 − (j−1)/d`
    (carried in the `tie_weight` column). Matches `survival::coxph(..., ties="efron")`.
    **Full risk set only**: a tie block whose risk set would be sampled is
    refused (see [`REM.check_tie_sampling`](@ref) for why)
  - `:batch` — rejected here, with a pointer to `:breslow` (see
    `NetworkCore.check_tie_policy`)

  On tie-free data every policy produces the identical design — a tie correction
  on untied data is a no-op, and that is tested.
- `eventtypes::Symbol=:error`: a sequence with more than one event type is
  refused until you say how the types enter the model. `:pool` makes every
  event a case and lets untyped statistics count all types together;
  [`OfType`](@ref) statistics read one type's history.
- `cases_of=nothing`: an event type (or a collection of types) to model —
  only those events are cases, every event still builds the history. The
  likelihood is then conditional on the type of each event. Combine with
  [`OfType`](@ref) statistics for type-specific effects.
- `rng::AbstractRNG=Random.default_rng()`: the source of the control draw, so
  `generate_observations(...; rng=Xoshiro(9))` is reproducible — never the
  global RNG behind the caller's back.

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
obs = generate_observations(seq, stats, CaseControlSampler(n_controls=5);
                            rng=Xoshiro(1))              # rng-driven, reproducible
size(obs, 1)                                             # 24 = 4 cases + 4·5 controls
und = generate_observations(seq, [Repetition(directed=false)],
                            CaseControlSampler(n_controls=100); directed=false)
und.risk_set_size[1]                                     # 10 = 5·4/2 unordered pairs
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
                               directed::Bool=true, eventtypes::Symbol=:error,
                               cases_of=nothing,
                               rng::AbstractRNG=Random.default_rng()) where T
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)
    case_types = _case_types(cases_of, seq, "generate_observations")
    _check_eventtypes(seq, eventtypes, "generate_observations";
                      acknowledged=case_types !== nothing)
    typed = _check_typed_needs(_typed_needs(stats), seq, "generate_observations")
    (1 <= start_index && end_index <= length(seq)) || throw(ArgumentError(
        "start_index = $start_index and end_index = $end_index must lie within " *
        "the sequence's 1:$(length(seq))"))

    # Undirected events: one orientation per pair, and unordered pairs at risk
    directed || (seq = _undirected_sequence(seq))

    # Tied timestamps: a tie is a run of equal times in the (time-sorted)
    # sequence, taken to its full extent even where it straddles the window.
    # `:error` refuses to fit; the corrections freeze the network state across
    # the run, so the tied events cannot enter each other's statistics;
    # `:ordered` keeps the legacy arbitrary sort.
    blocks = _tie_blocks(seq, start_index, end_index)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(seq, blocks)
    # What was ACTUALLY done — a correction on tie-free data corrected nothing
    tie_applied = has_ties ? ties : :none
    freeze = ties === :breslow || ties === :efron

    # Initialize network state — with the per-event log only if a statistic
    # in the set reads it (none of REM's own does; see `needs_history`)
    state = EventNetworkState(seq; decay=decay, window=window,
                              keep_history=_keeps_history(stats), types=typed)

    # Process the events before the first block to build the initial state.
    # The first block starts at `start_index` unless it straddles it; its
    # earlier members are then handled with the block (absorbed in sequence
    # order under `:ordered`, frozen with it under a correction).
    first_block = isempty(blocks) ? start_index : first(first(blocks))
    for i in 1:(first_block - 1)
        update!(state, seq[i])
    end

    # Resolve the risk-set specification into a per-event provider
    riskset_for = _riskset_provider(at_risk, seq, sampler)

    # Column-major accumulation of the design: the statistics of each row are
    # written by `compute_all!` into ONE reused buffer and appended to a flat
    # vector that IS the p × n_rows matrix, so a row costs p Float64s — not a
    # Vector{Float64} and an `Observation` per row (O(events) allocations)
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
        rs0 = nothing
        efron_block = ties === :efron && d > 1
        efron_block && _require_distinct_tied_cases(seq, block)

        for (j, event_idx) in enumerate(block)
            event = seq[event_idx]

            # A member of a straddling block outside the window is not a case:
            # it enters the state (in sequence order under `:ordered`, with the
            # whole block under a correction) and, under Efron, the denominator
            # of the block's strata as one of the tied cases
            if !(start_index <= event_idx <= end_index) ||
               (case_types !== nothing && !(event.eventtype in case_types))
                # (an event of a type that is not modelled — `cases_of` — builds
                # the history in the same way)
                freeze || update!(state, event)
                continue
            end

            # Advance the state clock without adding the event yet (counts
            # decay lazily on read relative to current_time). Under a tie
            # correction the state is NOT updated inside the block, so every
            # tied event is evaluated against the same pre-tie state.
            state.current_time = event.time

            # Risk set for this event; the case must belong to it (otherwise the
            # case and its controls come from different actor universes)
            rs, n_rs = riskset_for(event_idx, state)
            directed || _require_symmetric(rs, event_idx)
            n_rs = directed ? n_rs : _n_pairs(rs)
            risk_set_size = _validate_case!(rs, n_rs, event, event_idx)
            if freeze && d > 1
                rs0 === nothing ? (rs0 = rs) :
                    _require_shared_riskset(rs, rs0::RiskSet, block, ties)
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
            if efron_block
                for k in block
                    _in_riskset(rs, seq[k].sender, seq[k].receiver) || throw(ArgumentError(
                        "ties=:efron keeps every event of a tie block in the " *
                        "denominator, but event $k is not in the risk set of the " *
                        "tied event $event_idx"))
                end
            end
            dyad_list = rs.dyads
            forced = efron_block ?
                [(seq[k].sender, seq[k].receiver) for k in block if k != event_idx] :
                _NO_FORCED
            n_excluded = excluded === nothing ? 1 : length(excluded)

            # Number of distinct dyads available as controls
            max_controls = risk_set_size - n_excluded
            # Efron with a sampled risk set is biased: the forced tied rows enter with probability one and the
            # sampled controls do not. Refuse it, pointing at `:breslow`.
            efron_block && check_tie_sampling(ties, d, sampler.n_controls, max_controls;
                                              context="generate_observations")
            n_wanted = min(sampler.n_controls, max_controls)
            # More controls than the risk set holds besides the case: say so.
            # (Measured against the case alone, not the Efron block's other
            # tied cases — `n_controls = n(n−1) − 1`, the documented idiom for
            # the full risk set, must not warn because a tie is present.)
            if sampler.n_controls > risk_set_size - 1
                @warn "Requested $(sampler.n_controls) controls but only " *
                      "$(risk_set_size - 1) distinct dyads are available; using " *
                      "the full risk set instead" maxlog = 1
            end
            # Probability that a given non-case dyad enters the stratum as a
            # control (1 for the forced tied rows of an Efron block, which is
            # never sampled)
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
                if dyad_list !== nothing
                    for (s, r) in dyad_list
                        _is_excluded(excluded, s, r, case_s, case_r) && continue
                        push!(all_dyads, (s, r))
                    end
                else
                    for s in senders, r in receivers
                        (exclude_self_loops && s == r) && continue
                        (!directed && s > r) && continue
                        _is_excluded(excluded, s, r, case_s, case_r) && continue
                        push!(all_dyads, (s, r))
                    end
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
                # Sparse request: rejection-sample distinct dyads (uniform over
                # the ordered dyads, or — undirected — over the pairs s ≤ r,
                # by rejecting the draws with s > r)
                n_senders = length(senders)
                n_receivers = length(receivers)
                empty!(sampled)
                while length(sampled) < n_wanted
                    if dyad_list !== nothing
                        s, r = dyad_list[rand(rng, 1:length(dyad_list))]
                    else
                        s = senders[rand(rng, 1:n_senders)]
                        r = receivers[rand(rng, 1:n_receivers)]
                        (exclude_self_loops && s == r) && continue
                        (!directed && s > r) && continue
                    end
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
    compute_statistics(seq::EventSequence, stats; decay=0.0, window=nothing,
                       ties=:error, eventtypes=:error) -> DataFrame

Compute statistics for all events in a sequence (without sampling controls):
one row per event, each statistic read off the network state as it stands
**before** that event. `stats` may be a `StatisticSet` or a vector of
statistics. `decay` (eventnet's halflife model) and `window` (events older than
`current_time − window` stop counting; a `Real` in clock units or a
`Dates.Period` on a calendar clock) are the two memory models of
[`EventNetworkState`](@ref) and are mutually exclusive.

`ties` follows the ecosystem's tied-event contract, as in
[`generate_observations`](@ref): `:error` (default) refuses a sequence with
tied timestamps (ordering them would let the event placed first enter the
statistics of its simultaneous partner); `:breslow` and `:efron` freeze the
state across each tie block, so the tied events do not see each other (the
two give the same statistics — they differ only in a likelihood's
denominator); `:ordered` is sequence order. The policy that applied rides
along as the `"tie_method"` metadata. `eventtypes` is as in
[`generate_observations`](@ref): a multi-type sequence is refused unless
`eventtypes=:pool`.

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
tied = EventSequence([Event(1, 2, 1.0), Event(2, 1, 1.0)]; actors=ActorSet(1:3))
compute_statistics(tied, [Reciprocity()]; ties=:breslow).reciprocity   # [0.0, 0.0]
```
"""
function compute_statistics(events::AbstractVector{<:Event}, args...; kwargs...)
    throw(ArgumentError(_event_vector_hint("compute_statistics")))
end

function compute_statistics(seq::EventSequence, stats::Vector{<:AbstractStatistic};
                            kwargs...)
    return compute_statistics(seq, StatisticSet(stats); kwargs...)
end

function compute_statistics(seq::EventSequence{T}, stats::StatisticSet;
                            decay::Float64=0.0,
                            window::_WindowSpec=nothing,
                            ties::Symbol=:error,
                            eventtypes::Symbol=:error) where T
    check_tie_policy(ties, _REM_TIES_SUPPORTED; model=_REM_TIES_MODEL,
                     reasons=_REM_TIES_REASONS)
    _check_eventtypes(seq, eventtypes, "compute_statistics")
    # The tied-event contract: a tie is refused by default, and
    # the corrections freeze the state across it, exactly as the design does
    blocks = _tie_blocks(seq, 1, length(seq))
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(seq, blocks)
    freeze = ties === :breslow || ties === :efron

    typed = _check_typed_needs(_typed_needs(stats), seq, "compute_statistics")
    state = EventNetworkState(seq; decay=decay, window=window,
                              keep_history=_keeps_history(stats), types=typed)
    stat_names = stats.names
    n = length(seq)
    p = length(stats)

    X = Matrix{Float64}(undef, n, p)
    vals = Vector{Float64}(undef, p)
    senders = Vector{Int}(undef, n)
    receivers = Vector{Int}(undef, n)
    times = Vector{T}(undef, n)

    for block in blocks
        for i in block
            event = seq[i]
            # Advance the state clock (counts decay lazily on read relative
            # to current_time)
            state.current_time = event.time
            compute_all!(vals, stats, state, event.sender, event.receiver)
            X[i, :] .= vals
            senders[i] = event.sender
            receivers[i] = event.receiver
            times[i] = event.time
            # Under a correction the block is absorbed as a whole, below
            freeze || update!(state, event)
        end
        if freeze
            for i in block
                update!(state, seq[i])
            end
        end
    end

    df = DataFrame(sender = senders, receiver = receivers, time = times)
    for (k, sname) in enumerate(stat_names)
        df[!, sname] = X[:, k]
    end
    metadata!(df, "tie_method", string(has_ties ? ties : :none); style=:note)
    return df
end
