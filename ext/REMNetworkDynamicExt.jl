# NetworkDynamic.jl integration for REM.jl, loaded automatically when both
# REM and NetworkDynamic are in the environment (package extension).
#
# Bridges the temporal-network stack to relational-event modeling: the edge
# activation spells of a `DynamicNetwork` become a REM `EventSequence`
# (each spell onset is one event), so dynamic network data can flow into
# `generate_observations`/`fit_rem` without manual wrangling.
module REMNetworkDynamicExt

using REM
using NetworkDynamic
using Networks: ConversionReport, record_drop!, require_observed, n_missing_dyads

"""
    EventSequence(dnet::DynamicNetwork; eventtype=:onset, weight=1.0,
                  include_onset_censored=false, actors=nothing,
                  missing=:error, report=false) -> EventSequence

Convert a `NetworkDynamic.DynamicNetwork`'s edge activation spells into a
relational event sequence: each edge spell contributes one `Event` whose
sender/receiver are the spell's edge endpoints and whose time is the
spell's onset. Events are sorted by time (the `EventSequence` invariant).

Onset-censored spells (`onset_censored=true`) are skipped by default —
their recorded onset is the start of the observation window, not an
observed event; pass `include_onset_censored=true` to keep them.

For undirected dynamic networks, edges are stored with `(min, max)`
endpoint ordering, so the smaller vertex ID becomes the event sender.

The actor universe is **declared** from the network's vertex set, so vertices
that never carry an edge spell remain in the risk set as isolates (the REM
likelihood is conditional on the risk set, and dropping eligible nonparticipants
changes the estimand). Pass `actors` to override it with a narrower or wider
universe.

# Conversion invariants

Preserved: the actor universe (from the vertex set, isolates included), the
onset of every kept edge spell, and undirected `(min, max)` endpoint ordering.

An event is an instant, so this conversion is lossy by nature: **spell termini**
(an edge dissolution is not an event), terminus censoring, vertex activity
spells (actor presence/composition — the risk set is flat over time), static
and time-varying attributes, and the observation window are all dropped. Pass
`report=true` for `(seq, ::Networks.ConversionReport)` naming them.

An `Event` cannot record that a dyad is *unobserved*, so a dynamic network
whose base network carries a missing-dyad mask is **rejected** by default
(`missing=:error`): silently turning an unobserved dyad into a
never-happened non-event would bias the likelihood, which is conditional on
the risk set. Pass `missing=:face` to convert anyway.
"""
function REM.EventSequence(dnet::DynamicNetwork{T, Time};
                           eventtype::Symbol=:onset, weight::Float64=1.0,
                           include_onset_censored::Bool=false,
                           actors=nothing,
                           missing::Symbol=:error,
                           report::Bool=false) where {T, Time}
    require_observed(dnet.network, missing;
                     context="EventSequence(::DynamicNetwork)")

    events = Event{Time}[]
    for ((i, j), spells) in dnet.edge_spells
        for spell in spells
            (spell.onset_censored && !include_onset_censored) && continue
            push!(events, Event(Int(i), Int(j), spell.onset;
                                eventtype=eventtype, weight=weight))
        end
    end
    universe = isnothing(actors) ? collect(1:Int(NetworkDynamic.nv(dnet))) : actors
    seq = EventSequence(events; actors=universe)

    rep = ConversionReport(:DynamicNetwork, :EventSequence)
    record_drop!(rep, :spell_termini,
                 "an Event is an instant; spell termini (edge dissolutions) and " *
                 "terminus censoring are not events and are not emitted")
    !include_onset_censored && record_drop!(rep, :onset_censored_spells,
                 "onset-censored spells were skipped (their onset is the start " *
                 "of the observation window, not an observed event); pass " *
                 "include_onset_censored=true to keep them")
    record_drop!(rep, :vertex_spells,
                 "vertex activity spells are dropped; the declared actor " *
                 "universe is flat over time, not a time-varying risk set")
    record_drop!(rep, :attributes,
                 "static and time-varying vertex/edge/network attributes are " *
                 "not carried; supply them as REM NodeAttributes")
    record_drop!(rep, :observation_period,
                 "the observation window $(dnet.observation_period) has no " *
                 "event-sequence counterpart")
    n_mask = n_missing_dyads(dnet.network)
    n_mask > 0 && record_drop!(rep, :missing_dyads,
                 "$n_mask masked dyad(s) converted at face value under " *
                 "missing=:face; an unobserved dyad becomes a non-event")

    return report ? (seq, rep) : seq
end

end # module
