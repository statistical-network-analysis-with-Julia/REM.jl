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

"""
    EventSequence(dnet::DynamicNetwork; eventtype=:onset, weight=1.0,
                  include_onset_censored=false) -> EventSequence

Convert a `NetworkDynamic.DynamicNetwork`'s edge activation spells into a
relational event sequence: each edge spell contributes one `Event` whose
sender/receiver are the spell's edge endpoints and whose time is the
spell's onset. Events are sorted by time (the `EventSequence` invariant).

Onset-censored spells (`onset_censored=true`) are skipped by default —
their recorded onset is the start of the observation window, not an
observed event; pass `include_onset_censored=true` to keep them.

For undirected dynamic networks, edges are stored with `(min, max)`
endpoint ordering, so the smaller vertex ID becomes the event sender.

Note that only actors appearing in some event are recorded in
`seq.actors`; pass `at_risk` to `generate_observations` to widen the risk
set to all vertices of the network.
"""
function REM.EventSequence(dnet::DynamicNetwork{T, Time};
                           eventtype::Symbol=:onset, weight::Float64=1.0,
                           include_onset_censored::Bool=false) where {T, Time}
    events = Event{Time}[]
    for ((i, j), spells) in dnet.edge_spells
        for spell in spells
            (spell.onset_censored && !include_onset_censored) && continue
            push!(events, Event(Int(i), Int(j), spell.onset;
                                eventtype=eventtype, weight=weight))
        end
    end
    return EventSequence(events)
end

end # module
