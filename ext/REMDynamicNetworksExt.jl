# DynamicNetworks.jl integration for REM.jl, loaded automatically when both
# REM and DynamicNetworks are in the environment (package extension).
#
# Bridges the temporal-network stack to relational-event modeling: the edge
# activation spells of a `DynamicNetwork` become a REM `EventSequence`
# (each spell onset is one event), so dynamic network data can flow into
# `generate_observations`/`fit_rem` without manual wrangling.
module REMDynamicNetworksExt

using REM
using DynamicNetworks
using NetworkCore: ConversionReport, record_drop!, require_observed, n_missing_dyads
using NetworkCore: edges, src, dst, is_directed

"""
    EventSequence(dnet::DynamicNetwork; eventtype=:onset, weight=1.0,
                  include_onset_censored=false, actors=nothing,
                  missing=:error, report=false) -> EventSequence

Convert a `DynamicNetworks.DynamicNetwork`'s edge activation spells into a
relational event sequence: each edge spell contributes one `Event` whose
sender/receiver are the spell's edge endpoints and whose time is the
spell's onset. Events are sorted by time (the `EventSequence` invariant), and
events at the same time by sender, then receiver, so the order of tied events
is fixed by the data. (Fitting refuses tied times by default, `ties=:error`;
under `ties=:ordered` this is the order used.)

Onset-censored spells are skipped by default: their onset is not an
observed event. A spell is onset-censored when it carries the
`onset_censored=true` flag (its recorded onset is the start of the
observation window), when its onset lies **before the start of the
observation window**, and when its onset is **unbounded** — an open-left
spell such as `activate!(d, -Inf, 4.0; edge=(1, 2))`, which is how
networkDynamic data normally encode a tie present before observation. On
axes without infinities (integers, `DateTime`, `Date`) the open bound is the
axis minimum, `DynamicNetworks.unbounded_spell(Time).onset`, and is
recognised as such. All kinds are counted in the `:onset_censored_spells`
entry of the `ConversionReport`. Pass `include_onset_censored=true` to emit
them: a flagged spell at its recorded onset, an open-left or earlier spell
at the start of the observation period. An open-left spell on a network
without a bounded observation start has no time at which to place its
event, and is then an `ArgumentError`, never an event at `-Inf` or at the
axis minimum.

For undirected dynamic networks, edges are stored with `(min, max)`
endpoint ordering, so the smaller vertex ID becomes the event sender.

An edge of the base network with **no spell record** is active throughout
(DynamicNetworks.jl follows R's `active.default = TRUE`): it has no finite onset,
so it contributes no event. Such edges are counted and reported
(`:default_active_edges` in the `ConversionReport`), never dropped silently.

The actor universe is **declared** from the network's vertex set, so vertices
that never carry an edge spell remain in the risk set as isolates (the REM
likelihood is conditional on the risk set, and dropping eligible nonparticipants
changes the estimand). Pass `actors` to override it with a narrower or wider
universe.

# Conversion invariants

Preserved: the actor universe (from the vertex set, isolates included), the
onset of every kept edge spell, and undirected `(min, max)` endpoint ordering.
Base-network edges without a spell record (always active, no onset) emit no
event and are reported.

An event is an instant, so this conversion is lossy by nature: **spell termini**
(an edge dissolution is not an event), terminus censoring, vertex activity
spells (actor presence/composition — the risk set is flat over time), static
and time-varying attributes, and the observation window are all dropped. Pass
`report=true` for `(seq, ::NetworkCore.ConversionReport)` naming them.

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

    window = DynamicNetworks.get_observation_period(dnet)
    start = (window !== nothing && _finite_time(first(window))) ? first(window) : nothing
    events = Event{Time}[]
    n_flagged = 0
    n_open = 0
    n_early = 0
    for ((i, j), spells) in dnet.edge_spells
        for spell in spells
            open_left = !_finite_time(spell.onset)
            early = !open_left && start !== nothing && spell.onset < start
            if open_left || early || spell.onset_censored
                open_left ? (n_open += 1) : early ? (n_early += 1) : (n_flagged += 1)
                include_onset_censored || continue
            end
            t = spell.onset
            if open_left || early
                start === nothing && throw(ArgumentError(
                    "EventSequence(::DynamicNetwork): the spell of edge ($i, $j) " *
                    "begins before observation (onset $(spell.onset)) and the " *
                    "network has no finite observation start, so " *
                    "include_onset_censored=true has no time at which to place " *
                    "its event. Set the window with set_observation_period!, or " *
                    "leave include_onset_censored=false to skip such spells."))
                t = start
            end
            push!(events, Event(Int(i), Int(j), t;
                                eventtype=eventtype, weight=weight))
        end
    end
    # The spells come out of a Dict, so fix the order of simultaneous events by
    # their endpoints; EventSequence then sorts stably by time.
    sort!(events; by=e -> (e.time, e.sender, e.receiver))
    # Base edges with no spell record are active by default (R's
    # active.default): no finite onset, so no event — counted for the report
    directed = is_directed(dnet.network)
    n_default = 0
    for e in edges(dnet.network)
        i, j = Int(src(e)), Int(dst(e))
        key = directed ? (i, j) : minmax(i, j)
        haskey(dnet.edge_spells, key) || (n_default += 1)
    end
    universe = isnothing(actors) ? collect(1:Int(DynamicNetworks.nv(dnet))) : actors
    seq = EventSequence(events; actors=universe)

    rep = ConversionReport(:DynamicNetwork, :EventSequence)
    record_drop!(rep, :spell_termini,
                 "an Event is an instant; spell termini (edge dissolutions) and " *
                 "terminus censoring are not events and are not emitted")
    n_skip = n_flagged + n_open + n_early
    (!include_onset_censored && n_skip > 0) && record_drop!(rep, :onset_censored_spells,
                 "$n_skip onset-censored spell(s) were skipped " *
                 "($n_flagged flagged onset_censored, $n_open open-left with an " *
                 "unbounded onset, $n_early beginning before the observation " *
                 "window): their onset is not an observed event; pass " *
                 "include_onset_censored=true to keep them")
    record_drop!(rep, :vertex_spells,
                 "vertex activity spells are dropped; the declared actor " *
                 "universe is flat over time, not a time-varying risk set")
    record_drop!(rep, :attributes,
                 "static and time-varying vertex/edge/network attributes are " *
                 "not carried; supply them as REM NodeAttributes")
    window === nothing || record_drop!(rep, :observation_period,
                 "the observation window $(window) has no " *
                 "event-sequence counterpart")
    n_default > 0 && record_drop!(rep, :default_active_edges,
                 "$n_default base-network edge(s) have no spell record (active " *
                 "throughout by default) and no finite onset, so they emit no event")
    n_mask = n_missing_dyads(dnet.network)
    n_mask > 0 && record_drop!(rep, :missing_dyads,
                 "$n_mask masked dyad(s) converted at face value under " *
                 "missing=:face; an unobserved dyad becomes a non-event")

    return report ? (seq, rep) : seq
end

# A spell bound that is a time, not the open bound DynamicNetworks uses for
# "before observation": `-Inf` on a float clock, and on axes without
# infinities (integers, `DateTime`, `Date`) the axis minimum that
# `unbounded_spell` defines. Asked of DynamicNetworks, never re-derived here.
_finite_time(t::AbstractFloat) = isfinite(t)
_finite_time(t::T) where {T} = t != DynamicNetworks.unbounded_spell(T).onset

end # module
