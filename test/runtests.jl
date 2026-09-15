using REM
using Test
using DataFrames
using Dates
using Random
# Loading NetworkDynamic activates the REMNetworkDynamicExt package extension
using NetworkDynamic
# Cross-family co-loading: REM must compose with the stats ecosystem
using Statistics
using StatsBase
import StatsAPI
# The other half of the statnet workflow: cross-sections with ERGM, dynamics
# with REM. Imported qualified (not `using`) so the identity assertions in the
# "Namespace" testset below are about the PACKAGES' bindings, not this file's.
import ERGM
import Networks
using Networks: CoefficientTable, check_statsapi
using LinearAlgebra: diag, issymmetric
import LinearAlgebra

# A statistic REM does not know, reading the state's event log (the previous
# event's sender): stands in for Relevent's history-reading statistics in the
# `needs_history` tests below. Defined here because a `struct` must be top level.
struct _LogReader <: AbstractStatistic end
REM.compute(::_LogReader, state::EventNetworkState, s::Int, r::Int) =
    isempty(state.event_history) ? 0.0 : Float64(state.event_history[end][1] == s)
REM.name(::_LogReader) = "log_reader"

# Allocation pin helper: one warm-up call, then the bytes of one `compute`
# (top level so the arguments are locals of a compiled method, not globals)
function _alloc_compute(stat, state, s, r)
    compute(stat, state, s, r)
    return @allocated compute(stat, state, s, r)
end

# Simulate an ordinal relational-event sequence FROM THE MODEL: at each step
# every dyad's hazard is exp(β'x) on the current network state, one dyad is
# drawn, the state absorbs it. Correctly specified for `stats`, so both the
# full-risk-set and the sampled partial likelihoods centre on β — the
# generator behind the calibration and consistency testsets.
function _simulate_rem(rng::AbstractRNG, n_actors::Int, n_events::Int,
                       β::Vector{Float64}, stats)
    dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
    state = EventNetworkState{Float64}(n_actors=n_actors)
    state.actors = Set(1:n_actors)
    ss = StatisticSet(stats)
    x = zeros(length(ss))
    η = zeros(length(dyads))
    events = Event{Float64}[]
    for step in 1:n_events
        for (k, (s, r)) in enumerate(dyads)
            compute_all!(x, ss, state, s, r)
            η[k] = LinearAlgebra.dot(β, x)
        end
        w = exp.(η .- maximum(η)); w ./= sum(w)
        u = rand(rng); acc = 0.0; pick = length(dyads)
        for (k, p) in enumerate(w)
            acc += p
            if u <= acc; pick = k; break; end
        end
        ev = Event(dyads[pick][1], dyads[pick][2], Float64(step))
        push!(events, ev); update!(state, ev)
    end
    return EventSequence(events; actors=1:n_actors)
end

@testset "REM.jl" begin
    @testset "Event and EventSequence" begin
        # Test Event creation
        e1 = Event(1, 2, 1.0)
        @test e1.sender == 1
        @test e1.receiver == 2
        @test e1.time == 1.0
        @test e1.eventtype == :event
        @test e1.weight == 1.0

        e2 = Event(2, 3, 2.0; eventtype=:email, weight=2.0)
        @test e2.eventtype == :email
        @test e2.weight == 2.0

        # Test EventSequence
        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 3, 3.0),
            Event(3, 2, 4.0)
        ]
        seq = EventSequence(events)

        @test length(seq) == 4
        @test seq.n_actors == 3
        @test Set([1, 2, 3]) == seq.actors

        # Test iteration
        times = [e.time for e in seq]
        @test times == [1.0, 2.0, 3.0, 4.0]

        # Test push! maintains sorted order
        push!(seq, Event(2, 3, 2.5))
        @test length(seq) == 5
        @test seq[3].time == 2.5
    end

    @testset "EventSequence indexing" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
        seq = EventSequence(events; actors=ActorSet(1:4))
        # `eachindex`/`firstindex`/`lastindex` agree with `getindex`, so
        # `[f(k) for k in eachindex(seq)]` (per-event risk sets) works
        @test eachindex(seq) == 1:3
        @test firstindex(seq) == 1 && lastindex(seq) == 3
        @test seq[begin] == events[1] && seq[end] == events[3]
        @test [seq[k].sender for k in eachindex(seq)] == [1, 2, 1]
        # `eltype`, so `collect(seq)` is a Vector{Event{T}} that the constructor
        # accepts — the way to declare a universe on a sequence load_events inferred
        @test eltype(seq) == Event{Float64}
        @test collect(seq) isa Vector{Event{Float64}}
        inferred = EventSequence(events)
        @test !inferred.actors_declared
        redeclared = EventSequence(collect(inferred); actors=ActorSet(1:4))
        @test redeclared.actors_declared && redeclared.n_actors == 4
    end

    @testset "Data Loading" begin
        # Create test DataFrame
        df = DataFrame(
            sender = [1, 2, 1, 3],
            receiver = [2, 1, 3, 2],
            time = [1.0, 2.0, 3.0, 4.0]
        )

        seq = load_events(df)
        @test length(seq) == 4
        @test seq.n_actors == 3

        # Test with string actor names
        df_names = DataFrame(
            sender = ["Alice", "Bob", "Alice", "Carol"],
            receiver = ["Bob", "Alice", "Carol", "Bob"],
            time = [1.0, 2.0, 3.0, 4.0]
        )

        seq_names = load_events(df_names; actor_names=true)
        @test length(seq_names) == 4
        @test seq_names.n_actors == 3
    end

    @testset "EventNetworkState" begin
        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 2, 3.0),
            Event(1, 3, 4.0)
        ]
        seq = EventSequence(events)
        state = EventNetworkState(seq)

        # Process events
        for e in seq
            update!(state, e)
        end

        @test get_dyad_count(state, 1, 2) == 2.0
        @test get_dyad_count(state, 2, 1) == 1.0
        @test get_dyad_count(state, 1, 3) == 1.0
        @test get_dyad_count(state, 3, 1) == 0.0

        @test get_out_degree(state, 1) == 3.0
        @test get_in_degree(state, 2) == 2.0

        # `has_edge` is a METHOD of the shared Graphs/Networks generic, not a
        # rival function — so it is exported, and dispatches by state type
        @test REM.has_edge === Networks.has_edge
        @test has_edge(state, 1, 2)
        @test !has_edge(state, 3, 1)
    end

    @testset "Lazy decay matches eager reference" begin
        # Counts are stored as (value, last_update_time) and decayed on
        # read; this must agree with the eager reference that multiplies
        # every nonzero count by exp(-decay·Δt) at every event.
        rng = Random.Xoshiro(99)
        n = 12
        decay = halflife_to_decay(7.0)

        events = Event{Float64}[]
        t = 0.0
        while length(events) < 300
            s, r = rand(rng, 1:n), rand(rng, 1:n)
            s == r && continue
            t += 2 * rand(rng)
            push!(events, Event(s, r, t; weight=0.5 + rand(rng)))
        end
        seq = EventSequence(events)
        state = EventNetworkState(seq; decay=decay)

        # Eager reference implementation
        ref_dyad = Dict{Tuple{Int,Int}, Float64}()
        ref_und = Dict{Tuple{Int,Int}, Float64}()
        ref_out = Dict{Int, Float64}()
        ref_in = Dict{Int, Float64}()
        t_ref = 0.0

        for e in seq
            f = exp(-decay * (e.time - t_ref))
            for d in (ref_dyad, ref_und)
                map!(v -> v * f, values(d))
            end
            for d in (ref_out, ref_in)
                map!(v -> v * f, values(d))
            end
            t_ref = e.time

            ref_dyad[(e.sender, e.receiver)] =
                get(ref_dyad, (e.sender, e.receiver), 0.0) + e.weight
            ref_und[minmax(e.sender, e.receiver)] =
                get(ref_und, minmax(e.sender, e.receiver), 0.0) + e.weight
            ref_out[e.sender] = get(ref_out, e.sender, 0.0) + e.weight
            ref_in[e.receiver] = get(ref_in, e.receiver, 0.0) + e.weight

            update!(state, e)

            for s in 1:n, r in 1:n
                s == r && continue
                @test get_dyad_count(state, s, r) ≈
                      get(ref_dyad, (s, r), 0.0) atol = 1e-12
                @test get_undirected_count(state, s, r) ≈
                      get(ref_und, minmax(s, r), 0.0) atol = 1e-12
            end
            for a in 1:n
                @test get_out_degree(state, a) ≈ get(ref_out, a, 0.0) atol = 1e-12
                @test get_in_degree(state, a) ≈ get(ref_in, a, 0.0) atol = 1e-12
            end
        end

        # Reading at a later clock time decays further, without an update
        f = exp(-decay * 5.0)
        expected = get_dyad_count(state, events[end].sender, events[end].receiver) * f
        state.current_time = t + 5.0
        @test get_dyad_count(state, events[end].sender, events[end].receiver) ≈
              expected atol = 1e-12
    end

    @testset "Calendar timelines" begin
        # Decay with DateTime timestamps (1 hour halflife)
        dt_events = [
            Event(1, 2, DateTime(2024, 1, 1, 0, 0, 0)),
            Event(1, 2, DateTime(2024, 1, 1, 1, 0, 0))
        ]
        seq_dt = EventSequence(dt_events)
        state_dt = EventNetworkState(seq_dt; decay=halflife_to_decay(3600.0))

        update!(state_dt, seq_dt[1])
        update!(state_dt, seq_dt[2])
        @test get_dyad_count(state_dt, 1, 2) ≈ 1.5 atol=1e-8

        # Recency with Date timestamps (difference in seconds)
        date_events = [
            Event(1, 2, Date(2024, 1, 1)),
            Event(2, 1, Date(2024, 1, 2))
        ]
        seq_date = EventSequence(date_events)
        state_date = EventNetworkState(seq_date)
        update!(state_date, seq_date[1])
        update!(state_date, seq_date[2])
        state_date.current_time = Date(2024, 1, 3)

        recency = RecencyStatistic()
        @test compute(recency, state_date, 1, 2) ≈ 1 / (2 * 86400) atol=1e-12
    end

    @testset "Dyad Statistics" begin
        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 2, 3.0)
        ]
        seq = EventSequence(events)
        state = EventNetworkState(seq)

        # Process first two events
        update!(state, seq[1])
        update!(state, seq[2])
        state.current_time = seq[3].time

        # Test Repetition
        rep = Repetition()
        @test compute(rep, state, 1, 2) == 1.0  # 1→2 happened once
        @test compute(rep, state, 2, 1) == 1.0  # 2→1 happened once
        @test compute(rep, state, 1, 3) == 0.0  # 1→3 never happened

        # Test undirected repetition
        rep_undir = Repetition(directed=false)
        @test compute(rep_undir, state, 1, 2) == 2.0  # 1↔2 has 2 events

        # Test Reciprocity
        recip = Reciprocity()
        @test compute(recip, state, 1, 2) == 1.0  # 2→1 exists
        @test compute(recip, state, 2, 1) == 1.0  # 1→2 exists
        @test compute(recip, state, 1, 3) == 0.0  # 3→1 doesn't exist
    end

    @testset "Degree Statistics" begin
        events = [
            Event(1, 2, 1.0),
            Event(1, 3, 2.0),
            Event(2, 3, 3.0)
        ]
        seq = EventSequence(events)
        state = EventNetworkState(seq)

        for e in seq
            update!(state, e)
        end

        # Test sender activity
        sa = SenderActivity()
        @test compute(sa, state, 1, 4) == 2.0  # Actor 1 sent 2 events
        @test compute(sa, state, 2, 4) == 1.0  # Actor 2 sent 1 event

        # Test receiver popularity
        rp = ReceiverPopularity()
        @test compute(rp, state, 4, 3) == 2.0  # Actor 3 received 2 events
        @test compute(rp, state, 4, 2) == 1.0  # Actor 2 received 1 event
    end

    @testset "Triangle Statistics" begin
        # Create a network: 1→2, 2→3
        events = [
            Event(1, 2, 1.0),
            Event(2, 3, 2.0)
        ]
        seq = EventSequence(events)
        state = EventNetworkState(seq)

        for e in seq
            update!(state, e)
        end

        # Test transitive closure
        # For 1→3: we need k such that 1→k and k→3
        # 1→2 exists, 2→3 exists, so k=2 works (min(1, 1) = 1 weighted, 1 counted)
        tc = TransitiveClosure()
        @test compute(tc, state, 1, 3) == 1.0
        @test compute(TransitiveClosure(weighted=false), state, 1, 3) == 1.0

        # For 3→1: need k such that 3→k and k→1 - doesn't exist
        @test compute(tc, state, 3, 1) == 0.0
    end

    @testset "Node Statistics" begin
        # Create node attribute
        gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "M", 3 => "F"), "Unknown")

        # Test AttributeMatch
        match = AttributeMatch(gender)
        state = EventNetworkState{Float64}()
        @test compute(match, state, 1, 2) == 1.0  # Both M
        @test compute(match, state, 1, 3) == 0.0  # M vs F

        # Test numeric attribute
        age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0, 3 => 25.0), 0.0)
        diff = NodeDifference(age)
        @test compute(diff, state, 1, 2) == -5.0  # 25 - 30
        @test compute(diff, state, 2, 1) == 5.0   # 30 - 25

        diff_abs = NodeDifference(age; absolute=true)
        @test compute(diff_abs, state, 1, 2) == 5.0
    end

    @testset "Case-Control Sampling" begin
        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 3, 3.0),
            Event(3, 2, 4.0)
        ]
        seq = EventSequence(events)

        stats = [Repetition(), Reciprocity()]
        sampler = CaseControlSampler(n_controls=5, seed=42)

        obs = generate_observations(seq, stats, sampler)

        # Should have 4 cases + 4*5 controls = 24 observations
        @test nrow(obs) == 24

        # Check columns exist
        @test "is_event" in names(obs)
        @test "stratum" in names(obs)
        @test "repetition" in names(obs)
        @test "reciprocity" in names(obs)

        # Check we have correct number of cases
        @test sum(obs.is_event) == 4

        # Risk-set bookkeeping travels with the observations
        @test "risk_set_size" in names(obs)
        @test "sampling_prob" in names(obs)
        # 3 actors, no self-loops → 6 dyads; 5 controls out of the 5 non-case
        # dyads (the full risk set)
        @test all(obs.risk_set_size .== 6)
        @test all(obs.sampling_prob .== 1.0)
    end

    @testset "Actor universe" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]

        # Inferred universe: observed participants only (the fallback)
        seq = EventSequence(events)
        @test seq.actors == Set([1, 2, 3])
        @test !seq.actors_declared

        # Declared universe: isolates and noncontiguous IDs are kept
        seq_d = EventSequence(events; actors=ActorSet([1, 2, 3, 7, 42]))
        @test seq_d.actors_declared
        @test seq_d.actors == Set([1, 2, 3, 7, 42])
        @test seq_d.n_actors == 5

        # Any actor-universe spelling works
        @test EventSequence(events; actors=Set([1, 2, 3])).actors == Set([1, 2, 3])
        @test EventSequence(events; actors=[1, 2, 3, 9]).actors == Set([1, 2, 3, 9])
        @test EventSequence(events; actors=1:4).actors == Set([1, 2, 3, 4])
        @test_throws ArgumentError EventSequence(events; actors="everyone")

        # A declared universe must cover every event endpoint
        @test_throws ArgumentError EventSequence(events; actors=[1, 2])

        # ... and keeps covering them under push!
        @test_throws ArgumentError push!(seq_d, Event(1, 99, 4.0))
        push!(seq_d, Event(7, 42, 4.0))          # isolates may become active
        @test length(seq_d) == 4

        # Empty sequences can declare a universe up front
        empty_seq = EventSequence{Float64}(; actors=[1, 2, 3])
        @test empty_seq.n_actors == 3
        @test empty_seq.actors_declared
        @test length(EventSequence{Float64}()) == 0

        # ActorSet conveniences
        @test length(ActorSet(1:5)) == 5
        @test 3 in ActorSet([1, 3, 5])
        @test_throws ArgumentError ActorSet([1, 1, 2])
    end

    @testset "Risk sets in generate_observations/fit_rem" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
        stats = [Repetition(), Reciprocity()]

        # THE BUG: a case outside its risk set was silently accepted, its
        # controls drawn from a different actor universe. It now throws,
        # before any fitting happens.
        seq = EventSequence(events)
        sampler = CaseControlSampler(n_controls=2, seed=1)
        @test_throws ArgumentError generate_observations(seq, stats, sampler;
                                                         at_risk=Set([1, 2]))
        @test_throws ArgumentError fit_rem(seq, stats; n_controls=2, at_risk=Set([1, 2]))

        # Isolates: an actor with zero observed events stays in the risk set
        seq_iso = EventSequence(events; actors=ActorSet([1, 2, 3, 7]))
        obs = generate_observations(seq_iso, stats, CaseControlSampler(n_controls=50, seed=3))
        @test all(obs.risk_set_size .== 12)      # 4 actors, no self-loops
        @test all(obs.sampling_prob .== 1.0)     # full risk set enumerated
        @test 7 in obs.sender                    # the isolate is sampled as a control
        @test 7 in obs.receiver
        # Every stratum: one case, and the case is a real event dyad
        for st in unique(obs.stratum)
            sub = obs[obs.stratum .== st, :]
            @test sum(sub.is_event) == 1
            @test nrow(sub) == 12                # case + 11 controls
        end

        # Noncontiguous actor IDs
        ev_nc = [Event(10, 20, 1.0), Event(20, 10, 2.0), Event(10, 30, 3.0)]
        seq_nc = EventSequence(ev_nc; actors=[10, 20, 30, 99])
        obs_nc = generate_observations(seq_nc, stats, CaseControlSampler(n_controls=50, seed=4))
        @test all(obs_nc.risk_set_size .== 12)
        @test Set(obs_nc.sender) ⊆ Set([10, 20, 30, 99])
        @test 99 in obs_nc.sender

        # Receiver-only actors are eligible senders in the risk set
        ev_ro = [Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 4, 3.0)]
        seq_ro = EventSequence(ev_ro; actors=1:4)
        obs_ro = generate_observations(seq_ro, stats, CaseControlSampler(n_controls=50, seed=5))
        @test 4 in obs_ro.sender     # actor 4 only ever receives, but may send

        # Static risk set given directly to fit_rem (universe wider than the
        # sequence's own actors is fine as long as the cases belong to it)
        result = fit_rem(seq_iso, stats; n_controls=11, seed=6, at_risk=[1, 2, 3, 7])
        @test result isa REMResult
        @test result.strata == [1, 2, 3]
        @test result.risk_set_sizes == [12, 12, 12]
        @test result.sampling_probs == [1.0, 1.0, 1.0]
        @test occursin("Risk-set size: 12 dyads", sprint(show, result))

        # `riskset` is an alias for `at_risk`; passing both is an error
        result2 = fit_rem(seq_iso, stats; n_controls=11, seed=6, riskset=[1, 2, 3, 7])
        @test result2.coefficients == result.coefficients
        @test_throws ArgumentError fit_rem(seq_iso, stats; at_risk=[1, 2, 3, 7],
                                           riskset=[1, 2, 3, 7])

        # Sampled (not enumerated) controls: the sampling probability is recorded
        obs_s = generate_observations(seq_iso, stats,
                                      CaseControlSampler(n_controls=3, seed=7))
        @test all(obs_s.risk_set_size .== 12)
        @test all(obs_s.sampling_prob .≈ 3 / 11)
        @test sum(.!obs_s.is_event) == 9         # 3 events × 3 controls

        # Unsupported specifications
        @test_throws ArgumentError generate_observations(seq_iso, stats, sampler;
                                                         at_risk="everyone")
    end

    @testset "Time-varying risk sets" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
        stats = [Repetition()]
        seq = EventSequence(events; actors=1:4)

        # Per-event risk sets: actor 3 joins only at event 3, actor 4 at event 2
        per_event = [Set([1, 2]), Set([1, 2, 4]), Set([1, 2, 3, 4])]
        obs = generate_observations(seq, stats,
                                    CaseControlSampler(n_controls=20, seed=11);
                                    at_risk=per_event)
        sizes = [first(obs[obs.stratum .== st, :risk_set_size]) for st in 1:3]
        @test sizes == [2, 6, 12]                # n(n-1) dyads per event
        # Actor 3 must never appear before it joins the universe
        early = obs[obs.stratum .<= 2, :]
        @test !(3 in early.sender) && !(3 in early.receiver)

        # A case outside its own (time-varying) risk set throws
        bad = [Set([1, 2]), Set([1, 2]), Set([1, 2])]   # event 3 is 1 → 3
        @test_throws ArgumentError generate_observations(seq, stats,
                                                         CaseControlSampler(n_controls=2);
                                                         at_risk=bad)

        # Wrong number of per-event risk sets
        @test_throws ArgumentError generate_observations(seq, stats,
                                                         CaseControlSampler(n_controls=2);
                                                         at_risk=[Set([1, 2, 3])])

        # Callback risk set: (event_index, state) -> RiskSet, evaluated against
        # the live network state
        cb = (i, state) -> RiskSet(i, sort!(collect(state.actors)),
                                   sort!(collect(state.actors)))
        obs_cb = generate_observations(seq, stats,
                                       CaseControlSampler(n_controls=20, seed=12);
                                       at_risk=cb)
        @test first(obs_cb[obs_cb.stratum .== 1, :risk_set_size]) == 12  # all 4 actors
        # A callback returning a plain actor collection works too
        obs_cb2 = generate_observations(seq, stats,
                                        CaseControlSampler(n_controls=20, seed=12);
                                        at_risk=(i, state) -> [1, 2, 3])
        @test all(obs_cb2.risk_set_size .== 6)
        # Callbacks with the wrong signature are rejected, not ignored
        @test_throws ArgumentError generate_observations(seq, stats,
                                                         CaseControlSampler(n_controls=2);
                                                         at_risk=(i) -> [1, 2, 3])

        # Asymmetric (sender ≠ receiver) risk sets
        rs = RiskSet(0, [1, 2], [1, 2, 3])
        obs_a = generate_observations(seq, stats,
                                      CaseControlSampler(n_controls=20, seed=13);
                                      at_risk=rs)
        @test all(obs_a.risk_set_size .== 4)     # 2×3 dyads − 2 self-loops
        @test Set(obs_a.sender) ⊆ Set([1, 2])

        # A risk set that leaves no valid control throws
        @test_throws ArgumentError generate_observations(
            EventSequence([Event(1, 2, 1.0)]; actors=[1, 2]), stats,
            CaseControlSampler(n_controls=2); at_risk=RiskSet(0, [1], [2]))
    end

    @testset "Utility Functions" begin
        # Test halflife conversion
        halflife = 10.0
        decay = halflife_to_decay(halflife)
        @test decay ≈ log(2) / 10.0

        # Round-trip
        @test decay_to_halflife(decay) ≈ halflife
    end

    @testset "Integration Test" begin
        # Full pipeline test
        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 2, 3.0),
            Event(2, 3, 4.0),
            Event(3, 1, 5.0),
            Event(1, 3, 6.0)
        ]
        seq = EventSequence(events)

        stats = [
            Repetition(),
            Reciprocity(),
            SenderActivity(),
            ReceiverPopularity()
        ]

        # Generate observations: only 5 distinct control dyads exist among
        # 3 actors, so the request for 10 is capped (with a warning) and
        # the full risk set is used
        sampler = CaseControlSampler(n_controls=10, seed=123)
        obs = @test_logs (:warn, r"only 5 distinct dyads") match_mode = :any begin
            generate_observations(seq, stats, sampler)
        end

        @test nrow(obs) == 6 * 6  # 6 events * (1 case + 5 distinct controls)

        # No stratum may contain duplicate dyads
        for st in unique(obs.stratum)
            sub = obs[obs.stratum .== st, :]
            @test allunique(collect(zip(sub.sender, sub.receiver)))
        end

        # Fit end-to-end
        result = fit_rem(obs, ["repetition", "reciprocity"])
        @test result isa REMResult
        @test length(coef(result)) == 2
        @test all(isfinite, result.log_likelihood)

        # Compute statistics without sampling
        stats_df = compute_statistics(seq, stats)
        @test nrow(stats_df) == 6
    end

    @testset "Sampling without replacement" begin
        # Larger actor set: rejection sampling path; controls must be
        # distinct within each stratum
        events = [Event(i, mod1(i + 1, 10), Float64(i)) for i in 1:30]
        seq = EventSequence(events)
        sampler = CaseControlSampler(n_controls=20, seed=7)
        obs = generate_observations(seq, [Repetition()], sampler)

        @test nrow(obs) == 30 * 21
        for st in unique(obs.stratum)
            sub = obs[obs.stratum .== st, :]
            @test allunique(collect(zip(sub.sender, sub.receiver)))
            @test sum(sub.is_event) == 1
        end

        # Reproducible with the same seed, without touching the global RNG
        Random.seed!(1234)
        marker1 = rand()
        Random.seed!(1234)
        obs2 = generate_observations(seq, [Repetition()], sampler)
        marker2 = rand()
        @test obs == obs2 skip = false
        @test marker1 == marker2  # global RNG stream untouched by sampler seed
    end

    @testset "Neighbor sets are incremental" begin
        events = [Event(1, 2, 1.0), Event(2, 3, 2.0), Event(1, 3, 3.0),
                  Event(3, 1, 4.0)]
        seq = EventSequence(events)
        state = EventNetworkState(seq)
        for e in seq
            update!(state, e)
        end

        @test get_out_neighbors(state, 1) == Set([2, 3])
        @test get_in_neighbors(state, 3) == Set([1, 2])
        @test REM.get_common_receivers(state, 1, 2) == Set([3])
        @test REM.get_common_senders(state, 2, 3) == Set([1])

        reset!(state)
        @test isempty(get_out_neighbors(state, 1))
    end

    @testset "Clogit analytic check" begin
        # Three 1:1 strata with covariate differences (case − control) of
        # +1, +1, −1. The conditional-logit MLE solves 2 − 3σ(β) = 0,
        # i.e. β = log(2).
        obs = DataFrame(
            event_index = [1, 1, 2, 2, 3, 3],
            sender = [1, 2, 1, 2, 1, 2],
            receiver = [2, 1, 2, 1, 2, 1],
            x = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            is_event = [true, false, true, false, true, false],
            stratum = [1, 1, 2, 2, 3, 3]
        )

        result = fit_rem(obs, ["x"])
        @test result.converged
        @test result.coefficients[1] ≈ log(2) atol = 1e-6

        # Log-likelihood at the MLE: 2·log σ(β) + log σ(−β), β = log 2
        @test result.log_likelihood ≈ 2 * log(2 / 3) + log(1 / 3) atol = 1e-8

        # Input validation
        @test_throws ArgumentError fit_rem(obs, ["nonexistent"])
        bad = copy(obs)
        bad.is_event = [true, true, true, false, true, false]
        @test_throws ArgumentError fit_rem(bad, ["x"])
        # ... including strata with zero cases
        bad0 = copy(obs)
        bad0.is_event = [true, false, true, false, false, false]
        @test_throws ArgumentError fit_rem(bad0, ["x"])
    end

    @testset "Golden: coefficients vs R survival::clogit" begin
        # The analytic check above pins the estimator on a 6-row toy design.
        # This pins it — and the STATISTICS — against an independent
        # implementation of the same likelihood: R's survival::clogit, on a
        # design matrix recomputed from the raw edgelist in plain R, not
        # exported from Julia. See test/fixtures/r/rem_clogit.R.
        #
        # The risk set is enumerated in full on both sides (all n(n−1) = 90
        # ordered dyads per event), so there are no sampled controls to
        # reconcile across two RNGs and the comparison is exact, not
        # distributional.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_clogit.toml"))
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end

        n = Int(g.values["n_actors"])
        times = Float64.(g.values["input_time"])
        senders = Int.(g.values["input_sender"])
        receivers = Int.(g.values["input_receiver"])
        events = [Event(senders[i], receivers[i], times[i]) for i in eachindex(times)]
        seq = EventSequence(events; actors=ActorSet(collect(1:n)))

        # `weighted=false` is the COUNT of closing third parties (adjacency),
        # which is what the R script computes; the 0.2 default is eventnet's
        # min-weighted form, pinned by rem_eventnet.toml below
        stats = AbstractStatistic[Repetition(), Reciprocity(), SenderActivity(),
                                  ReceiverPopularity(), TransitiveClosure(weighted=false)]
        @test [name(s) for s in stats] == g.values["statistic_names"]

        # n_controls == the number of non-case dyads => the full risk set is
        # enumerated, deterministically, with no sampling.
        max_controls = n * (n - 1) - 1
        result = fit_rem(seq, stats; n_controls=max_controls, tol=1e-12)
        @test result.converged
        @test result.n_events == Int(g.values["n_strata"])
        @test all(==(Int(g.values["risk_set_size"])), result.risk_set_sizes)
        @test all(==(1.0), result.sampling_probs)   # nothing was sampled away

        @test report("coefficients", result.coefficients)
        @test report("std_errors", result.std_errors)
        @test report("loglik", result.log_likelihood)
        # The covariance the fit carries is the one the standard errors are the
        # diagonal of (StatsAPI `vcov`), and it is R's too
        @test report("std_errors", sqrt.(diag(vcov(result))))
        @test loglikelihood(result) == result.log_likelihood
        # The estimator is the shared Networks.newton_fit: it reports its own
        # iteration count and convergence, not a REM-local loop's
        @test result.iterations > 0
        # ... and nothing is singular or separated on this well-posed design
        @test !result.singular && isempty(result.singular_suspects) && isempty(result.separated)
        @test Networks.is_exact(result)

        # se=:sandwich has an R counterpart too: `clogit(... + strata(stratum)
        # + cluster(stratum), method="breslow")` — the stratum-clustered
        # robust variance (one event per stratum, so Breslow IS the exact
        # likelihood there; without `cluster(stratum)` R's robust variance is
        # the per-row dfbeta sandwich, a different number). Same MLE, pinned
        # to 1e-8 (observed agreement 4e-16).
        robust = fit_rem(seq, stats; n_controls=max_controls, tol=1e-12, se=:sandwich)
        @test coef(robust) == coef(result)
        @test Networks.se_method(robust) === :sandwich
        @test report("robust_std_errors", stderror(robust))
        @test report("robust_std_errors", sqrt.(diag(vcov(robust))))
        @test report("coefficients", coef(robust))
    end

    @testset "Golden: tie corrections vs R survival::coxph (Breslow, Efron)" begin
        # The fixture above has no tied event times. This one is ABOUT them
        # (issue REM#2, review finding 12): a continuous-time sequence observed
        # on a coarse clock, so 25 of its 53 timestamps carry ties, up to 4 deep.
        #
        # REM's conditional-logit partial likelihood IS a Cox partial likelihood
        # with one stratum per event, so `ties=:breslow` / `ties=:efron` must be
        # the SAME numbers as `coxph(..., ties="breslow"/"efron")` — not "a
        # Breslow-like correction of our own". The R script rebuilds the whole
        # counting-process design from the raw edgelist in plain R (statistics
        # included), and enumerates the risk set in FULL on both sides, so the
        # comparison is exact rather than distributional.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_ties.toml"))
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end

        n = Int(g.values["n_actors"])
        times = Float64.(g.values["input_time"])
        senders = Int.(g.values["input_sender"])
        receivers = Int.(g.values["input_receiver"])
        events = [Event(senders[i], receivers[i], times[i]) for i in eachindex(times)]
        seq = EventSequence(events; actors=ActorSet(collect(1:n)))

        # The data really is tied — otherwise this fixture would be checking that
        # a tie correction is a no-op, which is a different (also tested) claim
        @test !allunique(times)
        @test length(unique(times)) == Int(g.values["n_blocks"])
        @test Int(g.values["n_tied_blocks"]) == 25
        @test Int(g.values["max_block_size"]) == 4

        stats = AbstractStatistic[Repetition(), Reciprocity(), SenderActivity()]
        @test [name(s) for s in stats] == g.values["statistic_names"]

        # Full risk set on the Julia side too (n_controls = every non-case dyad)
        max_controls = n * (n - 1) - 1

        for method in ("breslow", "efron")
            fit = fit_rem(seq, stats; n_controls=max_controls,
                          ties=Symbol(method), tol=1e-12)
            @test fit.converged
            @test Networks.tie_method(fit) === Symbol(method)
            @test all(==(Int(g.values["risk_set_size"])), fit.risk_set_sizes)

            @test report("$(method)_coefficients", fit.coefficients)
            @test report("$(method)_std_errors", fit.std_errors)
            @test report("$(method)_loglik", fit.log_likelihood)
            @test report("$(method)_std_errors", sqrt.(diag(vcov(fit))))
        end

        # The two corrections are not the same correction (Efron is the better
        # approximation), and neither is the uncorrected sort: if they coincided
        # here, the fixture above would be checking nothing.
        b = fit_rem(seq, stats; n_controls=max_controls, ties=:breslow, tol=1e-12)
        e = fit_rem(seq, stats; n_controls=max_controls, ties=:efron, tol=1e-12)
        o = fit_rem(seq, stats; n_controls=max_controls, ties=:ordered, tol=1e-12)
        @test coef(b) != coef(e) && coef(o) != coef(e) && coef(o) != coef(b)
        # Breslow shrinks toward zero relative to Efron — the classical result,
        # and visible here on the two positive effects
        @test abs(coef(b)[1]) < abs(coef(e)[1])
        @test abs(coef(b)[2]) < abs(coef(e)[2])
    end

    @testset "P-values survive extreme z (no cdf underflow)" begin
        # 1:1 strata with case−control covariate difference +1 in `a`
        # strata and −1 in `b` strata: the clogit MLE is β = log(a/b) with
        # information (a+b)σ(1−σ), σ = a/(a+b). a = 3756, b = 939 gives
        # z ≈ 38, where the naive 2(1 − Φ(z)) formula (dead for |z| ≳ 8.3)
        # underflows to exactly 0 and the true p ≈ 1e-316 is a subnormal.
        # The ecosystem's `Networks.z_pvalues` (erfc-based) floors a finite
        # statistic's p at `floatmin(Float64)` — never 0.0, never a subnormal
        # that prints as one — so this is what REM reports.
        a, b = 3756, 939
        n_strata = a + b
        x = Float64[]
        is_event = Bool[]
        stratum = Int[]
        for k in 1:n_strata
            push!(x, k <= a ? 1.0 : 0.0)
            push!(x, k <= a ? 0.0 : 1.0)
            append!(is_event, [true, false])
            append!(stratum, [k, k])
        end
        obs = DataFrame(
            event_index = stratum, sender = fill(1, 2n_strata),
            receiver = fill(2, 2n_strata), x = x,
            is_event = is_event, stratum = stratum
        )

        result = fit_rem(obs, ["x"])
        @test result.converged
        @test result.coefficients[1] ≈ log(a / b) atol = 1e-6
        @test abs(result.z_values[1]) > 37.5
        p = result.p_values[1]
        @test p > 0            # the naive 2*(1 - cdf) formula returns exactly 0 here
        @test p < 1e-300
        @test p == floatmin(Float64)   # the shared floor: a finite z never has p == 0
        @test !issubnormal(p)
        # The p-values ARE the ecosystem's `Networks.z_pvalues` (panel 2026-09,
        # item 13): bit-for-bit, not a local 2·ccdf copy that happens to agree
        @test result.p_values == Networks.z_pvalues(result.z_values)
        zp = Networks.z_pvalues(result.coefficients, result.std_errors)
        @test result.z_values == zp.z && result.p_values == zp.p
    end

    @testset "StatsAPI co-loading" begin
        # `using REM, Statistics, StatsBase` must not create export
        # collisions: REM's coef/stderror/coeftable are StatsAPI methods,
        # the same generics StatsBase re-exports
        @test coef === StatsAPI.coef === StatsBase.coef
        @test stderror === StatsAPI.stderror === StatsBase.stderror
        @test coeftable === StatsAPI.coeftable === Networks.coeftable
        # ... all ten verbs of the ecosystem surface are the StatsAPI bindings
        @test vcov === StatsAPI.vcov === StatsBase.vcov
        @test confint === StatsAPI.confint === StatsBase.confint
        @test loglikelihood === StatsAPI.loglikelihood === StatsBase.loglikelihood
        @test nobs === StatsAPI.nobs === StatsBase.nobs
        @test dof === StatsAPI.dof === StatsBase.dof
        @test aic === StatsAPI.aic === StatsBase.aic
        @test bic === StatsAPI.bic === StatsBase.bic

        events = [
            Event(1, 2, 1.0),
            Event(2, 1, 2.0),
            Event(1, 2, 3.0),
            Event(2, 3, 4.0),
            Event(3, 1, 5.0),
            Event(1, 3, 6.0)
        ]
        seq = EventSequence(events)
        fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=5, seed=42)

        # The unqualified generics dispatch to the REMResult methods
        @test coef(fit) == fit.coefficients
        @test stderror(fit) == fit.std_errors
        @test StatsBase.coef(fit) == fit.coefficients
        # ... and Statistics still works alongside
        @test mean(coef(fit)) ≈ sum(fit.coefficients) / 2

        # The full StatsAPI surface (panel 2026-09, item 15), pinned by the ONE
        # ecosystem checker — every verb present AND mutually consistent
        @test check_statsapi(fit; strict=true) !== nothing
        # `coeftable` returns the shared inspectable table, not a DataFrame
        tbl = coeftable(fit)
        @test tbl isa CoefficientTable
        @test tbl.names == fit.stat_names
        @test tbl.estimates == fit.coefficients
        @test tbl.std_errors == fit.std_errors
        @test tbl.z_values == fit.z_values && tbl.p_values == fit.p_values
        @test tbl["reciprocity"].estimate == coef(fit)[2]
        @test length(tbl) == 2
        # ... and prints exactly the table `show(fit)` prints
        @test occursin("Pr(>|z|)", sprint(show, tbl))
        # The remaining verbs read straight off the fit
        @test vcov(fit) == fit.var_cov
        @test size(vcov(fit)) == (2, 2)
        @test stderror(fit) ≈ sqrt.(diag(vcov(fit)))
        @test loglikelihood(fit) == fit.log_likelihood
        @test nobs(fit) == fit.n_events == 6      # events, not case-control rows
        @test dof(fit) == 2
        @test aic(fit) ≈ -2 * fit.log_likelihood + 2 * 2
        @test bic(fit) ≈ -2 * fit.log_likelihood + 2 * log(6)
        ci = confint(fit)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .< coef(fit) .< ci[:, 2])
        @test ci[:, 2] .- ci[:, 1] ≈ 2 * 1.959963984540054 .* stderror(fit)
        ci90 = confint(fit; level=0.9)
        @test all(ci90[:, 2] .- ci90[:, 1] .< ci[:, 2] .- ci[:, 1])
        @test_throws ArgumentError confint(fit; level=1.5)
        # A fit from the DataFrame method carries the same surface
        obs = generate_observations(seq, [Repetition(), Reciprocity()],
                                    CaseControlSampler(n_controls=5, seed=42))
        fit_df = fit_rem(obs, ["repetition", "reciprocity"])
        @test check_statsapi(fit_df; strict=true) !== nothing
        @test coef(fit_df) == coef(fit) && vcov(fit_df) == vcov(fit)

        # show() renders the shared ecosystem coefficient table
        # (Networks.jl print_coeftable: z / Pr(>|z|) columns + signif codes)
        out = sprint(show, fit)
        @test occursin("Relational Event Model Results", out)
        @test occursin("repetition", out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)
    end

    @testset "Tuple-backed StatisticSet" begin
        stats = [Repetition(), Reciprocity(), SenderActivity()]
        ss = StatisticSet(stats)

        # Tuple storage: the concrete statistic types are in the type
        # parameter, so compute_all is dispatch-free in the inner loop
        @test ss.statistics isa Tuple{Repetition, Reciprocity, SenderActivity}
        @test length(ss) == 3
        @test ss[2] isa Reciprocity
        @test collect(ss) == collect(ss.statistics)
        @test ss.names == [REM.name(s) for s in stats]

        # Construction directly from a tuple, and input validation
        @test StatisticSet((Repetition(), Reciprocity())).names ==
              ["repetition", "reciprocity"]
        @test_throws ArgumentError StatisticSet((Repetition(), 1.0))

        # compute_all / compute_all! agree with the Vector path
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)]
        seq = EventSequence(events)
        state = EventNetworkState(seq)
        update!(state, seq[1])
        update!(state, seq[2])
        state.current_time = seq[3].time

        expected = compute_all(stats, state, 1, 2)
        @test compute_all(ss, state, 1, 2) == expected
        dest = zeros(3)
        @test compute_all!(dest, ss, state, 1, 2) === dest
        @test dest == expected

        # The estimation entry points accept a StatisticSet directly and
        # match the Vector-based results exactly
        big_events = [Event(mod1(i, 5), mod1(i + 2, 5), Float64(i)) for i in 1:40]
        big_seq = EventSequence(big_events)
        obs_vec = generate_observations(big_seq, stats,
                                        CaseControlSampler(n_controls=3, seed=11))
        obs_set = generate_observations(big_seq, ss,
                                        CaseControlSampler(n_controls=3, seed=11))
        @test obs_vec == obs_set
        @test compute_statistics(big_seq, stats) == compute_statistics(big_seq, ss)

        fit_vec = fit_rem(big_seq, stats; n_controls=3, seed=11)
        fit_set = fit_rem(big_seq, ss; n_controls=3, seed=11)
        @test coef(fit_vec) == coef(fit_set)
        @test stderror(fit_vec) == stderror(fit_set)
    end

    @testset "NetworkDynamic extension: EventSequence(::DynamicNetwork)" begin
        @test Base.get_extension(REM, :REMNetworkDynamicExt) !== nothing

        # Directed dynamic network: spell onsets become events
        dnet = DynamicNetwork(4; observation_start=0.0, observation_end=10.0,
                              directed=true)
        activate!(dnet, 1.0, 3.0; edge=(1, 2))
        activate!(dnet, 2.0, 5.0; edge=(2, 3))
        activate!(dnet, 4.0, 6.0; edge=(1, 2))   # second spell on the same edge
        activate!(dnet, 5.0, 5.0; edge=(3, 1))   # point spell (instantaneous)

        seq = EventSequence(dnet)
        @test seq isa EventSequence{Float64}
        @test length(seq) == 4
        @test [e.time for e in seq] == [1.0, 2.0, 4.0, 5.0]  # sorted by onset
        @test (seq[1].sender, seq[1].receiver) == (1, 2)
        @test (seq[2].sender, seq[2].receiver) == (2, 3)
        @test (seq[3].sender, seq[3].receiver) == (1, 2)
        @test (seq[4].sender, seq[4].receiver) == (3, 1)
        @test all(e.eventtype == :onset for e in seq)
        # The actor universe is declared from the network's vertex set: vertex 4
        # carries no edge spell but stays in the risk set as an isolate
        @test seq.actors == Set([1, 2, 3, 4])
        @test seq.actors_declared
        # ... and can be overridden
        @test EventSequence(dnet; actors=[1, 2, 3]).actors == Set([1, 2, 3])

        # Custom eventtype/weight
        seq_w = EventSequence(dnet; eventtype=:tie_onset, weight=2.0)
        @test all(e.eventtype == :tie_onset && e.weight == 2.0 for e in seq_w)

        # Onset-censored spells are skipped by default (their onset is the
        # observation-window start, not an observed event)
        add_spell!(dnet, NetworkDynamic.Spell(0.0, 2.0; onset_censored=true);
                   edge=(2, 4))
        @test length(EventSequence(dnet)) == 4
        @test length(EventSequence(dnet; include_onset_censored=true)) == 5

        # Undirected networks: edges are stored (min, max), so the smaller
        # ID is the sender
        undnet = DynamicNetwork(3; observation_start=0.0, observation_end=10.0,
                                directed=false)
        activate!(undnet, 1.0, 2.0; edge=(3, 2))
        useq = EventSequence(undnet)
        @test (useq[1].sender, useq[1].receiver) == (2, 3)

        # The converted sequence feeds straight into the REM pipeline
        rng = Random.Xoshiro(3)
        big = DynamicNetwork(6; observation_start=0.0, observation_end=100.0)
        for t in 1:60
            i, j = rand(rng, 1:6), rand(rng, 1:6)
            i == j && continue
            activate!(big, Float64(t), Float64(t) + 1.0; edge=(i, j))
        end
        big_seq = EventSequence(big)
        result = fit_rem(big_seq, [Repetition(), Reciprocity()];
                         n_controls=5, seed=1)
        @test result isa REMResult
        @test all(isfinite, coef(result))

        # Empty dynamic network converts to an empty sequence
        @test length(EventSequence(DynamicNetwork(3))) == 0
    end

    # Conversion invariants (see the ecosystem table in
    # Networks.jl/docs/src/guide/conversion_invariants.md). An Event is an instant
    # and cannot say "this dyad is unobserved", so a masked DynamicNetwork is
    # REJECTED by default: silently turning an unobserved dyad into a
    # never-happened non-event would bias a likelihood that is conditional on
    # the risk set.
    @testset "NetworkDynamic extension: conversion invariants" begin
        dnet = DynamicNetwork(4; observation_start=0.0, observation_end=10.0)
        activate!(dnet, 1.0, 3.0; edge=(1, 2))
        activate!(dnet, 2.0, 5.0; edge=(2, 3))

        # Lossy-by-nature fields are named, not silently dropped
        seq, rep = EventSequence(dnet; report=true)
        @test length(seq) == 2
        @test !Networks.is_lossless(rep)
        dropped = Networks.dropped_fields(rep)
        @test :spell_termini in dropped          # dissolutions are not events
        @test :vertex_spells in dropped          # the risk set is flat over time
        @test :observation_period in dropped
        @test :onset_censored_spells in dropped  # skipped under the default
        @test !(:onset_censored_spells in
                Networks.dropped_fields(EventSequence(dnet;
                    include_onset_censored=true, report=true)[2]))
        @test !(:missing_dyads in dropped)       # no mask on this network

        # A masked dyad (PRESENT face value) and one (ABSENT face value)
        Networks.set_missing_dyad!(dnet.network, 2, 3)
        Networks.set_missing_dyad!(dnet.network, 1, 4)
        @test_throws ArgumentError EventSequence(dnet)
        @test_throws ArgumentError EventSequence(dnet; missing=:error)
        @test_throws ArgumentError EventSequence(dnet; missing=:bogus)

        # Explicit opt-in converts at face value and reports the cost
        seq_face, rep_face = EventSequence(dnet; missing=:face, report=true)
        @test length(seq_face) == 2
        @test :missing_dyads in Networks.dropped_fields(rep_face)

        # Declaring the dyads observed makes the conversion legal again
        Networks.clear_missing_dyads!(dnet.network)
        @test length(EventSequence(dnet)) == 2
    end

    @testset "Coefficient recovery on simulated data" begin
        # Simulate events from a known ordinal REM with inertia
        # (repetition) and reciprocity effects, then recover the
        # coefficients with the full risk set.
        rng = Random.Xoshiro(20260706)
        n_actors = 8
        β_true = [0.6, 0.9]           # [repetition, reciprocity]
        stats = [Repetition(), Reciprocity()]

        dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
        state = EventNetworkState{Float64}(n_actors=n_actors)
        state.actors = Set(1:n_actors)

        events = Event{Float64}[]
        for step in 1:600
            η = [sum(β_true .* compute_all(stats, state, s, r)) for (s, r) in dyads]
            w = exp.(η .- maximum(η))
            w ./= sum(w)
            # Sample a dyad from the softmax
            u = rand(rng)
            acc = 0.0
            pick = length(dyads)
            for (k, p) in enumerate(w)
                acc += p
                if u <= acc
                    pick = k
                    break
                end
            end
            s, r = dyads[pick]
            ev = Event(s, r, Float64(step))
            push!(events, ev)
            update!(state, ev)
        end

        # The actor universe is the one the events were simulated from
        seq = EventSequence(events; actors=1:n_actors)
        # 55 < 56 distinct controls per case, so the sampler enumerates the
        # full risk set — no sampling noise beyond the simulation itself
        result = fit_rem(seq, stats; n_controls=100, seed=1)

        @test result.converged
        @test result.coefficients[1] ≈ β_true[1] atol = 0.25
        @test result.coefficients[2] ≈ β_true[2] atol = 0.25
        # Both effects strongly significant
        @test all(result.p_values .< 0.01)
        # Risk-set bookkeeping: the full 8×7 dyad risk set at every event
        @test length(result.risk_set_sizes) == length(seq)
        @test all(result.risk_set_sizes .== 56)
        @test all(result.sampling_probs .== 1.0)
    end

    @testset "Inferred actor universe warns when fitting" begin
        # Fitting against a universe read off the observed event endpoints
        # silently drops eligible nonparticipants: warn once.
        # (a case that repeats and a case that does not, so the repetition
        # coefficient has a finite maximum and the separation check stays quiet)
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0),
                  Event(3, 2, 4.0)]
        stats = [Repetition()]
        seq = EventSequence(events)               # universe inferred
        @test_logs (:warn, r"inferred from observed event") match_mode = :any begin
            fit_rem(seq, stats; n_controls=3, seed=1)
        end

        # No warning once the universe is declared ...
        seq_d = EventSequence(events; actors=1:5)
        @test_logs fit_rem(seq_d, stats; n_controls=10, seed=1)
        # ... or an explicit risk set is supplied
        @test_logs fit_rem(seq, stats; n_controls=5, seed=1, at_risk=1:3)
    end

    @testset "fit_rem rejects strata without controls" begin
        # A stratum consisting of the case alone contributes nothing but would
        # be silently accepted by the conditional likelihood
        obs = DataFrame(
            event_index = [1, 1, 2],
            sender = [1, 2, 1], receiver = [2, 1, 2],
            x = [1.0, 0.0, 1.0],
            is_event = [true, false, true],
            stratum = [1, 1, 2]
        )
        @test_throws ArgumentError fit_rem(obs, ["x"])
    end

    @testset "Namespace: co-loading ERGM and REM" begin
        # REM.jl#3. The core statnet workflow models cross-sections with ERGM
        # and dynamics with REM, in ONE session. It used to break on the shared
        # verbs: `ERGM.compute !== REM.compute`, so Julia's conflicting-export
        # rule left unqualified `compute` and `name` UNDEFINED after
        # `using ERGM, REM`. The fix is one set of generics in the foundation
        # (Networks.jl `src/statistics.jl`), which both packages extend.

        @testset "the verbs are ONE generic, not two" begin
            @test ERGM.compute === REM.compute === Networks.compute
            @test ERGM.name === REM.name === Networks.name
            @test ERGM.compute_all === REM.compute_all === Networks.compute_all

            # ... and each package still owns its own methods, dispatched on
            # signature: a term takes a network, a statistic takes a state.
            @test !isempty(methods(compute, (REM.AbstractStatistic,
                                             EventNetworkState, Int, Int)))
            @test !isempty(methods(compute, (ERGM.AbstractERGMTerm, Any)))
        end

        @testset "unqualified `compute`/`name` resolve after `using ERGM, REM`" begin
            # The acceptance test, run in a FRESH process because the binding
            # conflict Julia reports is a property of the importing module's
            # namespace, not of this file's (which imports both qualified).
            code = """
                using ERGM, REM
                compute isa Function     || exit(1)   # UndefVarError before the fix
                name isa Function        || exit(2)
                compute_all isa Function || exit(3)
                # the terms/statistics are still reachable unqualified, and the
                # mixing term that remains exported is unambiguously ERGM's
                NodeMix === ERGM.NodeMix || exit(4)
                ActorMix === REM.ActorMix || exit(5)
                # and the verbs actually WORK unqualified, on both domains
                net = network(3); add_edge!(net, 1, 2)
                compute(Edges(), net) == 1.0 || exit(6)
                name(Edges()) == "edges"  || exit(7)
                seq = EventSequence([Event(1, 2, 1.0)]; actors=ActorSet([1, 2, 3]))
                st = EventNetworkState(seq)
                update!(st, seq[1])
                compute(Repetition(), st, 1, 2) == 1.0 || exit(8)
                exit(0)
                """
            @test success(`$(Base.julia_cmd()) --project=$(Base.active_project())
                           --startup-file=no -e $code`)
        end

        @testset "NodeMix belongs to ERGM; REM's is ActorMix" begin
            # Two distinct exported types cannot share a name either. REM's
            # relational-event mixing statistic is `ActorMix`; the old name
            # survives as a deprecated, NON-exported alias, so pre-v0.2 code
            # gets a deprecation warning instead of an UndefVarError.
            @test :ActorMix in names(REM)
            @test !(:NodeMix in names(REM))
            @test REM.NodeMix === REM.ActorMix

            gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"), "")
            stat = REM.NodeMix(gender, "M", "F")
            @test stat isa ActorMix
            @test REM.name(stat) == "mix_gender_M_F"

            st = EventNetworkState(EventSequence([Event(1, 2, 1.0)];
                                                 actors=ActorSet([1, 2])))
            @test REM.compute(stat, st, 1, 2) == 1.0
            @test REM.compute(stat, st, 2, 1) == 0.0
        end
    end

    @testset "Result metadata protocol" begin
        # A relational-event fit must say what it actually did: which objective,
        # whether the risk set was sampled, and how tied times were handled.
        # (six events, so that both coefficients have a finite maximum: the
        # three-event version of this toy separates — coefficients −42 / +25
        # with standard errors of 7e4 — and round 2 made that loud)
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
                  Event(3, 1, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
        stats = [Repetition(), Reciprocity()]
        seq = EventSequence(events; actors=ActorSet([1, 2, 3, 7]))

        # FULL risk set (12 dyads, 11 controls per case): the conditional-logit
        # partial likelihood IS the exact ordinal relational-event likelihood.
        full = fit_rem(seq, stats; n_controls=11, seed=6)
        @test full.sampling_probs == fill(1.0, 6)
        @test isempty(full.separated) && !full.singular

        md_full = Networks.fit_metadata(full)
        @test md_full.estimand == :relational_event
        @test md_full.objective == :partial_likelihood
        @test md_full.is_exact                       # nothing was sampled away
        @test md_full.se_method == :hessian
        @test md_full.missing_method == :none
        # This sequence has no tied timestamps, so no tie policy could bite:
        # `tie_method` says `:none` — not the name of a correction that corrected
        # nothing — and there is no tie caveat to carry.
        @test md_full.tie_method == :none
        @test !any(occursin("tied event times", a) for a in md_full.approximations)
        # A full risk set carries no case-control caveat
        @test !any(occursin("case-control", a) for a in md_full.approximations)
        # ... and with neither approximation in play, the fit is exact, full stop
        @test isempty(md_full.approximations)

        # SAMPLED risk set: the SAME estimator, now an approximation — and the
        # standard errors are conditional on the one control set that was drawn
        # (issue #2).
        sampled = fit_rem(seq, stats; n_controls=4, seed=7)
        @test all(sampled.sampling_probs .< 1.0)
        @test isempty(sampled.separated)             # this draw has a finite maximum

        md_s = Networks.fit_metadata(sampled)
        @test md_s.objective == :partial_likelihood  # same objective
        @test !md_s.is_exact                         # sampled risk set
        @test md_s.se_method == :hessian
        @test any(occursin("case-control sampling of the risk set", a)
                  for a in md_s.approximations)
        # ... and the standard errors are described for what they are — the
        # observed information of the sampled likelihood, a consistent
        # variance estimator — never as "understated"
        @test any(occursin("consistent variance estimator", a) for a in md_s.approximations)
        @test !any(occursin("understated", a) for a in md_s.approximations)
        @test md_s.tie_method == :none               # still no ties in the data

        # The accessors are callable directly, and agree with what `show` prints
        @test Networks.is_exact(full) && !Networks.is_exact(sampled)
        @test occursin("control sampling probability: 1.0", sprint(show, full))

        # UNCONVERGED: the same full-risk-set fit, cut off after one Newton step.
        # An unconverged fit is a loud result — it warns (naming maxiter, the
        # gradient norm and tol), is never exact, and says so in the
        # approximations list and in `show`.
        unconv = @test_logs (:warn, r"did not converge.*maxiter = 1.*gradient norm.*tol") match_mode=:any fit_rem(
            seq, stats; n_controls=11, seed=6, maxiter=1)
        @test !unconv.converged
        @test unconv.iterations == 1
        @test !Networks.is_exact(unconv)             # full risk set, no ties — but unconverged
        md_u = Networks.fit_metadata(unconv)
        @test !md_u.is_exact
        @test any(occursin("did not converge", a) for a in md_u.approximations)
        @test any(occursin("not maximum-partial-likelihood", a) for a in md_u.approximations)
        out_u = sprint(show, unconv)
        @test occursin("Converged: false (1 iteration; see approximations)", out_u)
        @test occursin("did not converge", out_u)
        # The converged fit says how many steps it took, and carries no caveat
        @test full.converged && full.iterations > 1
        @test occursin("Converged: true ($(full.iterations) iterations)", sprint(show, full))
        @test !any(occursin("converge", a) for a in md_full.approximations)
    end

    @testset "Estimation runs on Networks.newton_fit (item 14)" begin
        # REM hosts no Newton loop of its own: `converged` and `iterations` are
        # what the shared optimizer reports, and the fit at the cap is exactly
        # newton_fit's iterate — checked against a direct call on REM's own
        # kernel.
        obs = DataFrame(
            event_index = [1, 1, 2, 2, 3, 3],
            sender = [1, 2, 1, 2, 1, 2],
            receiver = [2, 1, 2, 1, 2, 1],
            x = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            is_event = [true, false, true, false, true, false],
            stratum = [1, 1, 2, 2, 3, 3]
        )
        one_step = @test_logs (:warn, r"converge") match_mode=:any fit_rem(obs, ["x"]; maxiter=1)
        @test !one_step.converged
        @test one_step.iterations == 1
        X = Matrix{Float64}(obs[!, ["x"]])
        idx = REM._strata_index(Vector{Int}(obs.stratum), Vector{Bool}(obs.is_event))
        work = REM._clogit_workspace(idx, 1)
        direct = Networks.newton_fit(REM._clogit_objective(X, idx, work, ones(6)),
                                     [0.0]; maxiter=1)
        @test coef(one_step) == direct.θ
        @test one_step.log_likelihood == direct.loglik
        @test one_step.iterations == direct.iterations == 1
        # ... and the converged fit is newton_fit's converged fit, covariance
        # included (the Cholesky-based inverse information)
        full = fit_rem(obs, ["x"])
        direct = Networks.newton_fit(REM._clogit_objective(X, idx, work, ones(6)), [0.0])
        @test full.converged && direct.converged
        @test coef(full) == direct.θ
        @test vcov(full) == direct.vcov
        @test full.iterations == direct.iterations
        @test full.coefficients[1] ≈ log(2) atol = 1e-6
        # An unconverged fit's covariance still matches its standard errors
        @test stderror(one_step) ≈ sqrt.(diag(vcov(one_step)))
    end

    @testset "Allocation-free clogit kernel (items 14, 26)" begin
        # The per-stratum softmax kernel writes into preallocated buffers and
        # allocates NOTHING after warm-up; the closure handed to newton_fit
        # allocates exactly the (p) gradient and (p×p) Hessian it returns —
        # the same for 300 and for 3000 strata (mirrors ERGMMulti's pin).
        function kernel_allocs(n_strata)
            rng = Random.Xoshiro(11)
            k = 6                              # 1 case + 5 controls per stratum
            n = n_strata * k
            X = randn(rng, n, 3)
            y = [i % k == 1 for i in 1:n]
            strata = [(i - 1) ÷ k + 1 for i in 1:n]
            idx = REM._strata_index(strata, y)
            work = REM._clogit_workspace(idx, 3)
            tw = ones(n)
            grad = zeros(3); hess = zeros(3, 3)
            β = [0.1, -0.2, 0.3]
            REM._clogit_derivatives!(grad, hess, X, idx, β, work, tw)      # warm up
            a_kernel = @allocated REM._clogit_derivatives!(grad, hess, X, idx, β, work, tw)
            f = REM._clogit_objective(X, idx, work, tw)
            f(β)
            a_closure = @allocated f(β)
            # the sandwich meat loop reuses the workspace: O(p²) per call
            bread = Matrix{Float64}(LinearAlgebra.I, 3, 3)
            REM._clogit_sandwich_cov(X, idx, β, bread, tw, work)
            a_sand = @allocated REM._clogit_sandwich_cov(X, idx, β, bread, tw, work)
            return a_kernel, a_closure, a_sand, idx
        end
        k_small, c_small, s_small, idx_small = kernel_allocs(300)
        k_big, c_big, s_big, idx_big = kernel_allocs(3000)
        @test k_small == 0
        @test k_big == 0
        @test c_small <= 512 && c_big <= 512
        @test c_big == c_small                 # p-sized, not strata-sized
        @test s_big == s_small
        @test s_big <= 2048

        # The CSR strata index: one block per stratum, the case located once,
        # deterministic (ascending id) order, and correct for sparse ids too
        @test REM._n_strata(idx_big) == 3000
        @test idx_big.offsets[end] == 18001
        @test all(==(1), idx_big.case)         # the case is row 1 of each block
        @test idx_big.max_size == 6
        strata = [1000, 7, 1000, 7, 7, 10^9]   # sparse ids → Dict path
        y = [true, true, false, false, false, true]
        idx = REM._strata_index(strata, y)
        @test REM._n_strata(idx) == 3
        @test sort([length(idx.offsets[s]:idx.offsets[s+1]-1) for s in 1:3]) == [1, 2, 3]
        @test all(!=(0), idx.case)
        # every row appears exactly once
        @test sort(idx.order) == 1:6
        @test_throws ArgumentError REM._strata_index(strata, y[1:5])
    end

    @testset "Conditional likelihood ignores within-stratum covariate shifts" begin
        # Interleaved strata, cases away from the first row, and a stratum
        # without a case. Column 3 varies between strata but is unidentifiable
        # within each. Integer covariates preserve their contrasts exactly
        # when shifted by large, exactly representable powers of two.
        X = [0.0 1 2; 2 -1 -3; 1 2 2; 0 0 -3; -1 0 2; 1 1 -3; 99 99 99]
        strata = [7, 1000, 7, 1000, 7, 1000, 4]
        y = [false, false, true, false, false, true, false]
        shifted = X .+ [s == 7 ? 2.0^40 : -2.0^40 for s in strata] * [1.0 -2.0 3.0]
        idx = REM._strata_index(strata, y)
        work = REM._clogit_workspace(idx, 3)
        β = [0.2, -0.3, 0.7]
        bread = Matrix{Float64}(LinearAlgebra.I, 3, 3)
        for tw in (ones(7), [1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0])
            ll, grad, hess = REM._clogit_objective(X, idx, work, tw)(β)
            sand = REM._clogit_sandwich_cov(X, idx, β, bread, tw, work)
            ll_shift, grad_shift, hess_shift = REM._clogit_objective(shifted, idx, work, tw)(β)
            sand_shift = REM._clogit_sandwich_cov(shifted, idx, β, bread, tw, work)
            @test ll_shift == ll
            @test grad_shift == grad
            @test hess_shift == hess
            @test sand_shift == sand
            @test grad[3] == 0.0
            @test all(iszero, hess[3, :]) && all(iszero, hess[:, 3])
            @test all(iszero, sand[3, :]) && all(iszero, sand[:, 3])
        end
    end

    @testset "Streams in O(events): generate_observations allocation pins" begin
        # The per-event cost must not carry the actor universe: the old
        # `n_dyads` built two Sets of every actor per event (139 KB/event at
        # 2000 actors), and every row was an Observation with its own Vector.
        # Now a static risk set's dyad count is cached, the design accumulates
        # column-major, and `n_dyads` itself is an allocation-free sorted merge.
        function stream_allocs(n_actors, n_events)
            rng = Random.Xoshiro(3)
            ev = Event{Float64}[]
            for k in 1:n_events
                s = rand(rng, 1:n_actors)
                r = rand(rng, 1:(n_actors - 1)); r >= s && (r += 1)
                push!(ev, Event(s, r, Float64(k)))
            end
            seq = EventSequence(ev; actors=ActorSet(1:n_actors))
            ss = StatisticSet([Repetition(), Reciprocity(), SenderActivity()])
            sampler = CaseControlSampler(n_controls=20, seed=1)
            generate_observations(seq, ss, sampler)                 # warm up
            return @allocated generate_observations(seq, ss, sampler)
        end
        a_100_2000 = stream_allocs(100, 2000)
        a_2000_2000 = stream_allocs(2000, 2000)
        a_100_4000 = stream_allocs(100, 4000)
        # 20× the actors: at most 1.5× the bytes (the O(n_actors) terms are
        # one-time — the sorted id vector and the state's actor set)
        @test a_2000_2000 <= 1.5 * a_100_2000
        # 2× the events: at most 2.2× the bytes (linear in the stream)
        @test a_100_4000 <= 2.2 * a_100_2000

        # The fit's stratum validation runs behind a function barrier: on the
        # DataFrame's `AbstractVector`-typed columns it used to dispatch per row
        # (≈10 allocations / 250 B per row — 420k allocations on a 2000-event
        # design). Now O(strata) Dict growth only: a few dozen allocations for
        # 42 000 rows (measured 47), and 84 000 rows cost at most the extra
        # doublings of two Dicts (measured 59).
        function fit_allocations(n_events)
            rng = Random.Xoshiro(3)
            ev = [Event(rand(rng, 1:100), rand(rng, 1:100), Float64(k)) for k in 1:n_events]
            ev = [Event(e.sender, e.sender == e.receiver ? mod1(e.receiver + 1, 100) : e.receiver, e.time)
                  for e in ev]
            seq = EventSequence(ev; actors=ActorSet(1:100))
            obs = generate_observations(seq, [Repetition(), Reciprocity()],
                                        CaseControlSampler(n_controls=20, seed=1))
            REM._stratum_counts(obs.stratum, obs.is_event)               # warm up
            return @allocations REM._stratum_counts(obs.stratum, obs.is_event)
        end
        n_2000 = fit_allocations(2000)
        n_4000 = fit_allocations(4000)
        @test n_2000 <= 64
        @test n_4000 <= n_2000 + 16

        # `n_dyads` on a sorted risk set allocates nothing and is right; on an
        # unsorted or duplicated one it still counts correctly (through sets)
        ids = collect(1:2000)
        rs = RiskSet(1, ids, ids)
        n_dyads(rs)
        nd(rs) = n_dyads(rs)
        @test @allocated(nd(rs)) == 0
        @test n_dyads(rs) == 2000 * 1999
        @test n_dyads(RiskSet(1, [3, 1, 2], [2, 2, 1])) == 3 * 3 - 2   # unsorted, dup
        @test n_dyads(RiskSet(1, [1, 2], [3, 4])) == 4                  # disjoint
        @test n_dyads(RiskSet(1, [1, 2], [1, 2]; exclude_self_loops=false)) == 4

        # The per-event log is kept only for statistics that read it: none of
        # REM's own do, so a fit on them retains O(actors + dyads), not
        # O(events); a foreign statistic (Relevent's PShift, a user's) is assumed
        # to need it until it says otherwise — nothing is silently dropped
        for st in (Repetition(), Reciprocity(), SenderActivity(), TransitiveClosure(),
                   FourCycle(), NodeSum(NodeAttribute(:x, 0.0)), RecencyStatistic())
            @test !needs_history(st)
        end
        @test !REM._keeps_history(StatisticSet([Repetition(), TransitiveClosure()]))
        @test needs_history(_LogReader())          # defined at top level above
        @test REM._keeps_history(StatisticSet([Repetition(), _LogReader()]))
        ev = [Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0), Event(1, 2, 4.0)]
        sq = EventSequence(ev; actors=ActorSet(1:4))
        # ... and the log really is there for it: the previous sender at event
        # k is seq[k-1].sender
        df = compute_statistics(sq, [Repetition(), _LogReader()])
        @test df.log_reader == [0.0, 1.0, 0.0, 0.0]
        st = EventNetworkState(sq; keep_history=false)
        for e in sq; update!(st, e); end
        @test isempty(st.event_history)
        @test get_dyad_count(st, 1, 2) == 2.0
        st2 = EventNetworkState(sq)                       # default keeps it
        for e in sq; update!(st2, e); end
        @test length(st2.event_history) == 4
        reset!(st2)
        @test isempty(st2.event_history)
    end

    @testset "rng contract: every draw flows through `rng` (item 16)" begin
        rng0 = MersenneTwister(4)
        evs = Event{Float64}[]
        for k in 1:60
            s_, r_ = rand(rng0, 1:12), rand(rng0, 1:12)
            while r_ == s_
                r_ = rand(rng0, 1:12)
            end
            push!(evs, Event(s_, r_, Float64(k)))
        end
        seq = EventSequence(evs; actors=ActorSet(collect(1:12)))
        stats = [Repetition(), Reciprocity(), SenderActivity()]

        # Without `seed`, the control draw comes from `rng` — not from the global
        # RNG behind the caller's back — so a fixed rng reproduces the fit
        a = fit_rem(seq, stats; n_controls=5, rng=Random.Xoshiro(9))
        b = fit_rem(seq, stats; n_controls=5, rng=Random.Xoshiro(9))
        @test coef(a) == coef(b)
        @test vcov(a) == vcov(b)
        @test coef(fit_rem(seq, stats; n_controls=5, rng=Random.Xoshiro(10))) != coef(a)
        # ... and the global RNG stream is untouched by it
        Random.seed!(1234); m1 = rand()
        Random.seed!(1234); fit_rem(seq, stats; n_controls=5, rng=Random.Xoshiro(9)); m2 = rand()
        @test m1 == m2

        # `seed` pins the control draw and takes precedence over `rng`
        s1 = fit_rem(seq, stats; n_controls=5, seed=42)
        s2 = fit_rem(seq, stats; n_controls=5, seed=42, rng=Random.Xoshiro(9))
        @test coef(s1) == coef(s2) && vcov(s1) == vcov(s2)

        # The same at the observation level
        ss = StatisticSet(stats)
        o1 = generate_observations(seq, ss, CaseControlSampler(n_controls=5); rng=Random.Xoshiro(9))
        o2 = generate_observations(seq, ss, CaseControlSampler(n_controls=5); rng=Random.Xoshiro(9))
        @test o1 == o2
        o3 = generate_observations(seq, ss, CaseControlSampler(n_controls=5, seed=42); rng=Random.Xoshiro(9))
        o4 = generate_observations(seq, ss, CaseControlSampler(n_controls=5, seed=42))
        @test o3 == o4

        # `control_draw_cov` with a fixed rng: every per-draw seed comes from
        # `rng` (drawn up front, through Networks.bootstrap_cov), so two runs
        # are identical, the global RNG is untouched, and the threaded and the
        # serial loops give the SAME bits — which is the thread-count
        # independence contract, pinned here rather than asserted in prose
        # (CI runs this file at 1 and at 4 threads)
        d1 = control_draw_cov(seq, stats; n_controls=5, n_draws=8, rng=Random.Xoshiro(7))
        d2 = control_draw_cov(seq, stats; n_controls=5, n_draws=8, rng=Random.Xoshiro(7))
        @test d1.cov == d2.cov && d1.replicates == d2.replicates
        Random.seed!(1234); m1 = rand()
        Random.seed!(1234); control_draw_cov(seq, stats; n_controls=5, n_draws=8, rng=Random.Xoshiro(7)); m2 = rand()
        @test m1 == m2
        serial = control_draw_cov(seq, stats; n_controls=5, n_draws=8, rng=Random.Xoshiro(7),
                                  threaded=false)
        @test serial.cov == d1.cov && serial.replicates == d1.replicates
        @test control_draw_cov(seq, stats; n_controls=5, n_draws=8, rng=Random.Xoshiro(8)).cov != d1.cov
    end

    @testset "The design is the DataFrame: no Observation row type" begin
        # `generate_observations` builds its frame column-major; the pre-0.2
        # `Observation` row type and its `observations_to_dataframe` converter
        # were produced by nothing and consumed by nothing (and had no test),
        # so they are gone rather than kept as a second design builder that
        # drifts. A hand-built design is a DataFrame with `is_event`,
        # `stratum` and the statistic columns.
        @test !isdefined(REM, :Observation)
        @test !isdefined(REM, :observations_to_dataframe)
        @test !(:Observation in names(REM))
        hand = DataFrame(is_event=[true, false, true, false], stratum=[1, 1, 2, 2],
                         x=[1.0, 0.0, 0.0, 1.0])
        hf = fit_rem(hand, ["x"])
        @test hf.n_events == 2 && hf.converged
        @test coef(hf)[1] ≈ 0.0 atol=1e-8
    end

    @testset "No export collides with Base (no `rem` alias)" begin
        # The statnet-style verb of this package would be `rem` — which is
        # `Base.rem`, the remainder function, exported by Base. A rival export
        # would make `rem(7, 2)` an ambiguity error in every `using REM`
        # session, so `fit_rem` is the ONE entry point (relevent's R names live
        # in Relevent.jl as `rem_dyad`). Pinned for every exported name.
        clashes = [n for n in names(REM) if Base.isexported(Base, n)]
        @test isempty(clashes)
        @test !(:rem in names(REM))
        @test rem(7, 2) == 1                   # still Base.rem after `using REM`
        @test isdefined(REM, :fit_rem) && Base.isexported(REM, :fit_rem)
    end

    @testset "Tied event times: ties=:error|:ordered|:breslow|:efron" begin
        # Issue REM#2 / review finding 12. The likelihood is a likelihood over
        # the ORDER of the events, and the statistics are read off the network
        # state as it stands BEFORE each event — so ordering a tie does not just
        # pick a sort, it lets the event placed first enter the STATISTICS of the
        # event placed second. That is invented information, and the default is
        # now to refuse it.
        actors = ActorSet([1, 2, 3, 4])
        tied = [Event(1, 2, 1.0), Event(2, 1, 1.0),      # a tie at t = 1
                Event(1, 3, 2.0),
                Event(3, 1, 3.0), Event(2, 3, 3.0),      # a tie at t = 3
                Event(1, 2, 4.0)]
        seq = EventSequence(tied; actors=actors)
        untied = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
                  Event(3, 1, 4.0), Event(2, 3, 5.0), Event(1, 2, 6.0)]
        seq_u = EventSequence(untied; actors=actors)
        stats = [Repetition(), Reciprocity()]
        full = 11                                        # 4·3 dyads − the case

        # --- the default REFUSES, and names the tie -------------------------
        err = try
            fit_rem(seq, stats; n_controls=full); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("tied timestamps", msg)
        @test occursin("events 1–2", msg)                # WHICH events
        @test occursin("t = 1.0", msg)                   # at WHICH time
        @test occursin("2 timestamps carry ties", msg)   # and how many in all
        @test occursin(":breslow", msg) && occursin(":efron", msg)  # what to do
        # `generate_observations` refuses at the same door (the design is where
        # the tie is actually resolved)
        @test_throws ArgumentError generate_observations(
            seq, stats, CaseControlSampler(n_controls=full))

        # --- an unimplemented option FAILS LOUDLY, never no-ops --------------
        e_batch = try; fit_rem(seq, stats; ties=:batch); catch e; e; end
        @test e_batch isa ArgumentError
        m_batch = sprint(showerror, e_batch)
        @test occursin("`:batch` is not defined", m_batch)
        @test occursin("IS the Breslow correction", m_batch)   # ... and why
        e_junk = try; fit_rem(seq, stats; ties=:hamburger); catch e; e; end
        @test e_junk isa ArgumentError
        @test occursin("unknown tie policy", sprint(showerror, e_junk))
        @test occursin("Networks.TIE_POLICIES", sprint(showerror, e_junk))
        # The vocabulary is the ecosystem's, defined once in Networks.jl
        @test Networks.TIE_POLICIES ==
              (:error, :ordered, :breslow, :efron, :batch)

        # --- on TIE-FREE data every policy is a no-op ------------------------
        # The sharpest correctness check available: a tie correction that changes
        # anything on data without ties is not a tie correction.
        fits_u = [fit_rem(seq_u, stats; n_controls=full, ties=t)
                  for t in (:error, :ordered, :breslow, :efron)]
        for f in fits_u[2:end]
            @test coef(f) == coef(fits_u[1])
            @test stderror(f) == stderror(fits_u[1])
            @test f.log_likelihood == fits_u[1].log_likelihood
        end
        # ... and none of them CLAIMS to have done anything
        @test all(Networks.tie_method(f) === :none for f in fits_u)
        @test all(Networks.is_exact(f) for f in fits_u)
        @test all(isempty(Networks.approximations(f)) for f in fits_u)
        # the design itself is identical, row for row
        dfs = [generate_observations(seq_u, stats, CaseControlSampler(n_controls=full);
                                     ties=t) for t in (:ordered, :breslow, :efron)]
        @test all(df -> df == dfs[1], dfs)
        @test all(==(1.0), dfs[1].tie_weight)

        # --- on TIED data the three policies genuinely differ ----------------
        o = fit_rem(seq, stats; n_controls=full, ties=:ordered)
        b = fit_rem(seq, stats; n_controls=full, ties=:breslow)
        ef = fit_rem(seq, stats; n_controls=full, ties=:efron)
        @test coef(o) != coef(b) && coef(b) != coef(ef)

        # `tie_method` reports what was ACTUALLY used — never `:error`, which
        # cannot survive to a result
        @test Networks.tie_method(o) === :ordered
        @test Networks.tie_method(b) === :breslow
        @test Networks.tie_method(ef) === :efron
        @test Networks.fit_metadata(ef).tie_method === :efron

        # ... and `approximations` carries the caveat, naming the correction
        @test any(occursin("BRESLOW correction", a) for a in Networks.approximations(b))
        @test any(occursin("EFRON correction", a) for a in Networks.approximations(ef))
        @test any(occursin("ordered arbitrarily", a) for a in Networks.approximations(o))
        # Breslow is the cruder approximation and says so
        @test any(occursin("cruder", a) for a in Networks.approximations(b))
        # A tied fit is NOT exact, whatever the policy — the order the likelihood
        # is over is not in the data
        @test !Networks.is_exact(o) && !Networks.is_exact(b) && !Networks.is_exact(ef)
        @test all(f -> all(≈(1.0), sampling_probs(f)), (o, b, ef))   # full risk set
        @test occursin("Tied event times: efron", sprint(show, ef))
        @test !occursin("Tied event times", sprint(show, fits_u[1]))

        # --- the corrections FREEZE the state across the tie -----------------
        # Under `:ordered`, event 2 (2→1) sees event 1 (1→2) and its Reciprocity
        # is 1. Under a correction it cannot: simultaneous events did not precede
        # one another.
        df_o = generate_observations(seq, stats, CaseControlSampler(n_controls=full);
                                     ties=:ordered)
        df_b = generate_observations(seq, stats, CaseControlSampler(n_controls=full);
                                     ties=:breslow)
        case2_o = df_o[(df_o.stratum .== 2) .& df_o.is_event, :]
        case2_b = df_b[(df_b.stratum .== 2) .& df_b.is_event, :]
        @test only(case2_o.reciprocity) == 1.0    # invented by the arbitrary sort
        @test only(case2_b.reciprocity) == 0.0    # the tie cannot see itself
        # ... and the block is absorbed as a WHOLE: the next stratum sees both of
        # its events (frozen ≠ dropped)
        s3_b = df_b[df_b.stratum .== 3, :]
        @test only(s3_b[(s3_b.sender .== 1) .& (s3_b.receiver .== 2), :].repetition) == 1.0
        @test only(s3_b[(s3_b.sender .== 2) .& (s3_b.receiver .== 1), :].repetition) == 1.0

        # --- Efron's weights are the 1 − (j−1)/d of the textbook -------------
        df_e = generate_observations(seq, stats, CaseControlSampler(n_controls=full);
                                     ties=:efron)
        # tie at t = 1: d = 2, so the two tied cases enter stratum 1 with weight 1
        # and stratum 2 with weight 1/2 — and NOTHING else is weighted
        w1 = df_e[df_e.stratum .== 1, :]
        w2 = df_e[df_e.stratum .== 2, :]
        @test all(==(1.0), w1.tie_weight)
        @test sum(w2.tie_weight .== 0.5) == 2         # both tied cases, down-weighted
        @test sum(w2.tie_weight .== 1.0) == nrow(w2) - 2
        # the other tied case is IN the stratum as a denominator row, not a case
        @test sum(w2.is_event) == 1
        @test nrow(w2) == 12                          # the whole risk set, still
        # untied strata are untouched
        @test all(==(1.0), df_e[df_e.stratum .== 3, :].tie_weight)

        # --- the DataFrame method reports the truth it was handed ------------
        # The policy is applied when the design is built; a fit from the design
        # must not be able to claim otherwise (or to forget).
        fit_df = fit_rem(df_e, ["repetition", "reciprocity"])
        @test Networks.tie_method(fit_df) === :efron
        @test coef(fit_df) ≈ coef(ef)
        # A hand-built frame with neither weights nor metadata is unweighted and
        # says `:none` — it cannot know, so it does not claim
        plain = DataFrame(repetition=df_e.repetition, reciprocity=df_e.reciprocity,
                          is_event=df_e.is_event, stratum=df_e.stratum)
        plain_fit = fit_rem(plain, ["repetition", "reciprocity"])
        @test Networks.tie_method(plain_fit) === :none
        # ... and unweighted, which on the Efron DESIGN (same rows, same frozen
        # statistics) is exactly Breslow. Efron is Breslow plus the weights, and
        # here that identity is visible rather than asserted.
        @test coef(plain_fit) ≈ coef(b)
        # A design that claims the Efron correction but has lost the weights that
        # ARE the correction is a lie, and is refused rather than fitted
        @test_throws ArgumentError fit_rem(select(df_e, Not(:tie_weight)),
                                           ["repetition", "reciprocity"])

        # --- Efron needs DISTINCT tied cases --------------------------------
        # One dyad acting twice at one instant is a risk-set member competing with
        # itself; its Efron weight is not defined by anything. Refuse, rather than
        # invent one (Breslow's denominator is the plain risk-set sum and is fine).
        dup = EventSequence([Event(1, 2, 1.0), Event(1, 2, 1.0), Event(2, 3, 2.0)];
                            actors=actors)
        e_dup = try; fit_rem(dup, stats; n_controls=full, ties=:efron); catch e; e; end
        @test e_dup isa ArgumentError
        @test occursin("distinct dyads", sprint(showerror, e_dup))
        @test fit_rem(dup, stats; n_controls=full, ties=:breslow).converged

        # --- the policy survives the OTHER options ---------------------------
        # (the standard-error estimator and the tie correction are orthogonal,
        # and the control-draw diagnostic builds every redraw under the policy)
        sand = fit_rem(seq, stats; n_controls=3, seed=3, ties=:efron, se=:sandwich)
        @test Networks.tie_method(sand) === :efron
        draws = control_draw_cov(seq, stats; n_controls=3, n_draws=5, ties=:efron,
                                 rng=MersenneTwister(1))
        @test size(draws.replicates) == (5, length(stats))
        @test_throws ArgumentError control_draw_cov(seq, stats; n_controls=3, n_draws=5,
                                                    rng=MersenneTwister(1))   # ties=:error
        @test_throws ArgumentError control_draw_cov(seq, stats; n_controls=3, n_draws=5,
                                                    ties=:batch, rng=MersenneTwister(1))
    end
    @testset "Robust standard errors: se=:sandwich, and control_draw_cov" begin
        # Two standard-error estimators, and one diagnostic that is NOT one:
        #   :hessian   — the observed information of the (sampled) partial
        #                likelihood; a consistent variance estimator under
        #                nested case-control sampling (Goldstein & Langholz
        #                1992) — see the calibration testset
        #   :sandwich  — event-clustered Godambe, drops the information equality
        #   control_draw_cov — the spread of the point estimate ACROSS control
        #                draws; a sensitivity diagnostic, never combined with
        #                the Hessian (that would double-count) and never
        #                reported by `stderror`
        rng0 = MersenneTwister(4)
        evs = Event{Float64}[]
        for k in 1:80
            s_, r_ = rand(rng0, 1:12), rand(rng0, 1:12)
            while r_ == s_
                r_ = rand(rng0, 1:12)
            end
            push!(evs, Event(s_, r_, Float64(k)))
        end
        seq = EventSequence(evs; actors=ActorSet(collect(1:12)))
        stats = [Repetition(), Reciprocity(), SenderActivity()]

        hess = fit_rem(seq, stats; n_controls=5, seed=42)
        sand = fit_rem(seq, stats; n_controls=5, seed=42, se=:sandwich)

        # The sandwich replaces only the COVARIANCE: the point estimates (and the
        # likelihood, and the risk-set bookkeeping) are identical
        @test coef(sand) == coef(hess)
        @test sand.log_likelihood == hess.log_likelihood
        @test sampling_probs(sand) == sampling_probs(hess)
        @test stderror(sand) != stderror(hess)
        @test all(isfinite, stderror(sand))

        # Each result carries the covariance its standard errors are the diagonal
        # of — the inverse information, the sandwich — so `vcov` matches
        # `se_method` under both (StatsAPI, item 15)
        for f in (hess, sand)
            @test stderror(f) ≈ sqrt.(diag(vcov(f)))
            @test check_statsapi(f; strict=true) !== nothing
            @test issymmetric(round.(vcov(f), digits=12))
        end
        @test vcov(sand) != vcov(hess)

        # `se_method` reports what was ACTUALLY used, both ways
        @test Networks.se_method(hess) === :hessian
        @test Networks.se_method(sand) === :sandwich
        @test Networks.fit_metadata(sand).se_method === :sandwich

        # ... and so does the printed output. A sampled fit says the risk set
        # was sampled and what to do about a misspecified model — it does NOT
        # call any estimator "understated" (the observed information of the
        # sampled likelihood is already the right variance)
        out_h, out_s = sprint.(show, (hess, sand))
        @test occursin("inverse Hessian (one control draw)", out_h)
        @test occursin("Note: the risk set was sampled (5 of 131 controls per event)", out_h)
        @test occursin("control_draw_cov", out_h)
        @test !occursin("understated", out_h)
        @test occursin("event-clustered sandwich", out_s)
        @test occursin("robust to", out_s)
        @test !occursin("understated", out_s)
        @test !occursin("bootstrap", out_h) && !occursin("bootstrap", out_s)

        # The approximations list agrees with the printed prose
        @test any(occursin("consistent variance estimator", a) for a in Networks.approximations(hess))
        @test any(occursin("robust to misspecification", a)
                  for a in Networks.approximations(sand))
        @test !any(occursin("understated", a) for a in Networks.approximations(hess))
        @test !any(occursin("understated", a) for a in Networks.approximations(sand))
        # The tie-handling report is a property of the LIKELIHOOD, not of the
        # standard errors, so it survives both untouched — and this sequence
        # (one event per integer time) has no ties at all, so both say so
        @test all(Networks.tie_method(r) === :none for r in (hess, sand))
        @test all(!any(occursin("tied event times", a) for a in Networks.approximations(r))
                  for r in (hess, sand))

        # The control inclusion probabilities are exposed, not buried
        @test length(sampling_probs(hess)) == 80          # one per stratum/event
        @test length(risk_set_sizes(hess)) == 80
        @test all(0 .< sampling_probs(hess) .< 1)         # the risk set WAS sampled
        @test all(==(132), risk_set_sizes(hess))          # 12·11 ordered dyads
        @test !Networks.is_exact(hess)

        # With the FULL risk set there is no control sampling: nothing is
        # approximated, so no note is printed and `is_exact` holds
        full = fit_rem(seq, stats; n_controls=200, seed=1)
        @test all(≈(1.0), sampling_probs(full))
        @test Networks.is_exact(full)
        @test !occursin("Note:", sprint(show, full)) && !occursin("Warning:", sprint(show, full))

        # --- control_draw_cov: the draw-to-draw spread, as a diagnostic -------
        draws = control_draw_cov(seq, stats; n_controls=5, n_draws=40, rng=MersenneTwister(9))
        @test draws.cov isa Matrix{Float64} && size(draws.cov) == (3, 3)
        @test issymmetric(round.(draws.cov, digits=12))
        @test draws.sd ≈ sqrt.(diag(draws.cov))
        @test size(draws.replicates) == (40, 3)
        @test draws.mean ≈ vec(sum(draws.replicates; dims=1) ./ 40)
        @test draws.n_unconverged == 0
        @test all(isfinite, draws.sd) && all(draws.sd .> 0)
        # reproducible under a fixed rng, different under another
        @test control_draw_cov(seq, stats; n_controls=5, n_draws=40, rng=MersenneTwister(9)).cov == draws.cov
        @test control_draw_cov(seq, stats; n_controls=5, n_draws=40, rng=MersenneTwister(10)).cov != draws.cov
        # it is a property of the sampling design: with the full risk set every
        # "draw" is the same design. Repeated coefficients must be identical;
        # covariance centering can still leave a rounding-sized residual.
        full_draws = control_draw_cov(seq, stats; n_controls=200, n_draws=3,
                                      rng=MersenneTwister(1))
        @test all(row -> row == full_draws.replicates[1, :],
                  eachrow(full_draws.replicates))
        @test all(full_draws.sd .<= eps.(max.(abs.(full_draws.mean), 1.0)))
        # it is NOT a standard error: nothing on a REMResult is it, and it is
        # never combined with the Hessian — `vcov(hess)` is the inverse
        # information, full stop
        @test !any(occursin("bootstrap", string(f)) for f in fieldnames(REMResult))
        @test_throws ArgumentError control_draw_cov(seq, stats; n_controls=5, n_draws=1)
        @test_throws ArgumentError control_draw_cov(seq, AbstractStatistic[]; n_controls=5)
        @test_throws ArgumentError control_draw_cov(evs, stats; n_controls=5)   # Vector{Event}

        # `se=:bootstrap` is refused — on both methods — with the reason and a
        # pointer to the diagnostic, before the shared validator runs
        obs = generate_observations(seq, StatisticSet(stats),
                                    CaseControlSampler(n_controls=5, seed=42))
        for f in (() -> fit_rem(seq, stats; se=:bootstrap),
                  () -> fit_rem(obs, ["repetition", "reciprocity"]; se=:bootstrap))
            err = try; f(); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("control_draw_cov", err.msg)
            @test occursin("not a standard error", err.msg)
        end
        # ... but the sandwich needs no redraw and works fine on the DataFrame
        @test Networks.se_method(fit_rem(obs, ["repetition", "reciprocity"]; se=:sandwich)) === :sandwich

        # Unknown se symbols are rejected, not silently ignored — through the ONE
        # shared validator (`Networks.check_se`, item 28), whose message names
        # the fitter, the vocabulary and the offender
        @test_throws ArgumentError fit_rem(seq, stats; se=:jackknife)
        @test_throws ArgumentError fit_rem(obs, ["repetition", "reciprocity"]; se=:jackknife)
        err = try
            fit_rem(obs, ["repetition", "reciprocity"]; se=:jackknife)
        catch e
            e
        end
        @test occursin("fit_rem(::DataFrame)", err.msg) && occursin(":jackknife", err.msg)
        @test occursin("(:hessian, :sandwich)", err.msg)
        err = try
            fit_rem(seq, stats; se=:jackknife)
        catch e
            e
        end
        @test occursin("(:hessian, :sandwich)", err.msg)
    end

    @testset "Sampled-risk-set standard errors are calibrated (nested case-control)" begin
        # Reference: Goldstein & Langholz (1992, Ann. Statist. 20, 1903–1928)
        # and Borgan, Goldstein & Langholz (1995, Ann. Statist. 23, 1749–1778):
        # the inverse observed information of the SAMPLED partial likelihood
        # is a consistent estimator of the variance of the nested case-control
        # estimator. The information lost by sampling fewer controls is what
        # makes it larger than the full-risk-set one; there is no further
        # "risk-set sampling variance" to add, and adding the between-draw
        # covariance of refits (the pre-round-2 `se=:bootstrap`) double-counted
        # it (97–99 % coverage measured). Pinned here by simulation from the
        # package's own correctly specified generator, events AND controls
        # redrawn on every replicate: Wald coverage of `se=:hessian` within
        # [0.92, 0.98], the mean reported standard error within 15 % of the
        # empirical standard deviation of the estimates.
        β_true = [0.6, 0.9]
        stats = [Repetition(), Reciprocity()]
        n_rep = 300
        rng = Random.Xoshiro(20260909)
        est = zeros(n_rep, 2); se_h = zeros(n_rep, 2)
        covered_h = zeros(Int, 2); covered_s = zeros(Int, 2)
        understated = false
        for r in 1:n_rep
            seq = _simulate_rem(rng, 12, 300, β_true, stats)
            seed = rand(rng, 1:10^9)
            f = fit_rem(seq, stats; n_controls=10, seed=seed)       # 10 of 131 controls
            fs = fit_rem(seq, stats; n_controls=10, seed=seed, se=:sandwich)
            est[r, :] .= coef(f); se_h[r, :] .= stderror(f)
            ci = confint(f); cis = confint(fs)
            for j in 1:2
                covered_h[j] += ci[j, 1] <= β_true[j] <= ci[j, 2]
                covered_s[j] += cis[j, 1] <= β_true[j] <= cis[j, 2]
            end
            understated |= any(occursin("understated", a) for a in Networks.approximations(f))
            understated |= occursin("understated", sprint(show, f))
        end
        coverage_h = covered_h ./ n_rep
        coverage_s = covered_s ./ n_rep
        # measured (this seed): coverage_h = [0.953, 0.967], coverage_s = [0.94, 0.94]
        @test all(0.92 .<= coverage_h .<= 0.98)
        @test all(0.88 .<= coverage_s .<= 0.98)          # the sandwich is slightly anti-conservative in small samples
        # the mean reported SE against the empirical spread of the estimates
        # (measured 0.82 and 0.80: log-hazard estimates are right-skewed in
        # samples this small, which inflates the sd without hurting coverage)
        emp_sd = vec(std(est; dims=1))
        mean_se = vec(sum(se_h; dims=1) ./ n_rep)
        @test all(0.75 .<= mean_se ./ emp_sd .<= 1.25)
        @test all(abs.(vec(sum(est; dims=1) ./ n_rep) .- β_true) .< 0.15)   # centred on the truth
        @test !understated                                # no estimator is labelled understated
    end

    @testset "Collinearity and separation are loud (criterion 2)" begin
        # Singular information makes the shared optimizer report nonconvergence;
        # separation can still satisfy its objective stopping rule. The bundled
        # WTC calls exhibit both identification failures: the
        # ICR covariate as NodeSum is the sum of SenderAttribute and
        # ReceiverAttribute (singular information), and no ICR actor ever calls
        # another (NodeProduct separates) — the two cases the estimation guide's
        # "Common Causes" table names.
        wtc = Networks.load_dataset(:wtc_police_calls)
        e = wtc.events
        seq = EventSequence([Event(e[k, 2], e[k, 3], Float64(e[k, 1])) for k in 1:size(e, 1)];
                            actors=ActorSet(1:37))
        icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:37))
        full = 37 * 36 - 1

        # --- singular: NaN standard errors are recorded, named and printed ----
        col = @test_logs (:warn, r"newton_fit: the Hessian") (:warn, r"singular.*Suspect statistics: `sum_icr`, `sender_icr`, `receiver_icr`") match_mode=:any fit_rem(
            seq, [NodeSum(icr), SenderAttribute(icr), ReceiverAttribute(icr)]; n_controls=full)
        @test !col.converged
        @test all(isnan, stderror(col)) && all(isnan, vcov(col))
        @test col.singular
        @test col.singular_suspects == ["sum_icr", "sender_icr", "receiver_icr"]
        @test isempty(col.separated)
        @test !Networks.is_exact(col)                   # full risk set and no ties cannot identify collinear effects
        @test any(occursin("singular", a) && occursin("`sum_icr`", a)
                  for a in Networks.approximations(col))
        out = sprint(show, col)
        @test occursin("Warning: the observed information at the solution is singular", out)
        @test occursin("`sum_icr`, `sender_icr`, `receiver_icr`", out)
        @test occursin("NodeSum(x)", out)
        # a statistic constant within every stratum is singular on its own:
        # a dyad covariate equal on every dyad
        flat = @test_logs (:warn, r"newton_fit") (:warn, r"Suspect statistics: `flat`") match_mode=:any fit_rem(
            seq, [Repetition(), DyadCovariate(Dict{Tuple{Int,Int},Float64}(); default=1.0, name="flat")];
            n_controls=20, seed=1)
        @test flat.singular && flat.singular_suspects == ["flat"]

        # --- separation: a coefficient running away is named, not tabled -----
        sep = @test_logs (:warn, r"coefficient on `product_icr` may be infinite \(separation\)") match_mode=:any fit_rem(
            seq, [Repetition(), NodeProduct(icr)]; n_controls=full)
        @test sep.converged
        @test abs(coef(sep)[2]) > 10 && stderror(sep)[2] > 1000   # the symptom
        @test sep.separated == ["product_icr"]
        @test !sep.singular
        @test !Networks.is_exact(sep)
        @test any(occursin("`product_icr`", a) && occursin("may be infinite", a)
                  for a in Networks.approximations(sep))
        out = sprint(show, sep)
        @test occursin("Warning: the coefficient on `product_icr` may be infinite (separation)", out)
        # the healthy coefficient in the same fit is not flagged
        @test !occursin("`repetition`", out)
        # ... and the DataFrame method carries the same diagnostics
        obs = generate_observations(seq, [Repetition(), NodeProduct(icr)],
                                    CaseControlSampler(n_controls=full))
        sep_df = @test_logs (:warn, r"may be infinite") match_mode=:any fit_rem(obs, ["repetition", "product_icr"])
        @test sep_df.separated == ["product_icr"]
        # ... and the verdict is SCALE-FREE (round 3): the same separated
        # product with the attribute coded 0/1000 (values 0/1e6) or 0/0.001
        # is flagged too. On the raw scale the ×1000 fit came back as
        # β = −2.4e-5, SE 0.13, separated = [], is_exact == true — a z ≈ 0
        # row a reader would take for "no effect" when the maximum is at −∞.
        # What is scale-free is the VERDICT and the direction of divergence,
        # not the finite value Newton stops at: the maximum is at −∞, so the
        # reported coefficient depends on the iteration path and does not obey
        # β(scaled) · scale == β(raw) (observed −2.4e-5 and −2.0e7 against
        # −20.1 on the raw scale).
        for (scale, tag) in ((1000.0, "icrk"), (0.001, "icrm"))
            att = NodeAttribute(Symbol(tag), Dict(i => scale * Float64(wtc.is_icr[i]) for i in 1:37))
            sc = @test_logs (:warn, Regex("coefficient on `product_$tag` may be infinite")) match_mode=:any fit_rem(
                seq, [Repetition(), NodeProduct(att)]; n_controls=full)
            @test sc.converged && !sc.singular
            @test sc.separated == ["product_$tag"]
            @test !Networks.is_exact(sc)
            @test sign(coef(sc)[2]) == sign(coef(sep)[2])             # same direction of divergence
        end

        # --- a healthy fit trips neither ------------------------------------
        ok = fit_rem(seq, [Repetition(), Reciprocity(), SenderAttribute(icr), ReceiverAttribute(icr)];
                     n_controls=full)
        @test !ok.singular && isempty(ok.singular_suspects) && isempty(ok.separated)
        @test Networks.is_exact(ok)
        @test !occursin("Warning", sprint(show, ok))
        # the `_separated_columns` rule on the Newton increment: 1e-15 on a
        # healthy coordinate, O(1) on a separated one
        @test REM._separated_columns([0.1, -20.0], [1e-15, -1.0]) == [2]
        @test isempty(REM._separated_columns([0.1, 2.0], [1e-6, 1e-4]))
        @test REM._separated_columns([30.0], [0.5]) == [1]        # 0.5 > 0.01·30
        @test isempty(REM._separated_columns([30.0], [0.2]))
        # the standardising scales: the column sd, 1.0 for a constant column
        # or a single row
        @test REM._column_scales([1.0 5.0; 3.0 5.0; 5.0 5.0]) == [2.0, 1.0]
        @test REM._column_scales(zeros(1, 2)) == [1.0, 1.0]
        # β·s and δ·s rescale together: the verdict on a column is the same
        # whatever unit it is measured in
        Xs = [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.0 0.0]
        for c in (1.0, 1000.0, 0.001)
            sc = REM._column_scales(Xs .* [c 1.0])
            @test REM._separated_columns([-20.0 / c, 0.1] .* sc, [-1.0 / c, 1e-15] .* sc) == [1]
        end
        # the null-direction finder on a rank-deficient information matrix
        H = -[2.0 1.0 1.0; 1.0 1.0 0.0; 1.0 0.0 1.0]          # column 1 = column 2 + column 3
        @test REM._singular_columns(H) == [1, 2, 3]
        @test REM._singular_columns(-[1.0 0.0; 0.0 0.0]) == [2]
    end

    @testset "RecencyStatistic at Δ = 0: exp_decay is 1, the singular transforms are capped" begin
        # A prior event exists and the clock has not moved (an event tied with
        # the last one on the dyad under `ties=:ordered`, or a hand-built state
        # read at its last event's time): exp(−λ·0) = 1 is "just happened" —
        # not 0, which would mean "never happened" — while 1/Δ and 1/log(1+Δ)
        # are singular there and return the documented cap 0.0. Δ → 0⁺ is
        # continuous for exp_decay and diverges for the other two.
        st = EventNetworkState{Float64}(n_actors=3)
        update!(st, Event(1, 2, 1.0))
        st.current_time = 1.0
        @test compute(RecencyStatistic(transform=:exp_decay, decay=0.5), st, 1, 2) == 1.0
        @test compute(RecencyStatistic(transform=:exp_decay, decay=0.5, directed=false), st, 2, 1) == 1.0
        @test compute(RecencyStatistic(transform=:inverse), st, 1, 2) == 0.0
        @test compute(RecencyStatistic(transform=:inverse_log), st, 1, 2) == 0.0
        # no prior event: 0.0 under every transform
        @test compute(RecencyStatistic(transform=:exp_decay), st, 1, 3) == 0.0
        @test compute(RecencyStatistic(transform=:exp_decay), st, 2, 1) == 0.0   # directed
        @test compute(RecencyStatistic(), st, 1, 3) == 0.0
        st.current_time = 1.0 + 1e-9
        Δ = st.current_time - 1.0                        # ≈ 1e-9, as the floating clock has it
        @test compute(RecencyStatistic(transform=:exp_decay, decay=0.5), st, 1, 2) == exp(-0.5 * Δ)
        @test compute(RecencyStatistic(transform=:inverse), st, 1, 2) == 1 / Δ
        @test compute(RecencyStatistic(transform=:inverse_log), st, 1, 2) == 1 / log1p(Δ)
        # it reaches a fit: a dyad acting twice at one timestamp under
        # ties=:ordered reads recency 1 (exp_decay) or 0 (inverse) for the second
        seq = EventSequence([Event(1, 2, 1.0), Event(1, 2, 1.0), Event(2, 1, 2.0)]; actors=ActorSet(1:3))
        cs = compute_statistics(seq, [RecencyStatistic(transform=:exp_decay, decay=0.5), RecencyStatistic()])
        @test cs.recency_exp_decay == [0.0, 1.0, 0.0]
        @test cs.recency_inverse == [0.0, 0.0, 0.0]
        @test compute_statistics(seq, [RecencyStatistic(transform=:exp_decay, decay=0.5, directed=false)]).recency_exp_decay ==
              [0.0, 1.0, exp(-0.5)]
    end

    @testset "Calendar clocks take Dates.Period windows and halflives" begin
        # The natural idiom on a Date/DateTime clock — `window=Day(2)`,
        # `halflife_to_decay(Day(7))` — is accepted and converted to the
        # seconds the clock counts in, instead of failing with a TypeError
        # that never mentions seconds (and `window=2.0` on a Date clock being
        # two seconds, which expires everything)
        events = [Event(1, 2, Date(2024, 1, 1)), Event(2, 1, Date(2024, 1, 2)),
                  Event(1, 2, Date(2024, 1, 3)), Event(2, 1, Date(2024, 1, 6)),
                  Event(1, 2, Date(2024, 1, 7)), Event(1, 3, Date(2024, 1, 8))]
        dseq = EventSequence(events; actors=ActorSet(1:3))
        stats = [Repetition(), Reciprocity()]
        by_period = compute_statistics(dseq, stats; window=Day(2))
        by_seconds = compute_statistics(dseq, stats; window=2 * 86400.0)
        @test by_period.repetition == by_seconds.repetition == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        @test by_period.reciprocity == by_seconds.reciprocity == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        # (an event exactly `window` old still counts: 1→2 on Jan 1 is 2 days
        # old on Jan 3; the Jan 6 event sees nothing, Jan 7 sees Jan 6)
        two_seconds = compute_statistics(dseq, stats; window=2.0)
        @test all(==(0.0), two_seconds.repetition)        # the documented seconds unit
        @test halflife_to_decay(Day(7)) == halflife_to_decay(7 * 86400)
        @test halflife_to_decay(Hour(1)) == halflife_to_decay(3600)
        @test halflife_to_decay(Day(1) + Hour(12)) == halflife_to_decay(129600)
        @test_throws ArgumentError halflife_to_decay(Day(0))
        @test_throws ArgumentError halflife_to_decay(Day(-1))
        @test_throws ArgumentError EventNetworkState(dseq; window=Day(0))
        @test EventNetworkState(dseq; window=Day(2)).window == 172800.0
        @test EventNetworkState(dseq; window=Hour(36) + Minute(30)).window == 131400.0
        # the same on DateTime, through every entry point
        tseq = EventSequence([Event(e.sender, e.receiver, DateTime(e.time)) for e in events];
                             actors=ActorSet(1:3))
        @test compute_statistics(tseq, stats; window=Day(2)).repetition == by_period.repetition
        obs_p = generate_observations(dseq, stats, CaseControlSampler(n_controls=5, seed=1); window=Day(2))
        obs_s = generate_observations(dseq, stats, CaseControlSampler(n_controls=5, seed=1); window=172800)
        @test obs_p == obs_s
        fp = fit_rem(dseq, stats; n_controls=5, seed=1, window=Day(2))
        fs = fit_rem(dseq, stats; n_controls=5, seed=1, window=172800.0)
        @test coef(fp) == coef(fs)
        @test coef(fit_rem(dseq, stats; n_controls=5, seed=1, decay=halflife_to_decay(Day(2)))) ==
              coef(fit_rem(dseq, stats; n_controls=5, seed=1, decay=halflife_to_decay(172800.0)))
        @test size(control_draw_cov(dseq, stats; n_controls=5, n_draws=3, window=Day(2),
                                    rng=Xoshiro(1)).replicates) == (3, 2)
        # decay and window stay mutually exclusive whatever the window's type
        @test_throws ArgumentError fit_rem(dseq, stats; n_controls=5, window=Day(2),
                                           decay=halflife_to_decay(Day(2)))
        # A Period on a NUMERIC clock is a category error and is refused
        # (round 3): converted to 172 800 clock units it was silently no
        # window at all, and `fit_rem(...; window=Day(2))` returned the
        # no-window fit. Every entry point routes through the constructor.
        nseq = EventSequence([Event(e.sender, e.receiver, Float64(k)) for (k, e) in enumerate(events)];
                             actors=ActorSet(1:3))
        for f in (() -> EventNetworkState(nseq; window=Day(2)),
                  () -> EventNetworkState{Float64}(n_actors=3, window=Hour(1) + Minute(5)),
                  () -> EventNetworkState{Int}(n_actors=3, window=Day(2)),
                  () -> compute_statistics(nseq, stats; window=Day(2)),
                  () -> generate_observations(nseq, stats, CaseControlSampler(n_controls=5, seed=1); window=Day(2)),
                  () -> fit_rem(nseq, stats; n_controls=5, seed=1, window=Day(2)),
                  () -> control_draw_cov(nseq, stats; n_controls=5, n_draws=2, window=Day(2), rng=Xoshiro(1)))
            err = try f() catch e; e end
            @test err isa ArgumentError
            @test occursin("calendar period", err.msg) && occursin("clock units", err.msg)
            @test occursin("Date/DateTime", err.msg)
        end
        # ... while the number in clock units and no window at all are fine
        @test EventNetworkState(nseq; window=2.0).window == 2.0
        @test EventNetworkState(nseq; window=nothing).window == Inf
        @test coef(fit_rem(nseq, stats; n_controls=5, seed=1, window=2.0)) isa Vector{Float64}
    end

    @testset "update! is allocation-free on a warmed state" begin
        # The per-event absorb of the streaming design: 0 B once the dyad has
        # been seen (the eager `get!(d, k, Set{Int}())` used to build an empty
        # set on every call — 160 B per event); with the log or the window
        # FIFO kept the only growth is their amortized push!.
        rng = Random.Xoshiro(5)
        ev = Event{Float64}[]
        for k in 1:400
            s = rand(rng, 1:20); r = rand(rng, 1:19); r >= s && (r += 1)
            push!(ev, Event(s, r, Float64(k); weight=0.5 + rand(rng)))
        end
        sq = EventSequence(ev; actors=ActorSet(1:20))
        function warm(st)
            for e in sq; update!(st, e); end
            return st
        end
        alloc_update(st, e) = (update!(st, e); @allocated update!(st, e))
        plain = warm(EventNetworkState(sq; keep_history=false))
        @test alloc_update(plain, sq[1]) == 0
        @test alloc_update(plain, sq[200]) == 0
        decayed = warm(EventNetworkState(sq; keep_history=false, decay=0.01))
        @test alloc_update(decayed, sq[1]) == 0
        # the event log and the window FIFO grow by one tuple per event —
        # amortized O(1), and a small strata-independent budget per call
        logged = warm(EventNetworkState(sq; keep_history=true))
        @test alloc_update(logged, sq[1]) <= 64
        windowed = warm(EventNetworkState(sq; keep_history=false, window=50.0))
        @test alloc_update(windowed, sq[1]) <= 64
    end

    # ------------------------------------------------------------------------
    # WP2 (panel 2026-09): eventnet statistic parity
    # ------------------------------------------------------------------------

    @testset "eventnet triadic statistics: aggregation × weighted × decay (hand-computed)" begin
        # Counts (no decay): w(1,2) = 2, w(2,3) = 3, w(1,4) = 1, w(4,3) = 1
        ev = [Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0), Event(2, 3, 4.0),
              Event(2, 3, 5.0), Event(1, 4, 6.0), Event(4, 3, 7.0)]
        seq = EventSequence(ev; actors=ActorSet(1:4))
        st = EventNetworkState(seq)
        for e in seq; update!(st, e); end

        # TransitiveClosure(1→3): two-paths 1→2→3 with (2, 3) and 1→4→3 with (1, 1)
        #   min: 2 + 1 = 3   max: 3 + 1 = 4   sum: 5 + 2 = 7   product: 6 + 1 = 7
        #   count of third parties: 2
        expected = Dict(:min => 3.0, :max => 4.0, :sum => 7.0, :product => 7.0)
        for (agg, val) in expected
            @test compute(TransitiveClosure(aggregation=agg), st, 1, 3) == val
            # CyclicClosure(3→1): k with 1→k→3 — the same two two-paths
            @test compute(CyclicClosure(aggregation=agg), st, 3, 1) == val
            # CommonNeighbors(1, 3): common {2, 4}; u(1,2) = 2, u(2,3) = 3; u(1,4) = 1, u(4,3) = 1
            @test compute(CommonNeighbors(aggregation=agg), st, 1, 3) == val
        end
        @test compute(TransitiveClosure(), st, 1, 3) == 3.0          # default = eventnet's min
        @test TransitiveClosure().weighted && TransitiveClosure().aggregation === :min
        @test compute(TransitiveClosure(weighted=false), st, 1, 3) == 2.0
        @test compute(CyclicClosure(weighted=false), st, 3, 1) == 2.0
        @test compute(CommonNeighbors(weighted=false), st, 1, 3) == 2.0
        # No two-path 1→k→2: out(1) = {2, 4}, in(2) = {1}
        @test compute(TransitiveClosure(), st, 1, 2) == 0.0
        @test compute(TransitiveClosure(weighted=false), st, 1, 2) == 0.0
        @test compute(CyclicClosure(), st, 1, 3) == 0.0               # no 3→k→1

        # SharedSender(2, 4): k = 1 sent to both, w(1,2) = 2, w(1,4) = 1
        @test compute(SharedSender(), st, 2, 4) == 1.0
        @test compute(SharedSender(aggregation=:max), st, 2, 4) == 2.0
        @test compute(SharedSender(aggregation=:sum), st, 2, 4) == 3.0
        @test compute(SharedSender(aggregation=:product), st, 2, 4) == 2.0
        @test compute(SharedSender(weighted=false), st, 2, 4) == 1.0
        # SharedReceiver(2, 4): k = 3 received from both, w(2,3) = 3, w(4,3) = 1
        @test compute(SharedReceiver(), st, 2, 4) == 1.0
        @test compute(SharedReceiver(aggregation=:max), st, 2, 4) == 3.0
        @test compute(SharedReceiver(aggregation=:sum), st, 2, 4) == 4.0
        @test compute(SharedReceiver(aggregation=:product), st, 2, 4) == 3.0
        @test compute(SharedReceiver(weighted=false), st, 2, 4) == 1.0
        # CommonNeighbors(2, 4): k = 1 (u(2,1) = 2, u(1,4) = 1) and k = 3 (u(2,3) = 3, u(3,4) = 1)
        @test compute(CommonNeighbors(), st, 2, 4) == 2.0
        @test compute(CommonNeighbors(aggregation=:max), st, 2, 4) == 5.0
        @test compute(CommonNeighbors(aggregation=:sum), st, 2, 4) == 7.0
        @test compute(CommonNeighbors(aggregation=:product), st, 2, 4) == 5.0
        @test compute(CommonNeighbors(weighted=false), st, 2, 4) == 2.0
        # symmetric in (s, r)
        @test compute(CommonNeighbors(aggregation=:sum), st, 4, 2) == 7.0

        # A third party may not be the sender or receiver: a self-loop puts 1 in
        # out(1) ∩ in(1) and must not close 1→1
        st_self = EventNetworkState(seq)
        for e in seq; update!(st_self, e); end
        update!(st_self, Event(1, 1, 8.0))
        @test compute(TransitiveClosure(), st_self, 1, 1) == 0.0
        @test compute(TransitiveClosure(weighted=false), st_self, 1, 1) == 0.0
        # CommonNeighbors(1, 1): 1's neighbours {2, 4} (min(2, 2) + min(1, 1) = 3);
        # 1 itself — a neighbour through the self-loop, u(1,1) = 1 — is excluded
        # (it would add 1 otherwise)
        @test compute(CommonNeighbors(), st_self, 1, 1) == 3.0
        @test compute(CommonNeighbors(weighted=false), st_self, 1, 1) == 2.0

        # Event weights ≠ 1 enter the aggregation: 1→2 (2.5), 2→3 (0.5)
        stw = EventNetworkState{Float64}(n_actors=3)
        update!(stw, Event(1, 2, 1.0; weight=2.5))
        update!(stw, Event(2, 3, 2.0; weight=0.5))
        @test compute(TransitiveClosure(), stw, 1, 3) == 0.5
        @test compute(TransitiveClosure(aggregation=:max), stw, 1, 3) == 2.5
        @test compute(TransitiveClosure(aggregation=:sum), stw, 1, 3) == 3.0
        @test compute(TransitiveClosure(aggregation=:product), stw, 1, 3) == 1.25
        @test compute(TransitiveClosure(weighted=false), stw, 1, 3) == 1.0

        # halflife = 1: 1→2 at t = 0, 2→3 at t = 1. Read at t = 1 the first dyad
        # has decayed to 0.5 while the count of third parties stays 1.0
        std = EventNetworkState{Float64}(n_actors=3, decay=halflife_to_decay(1.0))
        update!(std, Event(1, 2, 0.0))
        update!(std, Event(2, 3, 1.0))
        @test get_dyad_count(std, 1, 2) ≈ 0.5
        @test compute(TransitiveClosure(), std, 1, 3) ≈ 0.5                    # min(0.5, 1)
        @test compute(TransitiveClosure(aggregation=:max), std, 1, 3) ≈ 1.0
        @test compute(TransitiveClosure(aggregation=:sum), std, 1, 3) ≈ 1.5
        @test compute(TransitiveClosure(aggregation=:product), std, 1, 3) ≈ 0.5
        @test compute(TransitiveClosure(weighted=false), std, 1, 3) == 1.0       # never decays
        @test compute(CommonNeighbors(), std, 1, 3) ≈ 0.5
        @test compute(CommonNeighbors(weighted=false), std, 1, 3) == 1.0
        std.current_time = 2.0                                                   # one more halflife
        @test compute(TransitiveClosure(), std, 1, 3) ≈ 0.25
        @test compute(TransitiveClosure(aggregation=:max), std, 1, 3) ≈ 0.5
        @test compute(TransitiveClosure(aggregation=:sum), std, 1, 3) ≈ 0.75
        @test compute(TransitiveClosure(aggregation=:product), std, 1, 3) ≈ 0.125
        @test compute(TransitiveClosure(weighted=false), std, 1, 3) == 1.0
        # the other three families, same decayed two-path
        stc = EventNetworkState{Float64}(n_actors=3, decay=halflife_to_decay(1.0))
        update!(stc, Event(3, 2, 0.0)); update!(stc, Event(2, 1, 1.0))         # 3→2→1 closes 1→3
        @test compute(CyclicClosure(), stc, 1, 3) ≈ 0.5
        @test compute(CyclicClosure(weighted=false), stc, 1, 3) == 1.0
        sts = EventNetworkState{Float64}(n_actors=3, decay=halflife_to_decay(1.0))
        update!(sts, Event(3, 1, 0.0)); update!(sts, Event(3, 2, 1.0))         # 3→1, 3→2
        @test compute(SharedSender(), sts, 1, 2) ≈ 0.5
        @test compute(SharedSender(aggregation=:sum), sts, 1, 2) ≈ 1.5
        @test compute(SharedSender(weighted=false), sts, 1, 2) == 1.0
        str = EventNetworkState{Float64}(n_actors=3, decay=halflife_to_decay(1.0))
        update!(str, Event(1, 3, 0.0)); update!(str, Event(2, 3, 1.0))         # 1→3, 2→3
        @test compute(SharedReceiver(), str, 1, 2) ≈ 0.5
        @test compute(SharedReceiver(aggregation=:product), str, 1, 2) ≈ 0.5
        @test compute(SharedReceiver(weighted=false), str, 1, 2) == 1.0

        # The vocabulary is checked at construction
        @test_throws ArgumentError TransitiveClosure(aggregation=:mean)
        @test_throws ArgumentError CommonNeighbors(aggregation=:avg)
        @test_throws ArgumentError FourCycle(aggregation=:mean)
        err = try TransitiveClosure(aggregation=:mean) catch e; e end
        @test occursin(":min, :max, :sum or :product", err.msg)
    end

    @testset "FourCycle: all five cycle types on a 4-actor example (hand-computed)" begin
        # 1→2, 3→2 (twice), 3→4, 2→1, 2→3, 4→3: for the dyad (1, 4) every pattern
        # closes through j = 2, k = 3 exactly once
        ev = [Event(1, 2, 1.0), Event(3, 2, 2.0), Event(3, 2, 3.0), Event(3, 4, 4.0),
              Event(2, 1, 5.0), Event(2, 3, 6.0), Event(4, 3, 7.0)]
        seq = EventSequence(ev; actors=ActorSet(1:5))
        st = EventNetworkState(seq)
        for e in seq; update!(st, e); end
        #   :out_out  s→j←k→r  weights (w12, w32, w34) = (1, 2, 1)
        #   :in_in    s←j→k←r  weights (w21, w23, w43) = (1, 1, 1)
        #   :out_in   s→j→k→r  weights (w12, w23, w34) = (1, 1, 1)
        #   :in_out   s←j←k←r  weights (w21, w32, w43) = (1, 2, 1)
        for ct in (:out_out, :in_in, :out_in, :in_out)
            @test compute(FourCycle(cycle_type=ct, weighted=false), st, 1, 4) == 1.0
            @test compute(FourCycle(cycle_type=ct), st, 1, 4) == 1.0              # min
        end
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:max), st, 1, 4) == 2.0
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:sum), st, 1, 4) == 4.0
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:product), st, 1, 4) == 2.0
        @test compute(FourCycle(cycle_type=:in_out, aggregation=:sum), st, 1, 4) == 4.0
        @test compute(FourCycle(cycle_type=:in_out, aggregation=:product), st, 1, 4) == 2.0
        @test compute(FourCycle(cycle_type=:out_in, aggregation=:sum), st, 1, 4) == 3.0
        @test compute(FourCycle(cycle_type=:in_in, aggregation=:sum), st, 1, 4) == 3.0
        # :mixed is the sum of the four
        @test compute(FourCycle(cycle_type=:mixed, weighted=false), st, 1, 4) == 4.0
        @test compute(FourCycle(cycle_type=:mixed), st, 1, 4) == 4.0
        @test compute(FourCycle(cycle_type=:mixed, aggregation=:sum), st, 1, 4) == 14.0
        @test compute(FourCycle(cycle_type=:mixed, aggregation=:product), st, 1, 4) == 6.0
        # the reverse dyad (4, 1) closes :out_out through 4→3←2→1 (weights 1, 1, 1) …
        @test compute(FourCycle(cycle_type=:out_out), st, 4, 1) == 1.0
        # … and (2, 4) closes nothing: every candidate j/k is 2 or 4 itself
        for ct in (:out_out, :in_in, :out_in, :in_out, :mixed)
            @test compute(FourCycle(cycle_type=ct), st, 2, 4) == 0.0
            @test compute(FourCycle(cycle_type=ct, weighted=false), st, 2, 4) == 0.0
        end
        @test FourCycle().weighted && FourCycle().aggregation === :min
        @test name(FourCycle(cycle_type=:in_in)) == "four_cycle_in_in"

        # A second parallel three-path (1→5←3, 3→4 already there) adds to :out_out
        update!(st, Event(1, 5, 8.0)); update!(st, Event(3, 5, 9.0))
        @test compute(FourCycle(cycle_type=:out_out, weighted=false), st, 1, 4) == 2.0
        @test compute(FourCycle(cycle_type=:out_out), st, 1, 4) == 2.0
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:sum), st, 1, 4) == 4.0 + 3.0

        # Geometrically weighted four-cycles: e^α (1 − (1 − e^{−α})^n), n = 2 here
        α = 0.5
        gw(n) = exp(α) * (1 - (1 - exp(-α))^n)
        @test compute(GeometricWeightedFourCycles(cycle_type=:out_out, alpha=α), st, 1, 4) ≈ gw(2)
        @test compute(GeometricWeightedFourCycles(cycle_type=:in_in, alpha=α), st, 1, 4) ≈ gw(1)
        @test compute(GeometricWeightedFourCycles(cycle_type=:mixed, alpha=α), st, 1, 4) ≈ gw(5)
        @test compute(GeometricWeightedFourCycles(cycle_type=:out_out, alpha=α), st, 4, 1) ≈ gw(1)
        @test compute(GeometricWeightedFourCycles(cycle_type=:out_out, alpha=α), st, 2, 4) == 0.0
        @test_throws ArgumentError GeometricWeightedFourCycles(cycle_type=:diagonal)
        @test_throws ArgumentError GeometricWeightedFourCycles(alpha=0.0)

        # Decay applies to the three weights, not to the count
        std = EventNetworkState{Float64}(n_actors=4, decay=halflife_to_decay(1.0))
        update!(std, Event(1, 2, 0.0)); update!(std, Event(3, 2, 1.0)); update!(std, Event(3, 4, 2.0))
        # at t = 2: w12 = 0.25, w32 = 0.5, w34 = 1
        @test compute(FourCycle(cycle_type=:out_out), std, 1, 4) ≈ 0.25
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:max), std, 1, 4) ≈ 1.0
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:sum), std, 1, 4) ≈ 1.75
        @test compute(FourCycle(cycle_type=:out_out, aggregation=:product), std, 1, 4) ≈ 0.125
        @test compute(FourCycle(cycle_type=:out_out, weighted=false), std, 1, 4) == 1.0
        @test compute(GeometricWeightedFourCycles(cycle_type=:out_out, alpha=α), std, 1, 4) ≈ gw(1)
    end

    @testset "GeometricWeightedTriads: e^α(1 − (1 − e^{−α})^n) for n = 0..3, all closure types" begin
        α = 0.5
        gw(n) = exp(α) * (1 - (1 - exp(-α))^n)
        @test gw(0) == 0.0 && gw(1) ≈ 1.0
        # For (s, r) = (1, 5) build m third parties k ∈ {2, 3, 4} closing each pattern
        pattern = Dict(
            :transitive     => k -> (Event(1, k, 1.0), Event(k, 5, 2.0)),   # 1→k→5
            :cyclic         => k -> (Event(5, k, 1.0), Event(k, 1, 2.0)),   # 5→k→1
            :shared_sender  => k -> (Event(k, 1, 1.0), Event(k, 5, 2.0)),   # k→1, k→5
            :shared_receiver=> k -> (Event(1, k, 1.0), Event(5, k, 2.0)))   # 1→k, 5→k
        for (ct, mk) in pattern, m in 0:3
            st = EventNetworkState{Float64}(n_actors=5)
            for k in 2:(1 + m), e in mk(k)
                update!(st, e)
            end
            @test compute(GeometricWeightedTriads(closure_type=ct, alpha=α), st, 1, 5) ≈ gw(m)
            # repeated events on the same two-path do not change n
            m > 0 && (update!(st, mk(2)[1]); update!(st, mk(2)[2]))
            @test compute(GeometricWeightedTriads(closure_type=ct, alpha=α), st, 1, 5) ≈ gw(m)
        end
        @test name(GeometricWeightedTriads(closure_type=:cyclic)) == "gw_cyclic"
        @test_throws ArgumentError GeometricWeightedTriads(closure_type=:star)
        @test_throws ArgumentError GeometricWeightedTriads(alpha=-1.0)
    end

    @testset "Dyad and degree statistics (hand-computed, weights ≠ 1, Int/Date/DateTime clocks)" begin
        # --- Inertia ---
        st = EventNetworkState{Float64}(n_actors=3)
        update!(st, Event(1, 2, 1.0)); update!(st, Event(1, 2, 2.0)); update!(st, Event(2, 1, 3.0))
        @test compute(InertiaStatistic(), st, 1, 2) == 3.0                             # 2 + 1
        @test compute(InertiaStatistic(repetition_weight=2.0, reciprocity_weight=0.5), st, 1, 2) == 4.5
        @test compute(InertiaStatistic(), st, 2, 1) == 3.0
        @test compute(InertiaStatistic(), st, 1, 3) == 0.0
        @test name(InertiaStatistic()) == "inertia"

        # --- Recency: last 1→2 at t = 2, last 2→1 at t = 3, read at t = 5 ---
        st.current_time = 5.0
        @test compute(RecencyStatistic(), st, 1, 2) == 1 / 3
        @test compute(RecencyStatistic(), st, 2, 1) == 1 / 2
        @test compute(RecencyStatistic(), st, 1, 3) == 0.0                            # no prior event
        # undirected: the LATER of the two directions (was the (min, max) dyad only)
        @test compute(RecencyStatistic(directed=false), st, 1, 2) == 1 / 2
        @test compute(RecencyStatistic(directed=false), st, 2, 1) == 1 / 2
        @test compute(RecencyStatistic(transform=:inverse_log), st, 1, 2) ≈ 1 / log(4)
        @test compute(RecencyStatistic(transform=:exp_decay, decay=0.5), st, 1, 2) ≈ exp(-1.5)
        @test compute(RecencyStatistic(transform=:exp_decay, decay=0.5), st, 2, 1) ≈ exp(-1.0)
        @test name(RecencyStatistic(transform=:inverse_log)) == "recency_inverse_log"
        st.current_time = 3.0                                                        # tied with the last 2→1
        @test compute(RecencyStatistic(), st, 2, 1) == 0.0
        # only one direction observed: undirected reads it whichever it is
        st1 = EventNetworkState{Float64}(n_actors=2)
        update!(st1, Event(2, 1, 1.0)); st1.current_time = 3.0
        @test compute(RecencyStatistic(directed=false), st1, 1, 2) == 1 / 2
        @test compute(RecencyStatistic(directed=true), st1, 1, 2) == 0.0
        # Int clock
        sti = EventNetworkState{Int}(n_actors=2)
        update!(sti, Event(1, 2, 1)); update!(sti, Event(2, 1, 3)); sti.current_time = 5
        @test compute(RecencyStatistic(), sti, 1, 2) == 1 / 4
        @test compute(RecencyStatistic(directed=false), sti, 1, 2) == 1 / 2
        @test compute(Repetition(), sti, 1, 2) == 1.0
        # Date clock (seconds): 4 days = 345600 s
        stdt = EventNetworkState{Date}(n_actors=2)
        update!(stdt, Event(1, 2, Date(2024, 1, 1))); update!(stdt, Event(2, 1, Date(2024, 1, 3)))
        stdt.current_time = Date(2024, 1, 5)
        @test compute(RecencyStatistic(), stdt, 1, 2) ≈ 1 / 345600
        @test compute(RecencyStatistic(directed=false), stdt, 1, 2) ≈ 1 / 172800
        @test compute(RecencyStatistic(transform=:exp_decay, decay=1 / 86400), stdt, 1, 2) ≈ exp(-4.0)
        # DateTime clock: one hour = 3600 s
        stdd = EventNetworkState{DateTime}(n_actors=2)
        update!(stdd, Event(1, 2, DateTime(2024, 1, 1, 0, 0)))
        stdd.current_time = DateTime(2024, 1, 1, 1, 0)
        @test compute(RecencyStatistic(), stdd, 1, 2) ≈ 1 / 3600
        @test compute(RecencyStatistic(transform=:inverse_log), stdd, 1, 2) ≈ 1 / log1p(3600)

        # --- DyadCovariate ---
        dc = DyadCovariate(Dict((1, 2) => 10.0, (2, 1) => 12.0); default=100.0, name="distance")
        @test compute(dc, st, 1, 2) == 10.0
        @test compute(dc, st, 2, 1) == 12.0
        @test compute(dc, st, 1, 3) == 100.0
        @test name(dc) == "distance"

        # --- Degrees with event weights: 1→2 (1), 1→3 (2), 2→1 (1), 3→1 (3) ---
        sd = EventNetworkState{Float64}(n_actors=3)
        update!(sd, Event(1, 2, 1.0; weight=1.0)); update!(sd, Event(1, 3, 2.0; weight=2.0))
        update!(sd, Event(2, 1, 3.0; weight=1.0)); update!(sd, Event(3, 1, 4.0; weight=3.0))
        # out: 1 ↦ 3, 2 ↦ 1, 3 ↦ 3;  in: 1 ↦ 4, 2 ↦ 1, 3 ↦ 2
        @test compute(SenderActivity(), sd, 1, 2) == 3.0
        @test compute(ReceiverActivity(), sd, 2, 1) == 3.0        # receiver 1's out-degree
        @test compute(SenderPopularity(), sd, 1, 2) == 4.0        # sender 1's in-degree
        @test compute(ReceiverPopularity(), sd, 1, 2) == 1.0
        @test compute(ReceiverPopularity(), sd, 2, 3) == 2.0
        @test compute(TotalDegree(role=:sender), sd, 1, 2) == 7.0
        @test compute(TotalDegree(role=:receiver), sd, 1, 2) == 2.0
        @test compute(DegreeDifference(), sd, 1, 2) == 2.0                      # out: 3 − 1
        @test compute(DegreeDifference(degree_type=:in), sd, 1, 2) == 3.0       # in: 4 − 1
        @test compute(DegreeDifference(degree_type=:total), sd, 1, 2) == 5.0    # 7 − 2
        @test compute(DegreeDifference(), sd, 2, 1) == -2.0
        @test compute(DegreeDifference(absolute=true), sd, 2, 1) == 2.0
        @test compute(DegreeDifference(degree_type=:total, absolute=true), sd, 2, 1) == 5.0
        @test compute(LogDegree(), sd, 1, 2) ≈ log1p(3.0)
        @test compute(LogDegree(role=:receiver, degree_type=:in), sd, 1, 2) ≈ log1p(1.0)
        @test compute(LogDegree(role=:sender, degree_type=:total), sd, 1, 2) ≈ log1p(7.0)
        @test compute(LogDegree(role=:receiver, degree_type=:out), sd, 3, 2) ≈ log1p(1.0)
        @test name(DegreeDifference(degree_type=:in, absolute=true)) == "degree_diff_in_abs"
        @test name(LogDegree(role=:receiver, degree_type=:in)) == "log_receiver_in_degree"
        @test_throws ArgumentError TotalDegree(role=:both)
        @test_throws ArgumentError DegreeDifference(degree_type=:all)
        @test_throws ArgumentError LogDegree(degree_type=:all)
        # weighted dyad statistics on the same state
        @test compute(Repetition(), sd, 1, 3) == 2.0
        @test compute(Reciprocity(), sd, 1, 3) == 3.0
        @test compute(Repetition(directed=false), sd, 1, 3) == 5.0
        @test compute(InertiaStatistic(), sd, 1, 3) == 5.0
        # Date clock, weights, degrees
        sdd = EventNetworkState{Date}(n_actors=2)
        update!(sdd, Event(1, 2, Date(2024, 1, 1); weight=0.5)); update!(sdd, Event(1, 2, Date(2024, 1, 2); weight=0.25))
        @test compute(SenderActivity(), sdd, 1, 2) == 0.75
        @test compute(Repetition(), sdd, 1, 2) == 0.75
    end

    @testset "Node attribute statistics (hand-computed)" begin
        gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F", 3 => "M"))
        age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0, 3 => 40.0))
        years = NodeAttribute(:years, Dict(1 => 2, 2 => 5, 3 => 9))            # Int-valued
        st = EventNetworkState{Float64}(n_actors=3)
        @test compute(ActorMix(gender, "M", "F"), st, 1, 2) == 1.0
        @test compute(ActorMix(gender, "M", "F"), st, 2, 1) == 0.0
        @test compute(ActorMix(gender, "M", "M"), st, 1, 3) == 1.0
        @test name(ActorMix(gender, "M", "F")) == "mix_gender_M_F"
        @test compute(NodeSum(age), st, 1, 2) == 55.0
        @test compute(NodeSum(years), st, 1, 3) == 11.0
        @test compute(NodeProduct(age), st, 1, 2) == 750.0
        @test compute(NodeProduct(years), st, 2, 3) == 45.0
        @test compute(NodeDifference(years), st, 3, 1) == 7.0
        @test compute(NodeDifference(years; absolute=true), st, 1, 3) == 7.0
        @test compute(SenderAttribute(age), st, 1, 2) == 25.0
        @test compute(ReceiverAttribute(age), st, 1, 2) == 30.0
        @test compute(SenderAttribute(years), st, 3, 1) == 9.0
        @test compute(SenderCategorical(gender, "M"), st, 1, 2) == 1.0
        @test compute(SenderCategorical(gender, "M"), st, 2, 1) == 0.0
        @test compute(ReceiverCategorical(gender, "F"), st, 1, 2) == 1.0
        @test compute(ReceiverCategorical(gender, "F"), st, 2, 3) == 0.0
        @test name(SenderCategorical(gender, "M")) == "sender_gender_M"
        @test name(NodeProduct(age)) == "product_age"
        @test name(NodeSum(age; name="joint_age")) == "joint_age"
        # attribute values that were written after construction
        age[4] = 50.0
        @test compute(NodeSum(age), st, 1, 4) == 75.0
        # an Int value for a Float64 attribute is coerced exactly
        @test compute(ActorMix(NodeAttribute(:x, Dict(1 => 1.0, 2 => 2.0)), 1, 2), st, 1, 2) == 1.0
        @test compute(SenderCategorical(age, 25), st, 1, 2) == 1.0
    end

    @testset "Sliding window: EventNetworkState(window=) (hand-computed)" begin
        # 1→2 (t = 0, w = 1), 2→1 (t = 1, w = 2), 1→2 (t = 2.5, w = 1), 1→3 (t = 4, w = 1); window = 2
        ev = [Event(1, 2, 0.0), Event(2, 1, 1.0; weight=2.0), Event(1, 2, 2.5), Event(1, 3, 4.0)]
        seq = EventSequence(ev; actors=ActorSet(1:3))
        st = EventNetworkState(seq; window=2.0)
        @test st.window == 2.0
        for e in seq; update!(st, e); end
        # at t = 4: the events at t = 0 and t = 1 are older than 2 and have expired
        @test get_dyad_count(st, 1, 2) == 1.0
        @test get_dyad_count(st, 2, 1) == 0.0
        @test !has_edge(st, 2, 1)
        @test has_edge(st, 1, 2)
        @test get_undirected_count(st, 1, 2) == 1.0
        @test get_out_degree(st, 1) == 2.0
        @test get_in_degree(st, 1) == 0.0
        @test get_in_degree(st, 2) == 1.0
        @test get_out_degree(st, 2) == 0.0
        @test Set(get_out_neighbors(st, 1)) == Set([2, 3])
        @test isempty(get_in_neighbors(st, 1))
        @test isempty(get_out_neighbors(st, 2))
        @test compute(Repetition(), st, 1, 2) == 1.0
        @test compute(Reciprocity(), st, 1, 2) == 0.0
        @test compute(SenderActivity(), st, 1, 2) == 2.0
        # the last-event time is NOT windowed: recency is about the clock
        @test compute(RecencyStatistic(), st, 2, 1) == 1 / 3
        # an event exactly `window` old still counts …
        st.current_time = 4.5
        @test get_dyad_count(st, 1, 2) == 1.0
        # … one instant later it does not
        st.current_time = 4.6
        @test get_dyad_count(st, 1, 2) == 0.0
        @test !has_edge(st, 1, 2)
        @test Set(get_out_neighbors(st, 1)) == Set([3])
        @test get_out_degree(st, 1) == 1.0
        @test compute(SenderActivity(), st, 1, 3) == 1.0
        st.current_time = 7.0
        @test get_out_degree(st, 1) == 0.0
        @test isempty(get_out_neighbors(st, 1))
        @test isempty(st.dyad_counts) && isempty(st.out_degree) && isempty(st.live_dyad)
        # reset! clears the window bookkeeping too
        reset!(st)
        @test isempty(st.window_log) && st.window_head == 1 && isempty(st.live_out)

        # Fractional weights expire to an EXACT zero (the key is deleted, not
        # left at a floating-point residue)
        sw = EventNetworkState{Float64}(n_actors=2, window=1.0)
        update!(sw, Event(1, 2, 0.0; weight=0.1)); update!(sw, Event(1, 2, 0.5; weight=0.2))
        update!(sw, Event(1, 2, 0.9; weight=0.3))
        @test get_dyad_count(sw, 1, 2) ≈ 0.6
        sw.current_time = 1.6                          # the 0.1 and 0.2 events are gone
        @test get_dyad_count(sw, 1, 2) ≈ 0.3
        sw.current_time = 2.0
        @test get_dyad_count(sw, 1, 2) === 0.0
        @test !haskey(sw.dyad_counts, (1, 2)) && !haskey(sw.out_degree, 1) && !haskey(sw.in_degree, 2)
        @test !haskey(sw.undirected_counts, (1, 2))

        # Triadic statistics under a window: adjacency expires with the count
        tw = EventNetworkState{Float64}(n_actors=3, window=1.5)
        update!(tw, Event(1, 2, 0.0)); update!(tw, Event(2, 3, 1.0))
        @test compute(TransitiveClosure(), tw, 1, 3) == 1.0
        @test compute(TransitiveClosure(weighted=false), tw, 1, 3) == 1.0
        tw.current_time = 2.0                          # 1→2 is 2.0 old: gone
        @test compute(TransitiveClosure(), tw, 1, 3) == 0.0
        @test compute(TransitiveClosure(weighted=false), tw, 1, 3) == 0.0
        @test compute(CommonNeighbors(weighted=false), tw, 1, 3) == 0.0

        # The FIFO holds O(live events): 2000 events, ~10 in the window
        big = EventNetworkState{Float64}(n_actors=5, window=10.0)
        for k in 1:2000
            update!(big, Event(mod1(k, 5), mod1(k + 1, 5), Float64(k)))
        end
        @test length(big.window_log) - big.window_head + 1 == 11      # t ∈ [1990, 2000]
        @test length(big.window_log) <= 64
        @test get_out_degree(big, mod1(2000, 5)) == 3.0                 # k = 1990, 1995, 2000
        @test sum(get_out_degree(big, a) for a in 1:5) == 11.0

        # window = Inf is no window: identical to the plain state row for row
        # on the rem_clogit inputs
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_clogit.toml"))
        times = Float64.(g.values["input_time"])
        senders = Int.(g.values["input_sender"]); receivers = Int.(g.values["input_receiver"])
        gseq = EventSequence([Event(senders[i], receivers[i], times[i]) for i in eachindex(times)];
                             actors=ActorSet(1:Int(g.values["n_actors"])))
        gstats = [Repetition(), Reciprocity(), SenderActivity(), ReceiverPopularity(),
                  TransitiveClosure(), CommonNeighbors(), FourCycle()]
        sampler = CaseControlSampler(n_controls=20, seed=7)
        obs_plain = generate_observations(gseq, gstats, sampler)
        obs_inf = generate_observations(gseq, gstats, sampler; window=Inf)
        @test obs_plain == obs_inf
        @test compute_statistics(gseq, gstats) == compute_statistics(gseq, gstats; window=Inf)
        # a window wide enough to hold the whole sequence is also identical …
        @test compute_statistics(gseq, gstats; window=1e6) == compute_statistics(gseq, gstats)
        # … and a short one is not
        obs_short = generate_observations(gseq, gstats, sampler; window=0.05)
        @test obs_short.repetition != obs_plain.repetition
        @test all(obs_short.repetition .<= obs_plain.repetition)
        # a windowed fit runs, and reports as exact as any other full-risk-set fit
        fw = fit_rem(gseq, [Repetition(), Reciprocity()]; n_controls=89, window=0.5, tol=1e-10)
        @test fw.converged
        f0 = fit_rem(gseq, [Repetition(), Reciprocity()]; n_controls=89, tol=1e-10)
        @test coef(fw) != coef(f0)
        # control_draw_cov threads the window through the redraws
        dw = control_draw_cov(gseq, [Repetition(), Reciprocity()]; n_controls=20, window=0.5,
                              n_draws=4, rng=Xoshiro(3))
        @test all(isfinite, dw.sd) && size(dw.replicates) == (4, 2)
        d0 = control_draw_cov(gseq, [Repetition(), Reciprocity()]; n_controls=20,
                              n_draws=4, rng=Xoshiro(3))
        @test dw.replicates != d0.replicates              # same seeds, windowed design
        # compute_statistics(window=) agrees with a hand-windowed count
        cs = compute_statistics(gseq, [Repetition()]; window=0.5)
        hand = [count(k -> senders[k] == senders[m] && receivers[k] == receivers[m] &&
                            times[m] - times[k] <= 0.5, 1:(m - 1)) for m in eachindex(times)]
        @test cs.repetition == Float64.(hand)

        # decay and window are mutually exclusive, everywhere
        @test_throws ArgumentError EventNetworkState(gseq; decay=0.1, window=1.0)
        @test_throws ArgumentError generate_observations(gseq, gstats, sampler; decay=0.1, window=1.0)
        @test_throws ArgumentError compute_statistics(gseq, gstats; decay=0.1, window=1.0)
        @test_throws ArgumentError fit_rem(gseq, gstats; decay=0.1, window=1.0)
        err = try fit_rem(gseq, gstats; decay=0.1, window=1.0) catch e; e end
        @test occursin("eventnet offers no window", err.msg)
        @test_throws ArgumentError EventNetworkState(gseq; window=0.0)
        @test_throws ArgumentError EventNetworkState(gseq; window=-1.0)
        @test_throws ArgumentError EventNetworkState(gseq; decay=-0.1)
        # decay = 0 with a window is fine (0 is "no decay")
        @test EventNetworkState(gseq; decay=0.0, window=1.0).window == 1.0
    end

    @testset "Calendar timelines: window in seconds" begin
        # DateTime: events one hour apart, window = 3600 s
        evd = [Event(1, 2, DateTime(2024, 1, 1, 0, 0)), Event(1, 2, DateTime(2024, 1, 1, 1, 0)),
               Event(2, 1, DateTime(2024, 1, 1, 2, 0))]
        sq = EventSequence(evd; actors=ActorSet(1:2))
        st = EventNetworkState(sq; window=3600)
        for e in sq; update!(st, e); end
        # at 02:00 the 00:00 event is 7200 s old (expired), the 01:00 one exactly 3600 s (counts)
        @test get_dyad_count(st, 1, 2) == 1.0
        @test get_out_degree(st, 1) == 1.0
        st.current_time = DateTime(2024, 1, 1, 2, 0, 1)
        @test get_dyad_count(st, 1, 2) == 0.0
        @test !has_edge(st, 1, 2)
        @test get_dyad_count(st, 2, 1) == 1.0
        df = compute_statistics(sq, [Repetition()]; window=3600)
        @test df.repetition == [0.0, 1.0, 0.0]
        df2 = compute_statistics(sq, [Repetition()]; window=3599)
        @test df2.repetition == [0.0, 0.0, 0.0]
        # Date: window of two days = 172800 s
        evD = [Event(1, 2, Date(2024, 1, 1)), Event(1, 2, Date(2024, 1, 2)), Event(1, 2, Date(2024, 1, 5))]
        sqD = EventSequence(evD; actors=ActorSet(1:2))
        dfD = compute_statistics(sqD, [Repetition(), SenderActivity()]; window=2 * 86400)
        @test dfD.repetition == [0.0, 1.0, 0.0]         # on Jan 5 both earlier events are > 2 days old
        @test dfD.sender_activity == [0.0, 1.0, 0.0]
        stD = EventNetworkState(sqD; window=3 * 86400)
        for e in sqD; update!(stD, e); end
        @test get_dyad_count(stD, 1, 2) == 2.0          # Jan 2 (exactly 3 days old) and Jan 5 survive a 3-day window
        stD4 = EventNetworkState(sqD; window=4 * 86400)
        for e in sqD; update!(stD4, e); end
        @test get_dyad_count(stD4, 1, 2) == 3.0         # Jan 1 is exactly 4 days old: still counts
        # a fit on a calendar clock with a window
        evF = [Event(mod1(k, 4), mod1(k + 1, 4), DateTime(2024, 1, 1) + Hour(k)) for k in 1:40]
        sqF = EventSequence(evF; actors=ActorSet(1:5))
        fF = fit_rem(sqF, [Repetition()]; n_controls=19, window=5 * 3600, tol=1e-10)
        @test fF.converged
    end

    @testset "Allocation-free statistics (item 26): 0 B per compute on a warmed 30-actor state" begin
        rng = Random.Xoshiro(11)
        ev = Event{Float64}[]
        for k in 1:600
            s = rand(rng, 1:30); r = rand(rng, 1:29); r >= s && (r += 1)
            push!(ev, Event(s, r, Float64(k); weight=0.5 + rand(rng)))
        end
        sq = EventSequence(ev; actors=ActorSet(1:30))
        gender = NodeAttribute(:g, Dict(i => (isodd(i) ? "M" : "F") for i in 1:30))
        age = NodeAttribute(:a, Dict(i => Float64(i) for i in 1:30))
        every = AbstractStatistic[
            Repetition(), Repetition(directed=false), Reciprocity(), InertiaStatistic(),
            RecencyStatistic(), RecencyStatistic(directed=false, transform=:inverse_log),
            RecencyStatistic(transform=:exp_decay, decay=0.1),
            DyadCovariate(Dict((1, 2) => 1.0)),
            SenderActivity(), ReceiverActivity(), SenderPopularity(), ReceiverPopularity(),
            TotalDegree(), TotalDegree(role=:receiver), DegreeDifference(),
            DegreeDifference(degree_type=:in, absolute=true), DegreeDifference(degree_type=:total),
            LogDegree(), LogDegree(role=:receiver, degree_type=:total),
            TransitiveClosure(), TransitiveClosure(weighted=false), TransitiveClosure(aggregation=:sum),
            CyclicClosure(), CyclicClosure(aggregation=:max), SharedSender(),
            SharedSender(aggregation=:product), SharedReceiver(), SharedReceiver(weighted=false),
            CommonNeighbors(), CommonNeighbors(weighted=false), CommonNeighbors(aggregation=:sum),
            GeometricWeightedTriads(), GeometricWeightedTriads(closure_type=:cyclic),
            GeometricWeightedTriads(closure_type=:shared_sender),
            GeometricWeightedTriads(closure_type=:shared_receiver),
            FourCycle(), FourCycle(cycle_type=:in_in), FourCycle(cycle_type=:out_in),
            FourCycle(cycle_type=:in_out), FourCycle(cycle_type=:mixed), FourCycle(weighted=false),
            FourCycle(aggregation=:sum), GeometricWeightedFourCycles(),
            GeometricWeightedFourCycles(cycle_type=:mixed),
            AttributeMatch(gender), ActorMix(gender, "M", "F"), NodeDifference(age),
            NodeDifference(age; absolute=true), NodeSum(age), NodeProduct(age),
            SenderAttribute(age), ReceiverAttribute(age), SenderCategorical(gender, "M"),
            ReceiverCategorical(gender, "F")]
        # every exported statistic type is in the list
        exported_stats = [T for nm in names(REM) for T in (getfield(REM, nm),)
                          if T isa DataType && T <: AbstractStatistic && isconcretetype(T) ||
                             T isa UnionAll && T.body isa DataType && T.body <: AbstractStatistic]
        for T in exported_stats
            @test any(s -> s isa T, every)
        end
        for (label, st) in (("decay", EventNetworkState(sq; decay=0.01)),
                            ("window", EventNetworkState(sq; window=200.0)),
                            ("plain", EventNetworkState(sq)))
            for e in sq; update!(st, e); end
            @test !isempty(get_out_neighbors(st, 3))
            for stat in every
                a = _alloc_compute(stat, st, 3, 7)
                @test a == 0
                a == 0 || println(stderr, "$(name(stat)) allocates $a B on the $label state")
                # a dyad with nothing behind it and one with a self-neighbourless actor
                @test _alloc_compute(stat, st, 30, 29) == 0
            end
            ss = StatisticSet(every)
            dest = zeros(length(ss))
            compute_all!(dest, ss, st, 3, 7)
            @test @allocated(compute_all!(dest, ss, st, 3, 7)) == 0
        end
        # the neighbour accessors and the shared empty set
        st = EventNetworkState(sq); for e in sq; update!(st, e); end
        @test get_out_neighbors(st, 31) === REM._EMPTY_NEIGHBORS
        @test get_in_neighbors(st, 31) === REM._EMPTY_NEIGHBORS
        @test isempty(REM._EMPTY_NEIGHBORS) && length(REM._EMPTY_NEIGHBORS) == 0
        @test !(3 in REM._EMPTY_NEIGHBORS)
        @test collect(REM._EMPTY_NEIGHBORS) == Int[]
        @test REM._EMPTY_NEIGHBORS isa AbstractSet{Int}
        @test_throws ArgumentError push!(get_out_neighbors(st, 31), 1)      # immutable: cannot be corrupted
        @test copy(get_out_neighbors(st, 31)) == Set{Int}()
        @test intersect(get_out_neighbors(st, 31), Set([1, 2])) == Set{Int}()
        @test union(get_out_neighbors(st, 31), Set([1])) == Set([1])
        @test isempty(REM.get_common_senders(st, 31, 3))
        @test REM.get_common_receivers(st, 3, 7) == intersect(get_out_neighbors(st, 3), get_out_neighbors(st, 7))
        # the intersection helpers agree with `intersect`
        a, b = get_out_neighbors(st, 3), get_in_neighbors(st, 7)
        @test REM._count_common(a, b, 3, 7) == length(setdiff(intersect(a, b), (3, 7)))
        @test REM._sum_common(k -> 2.0, a, b, 3, 7) == 2.0 * REM._count_common(a, b, 3, 7)
        @test REM._count_common(a, REM._EMPTY_NEIGHBORS, 3, 7) == 0
        @test REM._sum_common(k -> 1.0, REM._EMPTY_NEIGHBORS, b, 3, 7) == 0.0
    end

    @testset "Golden: eventnet statistics vs R survival::clogit (min/max/sum/product, decay, window)" begin
        # rem_clogit.toml pins the count statistics; this fixture pins what makes
        # REM a port of eventnet — the weighted triadic family under each
        # aggregation, the undirected repetition, halflife-decayed counts and
        # sliding-window counts — every column rebuilt from the raw edgelist in
        # plain R (test/fixtures/r/rem_eventnet.R), full risk set on both sides.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_eventnet.toml"))
        @test g.provenance["script"] == "test/fixtures/r/rem_eventnet.R"
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end
        n = Int(g.values["n_actors"])
        times = Float64.(g.values["input_time"])
        senders = Int.(g.values["input_sender"]); receivers = Int.(g.values["input_receiver"])
        seq = EventSequence([Event(senders[i], receivers[i], times[i]) for i in eachindex(times)];
                            actors=ActorSet(1:n))
        max_controls = n * (n - 1) - 1
        # the same sequence as rem_clogit.toml
        gc = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_clogit.toml"))
        @test Float64.(gc.values["input_time"]) == times

        function check(prefix, stats; kwargs...)
            @test [name(s) for s in stats] == g.values["$(prefix)_names"]
            fit = fit_rem(seq, stats; n_controls=max_controls, tol=1e-12, kwargs...)
            @test fit.converged
            @test all(==(1.0), fit.sampling_probs)
            @test report("$(prefix)_coefficients", fit.coefficients)
            @test report("$(prefix)_std_errors", fit.std_errors)
            @test report("$(prefix)_std_errors", sqrt.(diag(vcov(fit))))
            @test report("$(prefix)_loglik", fit.log_likelihood)
            fit
        end
        # Model A: eventnet's min-weighted triadic family (the 0.2 defaults)
        check("eventnet", [Repetition(directed=false), TransitiveClosure(), CyclicClosure(),
                           SharedSender(), SharedReceiver()])
        # the four aggregations
        for agg in (:min, :max, :sum, :product)
            check("agg_$agg", [Repetition(), TransitiveClosure(aggregation=agg)])
        end
        # Model B: halflife decay on counts, degrees and the weighted closure
        hl = Float64(g.values["decay_halflife"])
        check("decay", [Repetition(), SenderActivity(), TransitiveClosure()];
              decay=halflife_to_decay(hl))
        # Model C: a sliding window on the same
        win = Float64(g.values["window_length"])
        check("window", [Repetition(), Reciprocity(), TransitiveClosure()]; window=win)
    end

    @testset "Golden: relevent::rem.dyad on the bundled WTC police calls" begin
        # The dataset relevent's tutorial uses, read by R and by Julia from the
        # SAME two TSV files (Networks.jl/data), so the 481 events and the 37-actor
        # universe are provably identical; rem.dyad's ordinal likelihood over the
        # full risk set IS the conditional logit REM fits when every non-case dyad
        # is enumerated. See test/fixtures/r/rem_relevent_wtc.R for the models.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "rem_relevent_wtc.toml"))
        @test g.provenance["relevent_version"] == "1.2.1"
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end
        wtc = Networks.load_dataset(:wtc_police_calls)
        n = Int(g.values["n_actors"]); M = Int(g.values["n_events"])
        @test wtc.n_actors == n == 37
        @test size(wtc.events, 1) == M == 481
        @test findall(wtc.is_icr) == Int.(g.values["icr_actors"])
        e = wtc.events
        seq = EventSequence([Event(e[k, 2], e[k, 3], Float64(e[k, 1])) for k in 1:M];
                            actors=ActorSet(1:n))
        @test length(seq) == M && seq.n_actors == n
        icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:n))
        max_controls = n * (n - 1) - 1                    # full risk set: 1332 dyads
        @test max_controls + 1 == Int(g.values["risk_set_size"])
        # The null log-likelihood is −M·log(n(n−1)) when the risk set is the full
        # dyad set on both sides — a check that the estimand is the same
        @test -M * log(n * (n - 1)) ≈ g.values["model1_loglik_null"] atol = 1e-8

        function check(prefix, stats)
            @test [name(s) for s in stats] == g.values["$(prefix)_names"]
            # tol = 1e-10: at ℓ ≈ −3100 a 1e-12 change in the log-likelihood is
            # below Float64 resolution (eps · 3100 ≈ 7e-13), so 1e-12 could never
            # be met; 1e-10 still puts the gradient norm below 1e-5, i.e. the
            # coefficients within ~1e-7 of the maximum — well inside the 1e-6
            # the fixture allows for optimizer termination slack
            fit = fit_rem(seq, stats; n_controls=max_controls, tol=1e-10)
            @test fit.converged
            @test Networks.is_exact(fit)
            @test all(==(1332), fit.risk_set_sizes)
            @test nobs(fit) == M
            @test report("$(prefix)_coefficients", coef(fit))
            @test report("$(prefix)_std_errors", stderror(fit))
            @test report("$(prefix)_std_errors", sqrt.(diag(vcov(fit))))
            @test report("$(prefix)_loglik", loglikelihood(fit))
            fit
        end
        # model 1: relevent's CovInt (x_i + x_j) — the tutorial's wtcfit1
        f1 = check("model1", [NodeSum(icr)])
        @test coef(f1)[1] ≈ 2.104 atol = 1e-3                  # the published ICR effect
        # model 2: CovSnd + CovRec
        check("model2", [SenderAttribute(icr), ReceiverAttribute(icr)])
        # model 3: + CovEvent(D): a dyadic covariate
        D = Dict((i, j) => ((7i + 3j) % 5) / 4 for i in 1:n for j in 1:n)
        check("model3", [SenderAttribute(icr), ReceiverAttribute(icr), DyadCovariate(D)])
    end

    @testset "Common-mistake errors are actionable (criterion 5)" begin
        # A NodeAttribute without a default refuses to invent a value
        age = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0))
        @test !has_default(age)
        @test has_default(NodeAttribute(:age, Dict(1 => 25.0), 0.0))
        @test has_default(NodeAttribute(:age, 0.0))
        st = EventNetworkState{Float64}(n_actors=3)
        err = try compute(NodeSum(age), st, 1, 3) catch e; e end
        @test err isa ArgumentError
        @test occursin(":age", err.msg) && occursin("actor 3", err.msg) && occursin("no default", err.msg)
        @test occursin("NodeAttribute(:age, values, default)", err.msg)
        @test compute(NodeSum(age), st, 1, 2) == 55.0
        # it reaches the fitting pipeline unchanged (the statistic is read for a
        # control dyad the attribute lacks)
        seq = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)]; actors=ActorSet(1:3))
        seq3 = EventSequence([Event(1, 3, 1.0), Event(3, 1, 2.0)]; actors=ActorSet(1:3))
        @test_throws ArgumentError compute_statistics(seq3, [ReceiverAttribute(age)])
        @test compute_statistics(seq, [SenderAttribute(age)]).sender_age == [25.0, 30.0, 25.0]
        @test_throws ArgumentError fit_rem(seq, [SenderAttribute(age)]; n_controls=5)
        # the explicit default is the opt-in
        filled = NodeAttribute(:age, Dict(1 => 25.0, 2 => 30.0), 0.0)
        @test compute(NodeSum(filled), st, 1, 3) == 25.0

        # A value of the wrong type for the attribute
        gender = NodeAttribute(:gender, Dict(1 => "M", 2 => "F"))
        err = try ActorMix(gender, 1, 2) catch e; e end
        @test err isa ArgumentError
        @test occursin(":gender holds String values", err.msg) && occursin("Int64", err.msg)
        err = try SenderCategorical(gender, :M) catch e; e end
        @test err isa ArgumentError && occursin("Symbol", err.msg)
        @test_throws ArgumentError ReceiverCategorical(gender, 2)
        # A numeric statistic on a categorical attribute
        for ctor in (NodeDifference, NodeSum, NodeProduct, SenderAttribute, ReceiverAttribute)
            err = try ctor(gender) catch e; e end
            @test err isa ArgumentError
            @test occursin("needs a numeric attribute", err.msg) && occursin(":gender holds String", err.msg)
            @test occursin("AttributeMatch", err.msg) && occursin("SenderCategorical", err.msg)
        end
        @test_throws ArgumentError NodeDifference(gender; absolute=true)

        # A misspelled statistic name lists what IS available
        obs = generate_observations(seq, [Repetition(), Reciprocity()], CaseControlSampler(n_controls=3, seed=1))
        err = try fit_rem(obs, ["repetiton"]) catch e; e end
        @test err isa ArgumentError
        @test occursin("\"repetiton\"", err.msg)
        @test occursin("\"repetition\", \"reciprocity\"", err.msg)
        @test !occursin("\"stratum\"", err.msg) && !occursin("\"tie_weight\"", err.msg)
        @test occursin("[name(s) for s in stats]", err.msg)

        # A Vector{Event} where an EventSequence is expected
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0)]
        for f in (() -> fit_rem(events, [Repetition()]),
                  () -> generate_observations(events, [Repetition()], CaseControlSampler(n_controls=2)),
                  () -> compute_statistics(events, [Repetition()]),
                  () -> EventNetworkState(events))
            err = try f() catch e; e end
            @test err isa ArgumentError
            @test occursin("not a Vector of Events", err.msg)
            @test occursin("EventSequence(events; actors=ActorSet(ids))", err.msg)
        end

        # The renamed recency transform points at its new name
        err = try RecencyStatistic(transform=:log) catch e; e end
        @test err isa ArgumentError && occursin(":inverse_log", err.msg)
        @test_throws ArgumentError RecencyStatistic(transform=:sqrt)
        @test_throws ArgumentError RecencyStatistic(decay=0.0)

        # An empty statistics list is a user mistake, not a tie_weights
        # consistency failure: every fit entry point says so
        for f in (() -> fit_rem(seq, AbstractStatistic[]; n_controls=5),
                  () -> fit_rem(seq, StatisticSet(AbstractStatistic[]); n_controls=5),
                  () -> fit_rem(obs, String[]),
                  () -> control_draw_cov(seq, AbstractStatistic[]; n_controls=5))
            err = try f() catch e; e end
            @test err isa ArgumentError
            @test occursin("at least one statistic", err.msg) && occursin("Repetition()", err.msg)
            @test !occursin("tie_weights", err.msg)
        end

        # A self-loop event under the default `exclude_self_loops=true` names
        # the loop and the keyword — not the actor universe, which is declared
        loop = EventSequence([Event(1, 1, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0)]; actors=ActorSet(1:3))
        err = try fit_rem(loop, [Repetition()]) catch e; e end
        @test err isa ArgumentError
        @test occursin("Event 1 is a self-loop (1 → 1)", err.msg)
        @test occursin("exclude_self_loops=true", err.msg) && occursin("exclude_self_loops=false", err.msg)
        @test occursin("Drop self-loop events", err.msg)
        @test !occursin("Declare the actor universe", err.msg)
        @test_throws ArgumentError generate_observations(loop, [Repetition()], CaseControlSampler(n_controls=3))
        # ... and the opt-in admits it: i → i is then a dyad like any other
        # (a loop that repeats, so the repetition coefficient has a finite maximum)
        loops = EventSequence([Event(1, 1, 1.0), Event(1, 2, 2.0), Event(1, 1, 3.0), Event(2, 3, 4.0)];
                              actors=ActorSet(1:3))
        admitted = fit_rem(loops, [Repetition()]; exclude_self_loops=false, n_controls=8)
        @test admitted.n_events == 4 && all(==(9), risk_set_sizes(admitted))
        @test isempty(admitted.separated) && Networks.is_exact(admitted)
        # a case outside the risk set still gets the universe message
        noloop = EventSequence([Event(1, 2, 1.0), Event(2, 3, 2.0), Event(3, 2, 3.0)]; actors=ActorSet(1:3))
        err = try fit_rem(noloop, [Repetition()]; at_risk=[2, 3]) catch e; e end
        @test err isa ArgumentError && occursin("Declare the actor universe", err.msg)

        # A `window` (or halflife) that is neither a number nor a period
        err = try fit_rem(seq, [Repetition()]; n_controls=5, window="2 days") catch e; e end
        @test err isa Union{ArgumentError, MethodError, TypeError}
        @test_throws ArgumentError fit_rem(seq, [Repetition()]; n_controls=5, window=-1.0)
        @test_throws ArgumentError fit_rem(seq, [Repetition()]; n_controls=5, window=Day(-1))
        # A calendar period on a numeric clock (the sequence counts events,
        # not seconds) names the clock type and both remedies
        err = try fit_rem(seq, [Repetition()]; n_controls=5, window=Day(2)) catch e; e end
        @test err isa ArgumentError
        @test occursin("Float64", err.msg) && occursin("clock units", err.msg)

        # One statistic instead of a vector is a one-element model, not a
        # MethodError; the arguments in the other order name the order
        six = EventSequence([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
                             Event(3, 1, 5.0), Event(1, 3, 6.0)]; actors=ActorSet(1:4))
        single = fit_rem(six, Repetition(); n_controls=11)
        @test single.stat_names == ["repetition"]
        @test coef(single) == coef(fit_rem(six, [Repetition()]; n_controls=11))
        @test size(control_draw_cov(six, Repetition(); n_controls=3, n_draws=2, rng=Xoshiro(1)).replicates) == (2, 1)
        for f in (() -> fit_rem([Repetition()], six; n_controls=5),
                  () -> fit_rem(Repetition(), six; n_controls=5),
                  () -> fit_rem(StatisticSet([Repetition()]), six; n_controls=5),
                  () -> control_draw_cov([Repetition()], six; n_controls=5),
                  () -> control_draw_cov(Repetition(), six; n_controls=5),
                  () -> control_draw_cov(StatisticSet([Repetition()]), six; n_controls=5))
            err = try f() catch e; e end
            @test err isa ArgumentError
            @test occursin("EventSequence FIRST", err.msg) && occursin("(seq, [Repetition(), Reciprocity()]", err.msg)
        end

        # A hand-built design writes `is_event` as 0/1: accepted; anything
        # else — and a non-integer `stratum` — is refused by column name
        hand = DataFrame(is_event=[1, 0, 1, 0, 1, 0], stratum=[1, 1, 2, 2, 3, 3],
                         x=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        asbool = DataFrame(is_event=Bool[1, 0, 1, 0, 1, 0], stratum=[1, 1, 2, 2, 3, 3],
                           x=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        @test coef(fit_rem(hand, ["x"])) == coef(fit_rem(asbool, ["x"]))
        @test fit_rem(hand, ["x"]).n_events == 3
        @test coef(fit_rem(DataFrame(is_event=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], stratum=hand.stratum, x=hand.x), ["x"])) ==
              coef(fit_rem(asbool, ["x"]))
        err = try fit_rem(DataFrame(is_event=[2, 0, 1, 0], stratum=[1, 1, 2, 2], x=[1.0, 0.0, 0.0, 1.0]), ["x"]) catch e; e end
        @test err isa ArgumentError && occursin("`is_event` must be a Bool column (or 0/1", err.msg)
        err = try fit_rem(DataFrame(is_event=["y", "n", "y", "n"], stratum=[1, 1, 2, 2], x=[1.0, 0.0, 0.0, 1.0]), ["x"]) catch e; e end
        @test err isa ArgumentError && occursin("`is_event`", err.msg) && occursin("String", err.msg)
        err = try fit_rem(DataFrame(is_event=[1, 0, 1, 0], stratum=["a", "a", "b", "b"], x=[1.0, 0.0, 0.0, 1.0]), ["x"]) catch e; e end
        @test err isa ArgumentError && occursin("`stratum` must be an integer column", err.msg)
    end

    @testset "Every exported docstring carries a runnable example (criterion 5)" begin
        # Grade-A criterion 5: every export has a docstring with a runnable
        # example. 32 of 88 exports had none before the 2026-09 round (a docs
        # build with checkdocs=:exports checks presence, not content). Walk the
        # docsystem so it cannot regress: each REM-owned docstring of an exported
        # binding — including the ones REM attaches to the shared Networks/
        # StatsAPI generics (`compute`, `name`, `has_edge`, `coef`, ...) — must
        # contain a fenced ```julia block (or a jldoctest). Copied from
        # Networks.jl's testset of the same name.
        meta = Base.Docs.meta(REM)
        missing_example = String[]
        undocumented = String[]
        blocks = Tuple{String,String}[]
        for nm in names(REM)
            nm === :REM && continue
            b = Base.Docs.Binding(REM, nm)
            if !haskey(meta, b)
                push!(undocumented, string(nm))
                continue
            end
            has_example = false
            for (_, ds) in meta[b].docs
                txt = ds.text isa AbstractString ? ds.text : join(string.(ds.text), "\n")
                for m in eachmatch(r"```julia\n(.*?)```"s, txt)
                    has_example = true
                    push!(blocks, (string(nm), String(m.captures[1])))
                end
                occursin("```jldoctest", txt) && (has_example = true)
            end
            has_example || push!(missing_example, string(nm))
        end
        @test isempty(undocumented)
        @test isempty(missing_example)
        # The StatsAPI verbs and the shared protocol generics are documented HERE
        # (REM-owned docstrings attached to the foreign bindings)
        for nm in (:coef, :stderror, :vcov, :confint, :loglikelihood, :nobs, :dof,
                   :aic, :bic, :coeftable, :compute, :name, :compute_all, :has_edge)
            @test haskey(meta, Base.Docs.Binding(REM, nm))
        end
        # And every block RUNS, each in a fresh module: "runnable" is a property
        # of the code, not of the fence
        @test length(blocks) >= 80
        for (nm, code) in blocks
            m = Module(Symbol("DocExample_", nm))
            ok = try
                Core.eval(m, :(using REM))
                Core.eval(m, Meta.parseall(code; filename="docstring:$nm"))
                true
            catch err
                println(stderr, "docstring example of $nm failed: ", sprint(showerror, err))
                false
            end
            @test ok
        end
    end

    @testset "show is compact and informative (criterion 5)" begin
        # `println(fit)` keeps the R-style block every tutorial relies on (the
        # family convention, shared with ERGM and Relevent); the data types get
        # one informative line each instead of a dump of their Dict internals.
        wtc = Networks.load_dataset(:wtc_police_calls)
        e = wtc.events
        seq = EventSequence([Event(e[k, 2], e[k, 3], Float64(e[k, 1])) for k in 1:size(e, 1)];
                            actors=ActorSet(1:37))
        s = sprint(show, seq)
        @test s == "EventSequence{Float64}(481 events, 37 actors, declared universe)"
        @test !occursin("Event(16", s)                        # not the event vector
        inferred = EventSequence([Event(1, 2, 1.0; eventtype=:a), Event(2, 1, 2.0; eventtype=:b)])
        @test sprint(show, inferred) ==
              "EventSequence{Float64}(2 events, 2 actors, inferred universe, 2 event types)"
        @test sprint(show, EventSequence([Event(1, 2, 1)]; actors=ActorSet(1:2))) ==
              "EventSequence{Int64}(1 event, 2 actors, declared universe)"
        @test sprint(show, EventSequence{Int}(; actors=ActorSet([5]))) ==
              "EventSequence{Int64}(0 events, 1 actor, declared universe)"
        # Containers print the same line (2-argument `show`, not only text/plain)
        @test sprint(show, [seq]) == "EventSequence{Float64}[$s]"

        state = EventNetworkState(seq)
        @test sprint(show, state) ==
              "EventNetworkState{Float64}(37 actors, 0 events absorbed, no memory decay, current_time = 0.0)"
        for ev in seq; update!(state, ev); end
        @test state.n_events == 481
        @test sprint(show, state) ==
              "EventNetworkState{Float64}(37 actors, 481 events absorbed, no memory decay, current_time = 481.0)"
        reset!(state)
        @test state.n_events == 0 && state.current_time == 0.0
        @test occursin("0 events absorbed", sprint(show, state))
        # A hand-built state counts too, whether or not it keeps the log
        lean = EventNetworkState(seq; keep_history=false)
        update!(lean, seq[1]); update!(lean, seq[2])
        @test lean.n_events == 2 && isempty(lean.event_history)
        @test sprint(show, EventNetworkState(seq; decay=halflife_to_decay(10.0))) ==
              "EventNetworkState{Float64}(37 actors, 0 events absorbed, halflife 10.0 (decay 0.06931), current_time = 0.0)"
        @test sprint(show, EventNetworkState(seq; window=2.0)) ==
              "EventNetworkState{Float64}(37 actors, 0 events absorbed, window 2.0, current_time = 0.0)"
        @test !occursin("Dict", sprint(show, state))

        @test sprint(show, ActorSet(1:37)) == "ActorSet(37 actors)"
        @test sprint(show, ActorSet(["Ann", "Bob"])) == "ActorSet(2 named actors)"
        @test sprint(show, RiskSet(5, [1, 2, 3], [1, 2, 3, 4])) ==
              "RiskSet(event 5: 3 senders × 4 receivers, 9 dyads, self-loops excluded)"
        @test sprint(show, RiskSet(1, [1, 2], [3]; exclude_self_loops=false)) ==
              "RiskSet(event 1: 2 senders × 1 receivers, 2 dyads, self-loops included)"
        @test sprint(show, CaseControlSampler(n_controls=100, seed=42)) ==
              "CaseControlSampler(n_controls=100, exclude_self_loops=true, seed=42)"
        @test sprint(show, CaseControlSampler(n_controls=5, exclude_self_loops=false)) ==
              "CaseControlSampler(n_controls=5, exclude_self_loops=false, seed=nothing)"
        @test sprint(show, StatisticSet([Repetition(), Reciprocity(), TransitiveClosure()])) ==
              "StatisticSet(3 statistics: repetition, reciprocity, transitive_closure)"
        many = StatisticSet([Repetition(), Reciprocity(), SenderActivity(), ReceiverActivity(),
                             SenderPopularity(), ReceiverPopularity(), TransitiveClosure(),
                             CyclicClosure(), SharedSender(), SharedReceiver()])
        @test sprint(show, many) ==
              "StatisticSet(10 statistics: repetition, reciprocity, sender_activity, receiver_activity, sender_popularity, receiver_popularity, transitive_closure, … (3 more))"
        @test sprint(show, Event(1, 2, 3.5)) == "Event(1 → 2 @ 3.5)"
        # A node attribute — and every statistic wrapping one — prints its name
        # and size, never the values Dict or the private sentinel type
        icr = NodeAttribute(:icr, Dict(i => Float64(wtc.is_icr[i]) for i in 1:37))
        @test sprint(show, icr) == "NodeAttribute{Float64}(:icr, 37 actors, no default)"
        @test sprint(show, NodeAttribute(:age, Dict(1 => 25.0), 0.0)) ==
              "NodeAttribute{Float64}(:age, 1 actor, default = 0.0)"
        @test sprint(show, NodeAttribute(:g, Dict(1 => "M", 2 => "F"))) ==
              "NodeAttribute{String}(:g, 2 actors, no default)"
        @test sprint(show, NodeAttribute(:g, "F")) == "NodeAttribute{String}(:g, 0 actors, default = \"F\")"
        @test sprint(show, REM._NO_DEFAULT) == "no default"
        s_sum = sprint(show, NodeSum(icr))
        @test occursin("NodeAttribute{Float64}(:icr, 37 actors, no default)", s_sum)
        @test !occursin("Dict", s_sum) && !occursin("_NoDefault", s_sum) && !occursin("=>", s_sum)
        @test !occursin("Dict", sprint(show, [NodeSum(icr), SenderAttribute(icr), AttributeMatch(icr)]))
        @test !occursin("Dict", sprint(show, StatisticSet([NodeSum(icr)])))

        # The fit keeps its R-style block: header lines, then the shared table
        fit = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=37 * 36 - 1)
        out = sprint(show, fit)
        @test occursin("Relational Event Model Results", out)
        @test occursin("Events: 481, Observations: 640692", out)
        @test occursin("Risk-set size: 1332 dyads, control sampling probability: 1.0", out)
        @test occursin("Converged: true ($(fit.iterations) iterations)", out)
        @test occursin("Std. errors: inverse Hessian (full risk set)", out)
        @test occursin("Signif. codes:", out)
        @test !occursin("Warning:", out)          # full risk set: nothing to caveat
        sampled = fit_rem(seq, [Repetition(), Reciprocity()]; n_controls=20, seed=1)
        @test occursin("Std. errors: inverse Hessian (one control draw)", sprint(show, sampled))
        @test occursin("Note: the risk set was sampled (20 of 1331 controls per event)", sprint(show, sampled))
        @test !occursin("Warning:", sprint(show, sampled))   # a sampled fit is a note, not a defect
    end

    @testset "Case-control sampling is consistent for a correctly specified model" begin
        # The tutorial fits the WTC calls twice — with sampled controls and with
        # the full risk set — and the two differ visibly there because that
        # model is crude; with a CORRECTLY specified model the sampled estimates
        # centre on the truth and tighten toward the full-risk-set fit as
        # n_controls grows (nested case-control sampling: Goldstein & Langholz
        # 1992). Same simulated sequence as "Coefficient recovery on simulated data".
        rng = Random.Xoshiro(20260706)
        n_actors = 8
        β_true = [0.6, 0.9]
        stats = [Repetition(), Reciprocity()]
        dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
        state = EventNetworkState{Float64}(n_actors=n_actors)
        state.actors = Set(1:n_actors)
        events = Event{Float64}[]
        for step in 1:600
            η = [sum(β_true .* compute_all(stats, state, s, r)) for (s, r) in dyads]
            w = exp.(η .- maximum(η)); w ./= sum(w)
            u = rand(rng); acc = 0.0; pick = length(dyads)
            for (k, p) in enumerate(w)
                acc += p
                if u <= acc; pick = k; break; end
            end
            ev = Event(dyads[pick][1], dyads[pick][2], Float64(step))
            push!(events, ev); update!(state, ev)
        end
        seq = EventSequence(events; actors=1:n_actors)
        full = fit_rem(seq, stats; n_controls=55)            # 55 = 8·7 − 1: full risk set
        @test all(sampling_probs(full) .== 1.0)
        # 20 of 55 controls, eight independent draws: every draw within 2.5 of
        # its own standard errors of the truth, and the mean of the draws within
        # 0.12 of the full-risk-set estimate (observed 0.03–0.04; per-draw sd 0.04)
        fits20 = [fit_rem(seq, stats; n_controls=20, seed=s) for s in 1:8]
        for f in fits20
            @test all(abs.(coef(f) .- β_true) .< 2.5 .* stderror(f))
            @test all(sampling_probs(f) .≈ 20 / 55)      # 56 dyads, the case excluded
        end
        mean20 = sum(coef.(fits20)) ./ length(fits20)
        @test all(abs.(mean20 .- coef(full)) .< 0.12)
        # Fewer controls: further from the full-risk-set fit, on average
        fits5 = [fit_rem(seq, stats; n_controls=5, seed=s) for s in 1:8]
        mean5 = sum(coef.(fits5)) ./ length(fits5)
        @test sum(abs.(mean5 .- coef(full))) > sum(abs.(mean20 .- coef(full)))
    end
end
