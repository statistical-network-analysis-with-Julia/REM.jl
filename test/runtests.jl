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

        # apply_decay! materializes without changing read values
        before = [get_out_degree(state, a) for a in 1:n]
        REM.apply_decay!(state, state.current_time)
        @test [get_out_degree(state, a) for a in 1:n] ≈ before atol = 1e-12
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
        # 1→2 exists, 2→3 exists, so k=2 works
        tc = TransitiveClosure()
        @test compute(tc, state, 1, 3) == 1.0

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

        stats = AbstractStatistic[Repetition(), Reciprocity(), SenderActivity(),
                                  ReceiverPopularity(), TransitiveClosure()]
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
        # underflows to exactly 0 but 2·ccdf(Normal(), z) is a nonzero
        # subnormal. (|z| ≈ 38.4 is the Float64 representability limit:
        # beyond it even ccdf underflows — at |z| = 40 the true p ≈ 7e-350
        # is smaller than the smallest subnormal.)
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
        @test issubnormal(p)
    end

    @testset "StatsAPI co-loading" begin
        # `using REM, Statistics, StatsBase` must not create export
        # collisions: REM's coef/stderror/coeftable are StatsAPI methods,
        # the same generics StatsBase re-exports
        @test coef === StatsAPI.coef === StatsBase.coef
        @test stderror === StatsAPI.stderror === StatsBase.stderror
        @test coeftable === StatsAPI.coeftable

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
        @test coeftable(fit) isa DataFrame
        @test StatsBase.coef(fit) == fit.coefficients
        # ... and Statistics still works alongside
        @test mean(coef(fit)) ≈ sum(fit.coefficients) / 2

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
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
                  Event(3, 2, 4.0)]
        stats = [Repetition()]
        seq = EventSequence(events)               # universe inferred
        @test_logs (:warn, r"inferred from observed event") match_mode = :any begin
            fit_rem(seq, stats; n_controls=3, seed=1)
        end

        # No warning once the universe is declared ...
        seq_d = EventSequence(events; actors=1:5)
        @test_logs fit_rem(seq_d, stats; n_controls=5, seed=1)
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
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
        stats = [Repetition(), Reciprocity()]
        seq = EventSequence(events; actors=ActorSet([1, 2, 3, 7]))

        # FULL risk set (12 dyads, 11 controls per case): the conditional-logit
        # partial likelihood IS the exact ordinal relational-event likelihood.
        full = fit_rem(seq, stats; n_controls=11, seed=6)
        @test full.sampling_probs == [1.0, 1.0, 1.0]

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
        sampled = fit_rem(seq, stats; n_controls=2, seed=7)
        @test all(sampled.sampling_probs .< 1.0)

        md_s = Networks.fit_metadata(sampled)
        @test md_s.objective == :partial_likelihood  # same objective
        @test !md_s.is_exact                         # sampled risk set
        @test md_s.se_method == :hessian
        @test any(occursin("case-control sampling of the risk set", a)
                  for a in md_s.approximations)
        @test any(occursin("conditional on the ONE", a) for a in md_s.approximations)
        @test md_s.tie_method == :none               # still no ties in the data

        # The accessors are callable directly, and agree with what `show` prints
        @test Networks.is_exact(full) && !Networks.is_exact(sampled)
        @test occursin("control sampling probability: 1.0", sprint(show, full))
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
        # (the standard-error estimator and the tie correction are orthogonal)
        sand = fit_rem(seq, stats; n_controls=3, seed=3, ties=:efron, se=:sandwich)
        boot = fit_rem(seq, stats; n_controls=3, seed=3, ties=:efron, se=:bootstrap,
                       n_boot=5, rng=MersenneTwister(1))
        @test Networks.tie_method(sand) === :efron
        @test Networks.tie_method(boot) === :efron     # redrawn controls, same policy
        @test Networks.se_method(boot) === :bootstrap
    end
    @testset "Robust standard errors: se=:sandwich and se=:bootstrap" begin
        # Issue REM#2: the inverse-Hessian SEs are computed conditional on ONE
        # sampled control set, and they assume the observed information equals the
        # score variance. Two different tools, because two different assumptions:
        #   :sandwich  — event-clustered Godambe, drops the information equality
        #   :bootstrap — repeated control sampling, drops the one-control-draw
        #                conditioning (law of total variance)
        # A PARAMETRIC bootstrap would be the wrong tool here — what is missing is
        # variance from the sampling design, not from the model.
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
        boot = fit_rem(seq, stats; n_controls=5, seed=42, se=:bootstrap,
                       n_boot=40, rng=MersenneTwister(9))

        # All three replace only the COVARIANCE: the point estimates (and the
        # likelihood, and the risk-set bookkeeping) are identical
        @test coef(sand) == coef(hess)
        @test coef(boot) == coef(hess)
        @test sand.log_likelihood == hess.log_likelihood
        @test boot.log_likelihood == hess.log_likelihood
        @test sampling_probs(boot) == sampling_probs(hess)
        @test stderror(sand) != stderror(hess)
        @test stderror(boot) != stderror(hess)
        @test all(isfinite, stderror(sand))
        @test all(isfinite, stderror(boot))

        # The bootstrap is reproducible under a fixed rng
        boot2 = fit_rem(seq, stats; n_controls=5, seed=42, se=:bootstrap,
                        n_boot=40, rng=MersenneTwister(9))
        @test stderror(boot2) == stderror(boot)
        @test stderror(fit_rem(seq, stats; n_controls=5, seed=42, se=:bootstrap,
                               n_boot=40, rng=MersenneTwister(10))) !=
              stderror(boot)

        # THE POINT OF THE ISSUE: the repeated-control SEs EXCEED the single-draw
        # Hessian SEs, on every coefficient. That gap is the risk-set sampling
        # variability — the variance component the Hessian conditions away — and
        # it is what the law-of-total-variance combination adds back.
        @test all(stderror(boot) .> stderror(hess))

        # `se_method` reports what was ACTUALLY used, all three ways
        @test Networks.se_method(hess) === :hessian
        @test Networks.se_method(sand) === :sandwich
        @test Networks.se_method(boot) === :bootstrap
        @test Networks.fit_metadata(boot).se_method === :bootstrap
        @test Networks.fit_metadata(sand).se_method === :sandwich

        # ... and so does the printed output. Each estimator gets its OWN caveat,
        # because each accounts for something different: only :hessian is
        # "understated", and only :bootstrap includes the sampling variance.
        out_h, out_s, out_b = sprint.(show, (hess, sand, boot))
        @test occursin("inverse Hessian (one control draw)", out_h)
        @test occursin("understated", out_h)
        @test occursin("event-clustered sandwich", out_s)
        @test occursin("still computed on the ONE sampled control set", out_s)
        @test !occursin("understated", out_s)
        @test occursin("repeated control sampling", out_b)
        @test occursin("DO include the variance", out_b)
        @test !occursin("understated", out_b)

        # The approximations list agrees with the printed prose
        @test any(occursin("understated", a) for a in Networks.approximations(hess))
        @test any(occursin("robust to misspecification", a)
                  for a in Networks.approximations(sand))
        @test any(occursin("DO include the variance", a)
                  for a in Networks.approximations(boot))
        # The tie-handling report is a property of the LIKELIHOOD, not of the
        # standard errors, so it survives all three untouched — and this sequence
        # (one event per integer time) has no ties at all, so all three say so
        @test all(Networks.tie_method(r) === :none for r in (hess, sand, boot))
        @test all(!any(occursin("tied event times", a) for a in Networks.approximations(r))
                  for r in (hess, sand, boot))

        # The control inclusion probabilities are exposed, not buried
        @test length(sampling_probs(hess)) == 80          # one per stratum/event
        @test length(risk_set_sizes(hess)) == 80
        @test all(0 .< sampling_probs(hess) .< 1)         # the risk set WAS sampled
        @test all(==(132), risk_set_sizes(hess))          # 12·11 ordered dyads
        @test !Networks.is_exact(hess)

        # With the FULL risk set there is no control sampling: nothing is
        # conditioned away, so no caveat is printed and `is_exact` holds
        full = fit_rem(seq, stats; n_controls=200, seed=1)
        @test all(≈(1.0), sampling_probs(full))
        @test Networks.is_exact(full)
        @test !occursin("understated", sprint(show, full))

        # :bootstrap redraws the CONTROLS, so it is meaningless once the controls
        # have been drawn into a DataFrame — it must be refused there, loudly,
        # with a pointer to the method that can do it
        obs = generate_observations(seq, StatisticSet(stats),
                                    CaseControlSampler(n_controls=5, seed=42))
        err = try
            fit_rem(obs, ["repetition", "reciprocity"]; se=:bootstrap)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("EventSequence", err.msg)
        # ... but the sandwich needs no redraw and works fine on the DataFrame
        @test Networks.se_method(fit_rem(obs, ["repetition", "reciprocity"]; se=:sandwich)) === :sandwich

        # Unknown se symbols are rejected, not silently ignored
        @test_throws ArgumentError fit_rem(seq, stats; se=:jackknife)
        @test_throws ArgumentError fit_rem(obs, ["repetition", "reciprocity"]; se=:jackknife)
    end
end
