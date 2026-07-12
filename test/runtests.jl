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

        # has_edge is unexported (it would collide with Graphs.has_edge)
        @test REM.has_edge(state, 1, 2)
        @test !REM.has_edge(state, 3, 1)
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
        @test seq.actors == Set([1, 2, 3])

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

        seq = EventSequence(events)
        # 55 < 56 distinct controls per case, so the sampler enumerates the
        # full risk set — no sampling noise beyond the simulation itself
        result = fit_rem(seq, stats; n_controls=100, seed=1)

        @test result.converged
        @test result.coefficients[1] ≈ β_true[1] atol = 0.25
        @test result.coefficients[2] ≈ β_true[2] atol = 0.25
        # Both effects strongly significant
        @test all(result.p_values .< 0.01)
    end
end
