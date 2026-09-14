#!/usr/bin/env julia
# benchmark/regression_tests.jl — allocation-regression assertions for the
# REM.jl hot loops. Standalone; run with
#     julia --project=benchmark benchmark/regression_tests.jl
# (CI runs it on the ubuntu / Julia 1.12 cell; the site's
# tools/run_benchmarks.jl runs it after benchmarks.jl).
#
# These are the pins the 2026-09 panel round established (items 14 and 26,
# criterion 4), measured at 0 bytes or at a stream-independent budget:
#
#   * the conditional-logit derivative kernel `_clogit_derivatives!` writes
#     into preallocated buffers — 0 B per evaluation; the closure handed to
#     `Networks.newton_fit` allocates exactly the (p) gradient and (p×p)
#     Hessian it returns, the same at 300 and at 3000 strata;
#   * every exported statistic evaluates in 0 B on a warmed state, whether the
#     state decays, keeps a sliding window or neither, and so does
#     `compute_all!` over the whole set;
#   * `update!` — the per-event absorb of the streaming design — is 0 B on a
#     warmed state (round 2: the eager `get!(d, k, Set{Int}())` cost 160 B
#     per event), and a small strata-independent budget with the event log
#     or the window FIFO kept;
#   * `generate_observations` streams in O(events): 20× the actors costs at
#     most 1.5× the bytes, 2× the events at most 2.2×; `n_dyads` on a sorted
#     risk set is an allocation-free merge.
#
# Any allocation appearing here is a performance regression — the tests
# assert the loops STAY allocation-free rather than tracking a noisy byte
# budget. The same pins live in test/runtests.jl ("Allocation-free clogit
# kernel", "Streams in O(events)", "Allocation-free statistics"); this file
# is the gate that runs without the rest of the suite.

using REM
using LinearAlgebra
using Random
using Test

"Random directed event stream with integer clock and non-unit weights."
function make_events(rng::AbstractRNG, n_actors::Int, n_events::Int; weights::Bool=false)
    ev = Event{Float64}[]
    for k in 1:n_events
        s = rand(rng, 1:n_actors)
        r = rand(rng, 1:(n_actors - 1)); r >= s && (r += 1)
        w = weights ? 0.5 + rand(rng) : 1.0
        push!(ev, Event(s, r, Float64(k); weight=w))
    end
    return ev
end

"Bytes allocated by `compute(stat, state, s, r)` on a warmed call."
function alloc_compute(stat, state, s, r)
    compute(stat, state, s, r)
    return @allocated compute(stat, state, s, r)
end

"Bytes allocated by the kernel, the newton_fit closure and the sandwich at `n_strata`."
function kernel_allocs(n_strata; p::Int=3, k::Int=6)
    rng = Random.Xoshiro(11)
    n = n_strata * k
    X = randn(rng, n, p)
    y = [i % k == 1 for i in 1:n]
    strata = [(i - 1) ÷ k + 1 for i in 1:n]
    idx = REM._strata_index(strata, y)
    work = REM._clogit_workspace(idx, p)
    tw = ones(n)
    grad = zeros(p); hess = zeros(p, p)
    β = [0.1, -0.2, 0.3]
    REM._clogit_derivatives!(grad, hess, X, idx, β, work, tw)          # warm up
    a_kernel = @allocated REM._clogit_derivatives!(grad, hess, X, idx, β, work, tw)
    f = REM._clogit_objective(X, idx, work, tw)
    f(β)
    a_closure = @allocated f(β)
    bread = Matrix{Float64}(I, p, p)
    REM._clogit_sandwich_cov(X, idx, β, bread, tw, work)
    a_sandwich = @allocated REM._clogit_sandwich_cov(X, idx, β, bread, tw, work)
    return a_kernel, a_closure, a_sandwich
end

"Bytes allocated by a warmed `generate_observations` on a fresh stream."
function stream_allocs(n_actors, n_events)
    seq = EventSequence(make_events(Random.Xoshiro(3), n_actors, n_events);
                        actors=ActorSet(1:n_actors))
    ss = StatisticSet([Repetition(), Reciprocity(), SenderActivity()])
    sampler = CaseControlSampler(n_controls=20, seed=1)
    generate_observations(seq, ss, sampler)                            # warm up
    return @allocated generate_observations(seq, ss, sampler)
end

@testset "REM allocation regressions" begin
    @testset "clogit derivative kernel is allocation-free (item 14)" begin
        k_small, c_small, s_small = kernel_allocs(300)
        k_big, c_big, s_big = kernel_allocs(3000)
        @test k_small == 0
        @test k_big == 0
        @test c_small <= 512 && c_big <= 512
        @test c_big == c_small                 # p-sized, not strata-sized
        @test s_big == s_small                 # sandwich reuses the workspace
        @test s_big <= 2048
    end

    @testset "every exported statistic evaluates in 0 B (item 26)" begin
        n = 30
        sq = EventSequence(make_events(Random.Xoshiro(11), n, 600; weights=true);
                           actors=ActorSet(1:n))
        gender = NodeAttribute(:g, Dict(i => (isodd(i) ? "M" : "F") for i in 1:n))
        age = NodeAttribute(:a, Dict(i => Float64(i) for i in 1:n))
        every = AbstractStatistic[
            Repetition(), Repetition(directed=false), Reciprocity(), InertiaStatistic(),
            RecencyStatistic(), RecencyStatistic(directed=false, transform=:inverse_log),
            RecencyStatistic(transform=:exp_decay, decay=0.1),
            DyadCovariate(Dict((1, 2) => 1.0)),
            SenderActivity(), ReceiverActivity(), SenderPopularity(), ReceiverPopularity(),
            TotalDegree(), DegreeDifference(), LogDegree(),
            TransitiveClosure(), TransitiveClosure(weighted=false),
            TransitiveClosure(aggregation=:sum), CyclicClosure(), SharedSender(),
            SharedReceiver(), CommonNeighbors(), CommonNeighbors(weighted=false),
            GeometricWeightedTriads(), GeometricWeightedTriads(closure_type=:cyclic),
            FourCycle(), FourCycle(cycle_type=:mixed), FourCycle(weighted=false),
            GeometricWeightedFourCycles(),
            AttributeMatch(gender), ActorMix(gender, "M", "F"), NodeDifference(age),
            NodeSum(age), NodeProduct(age), SenderAttribute(age), ReceiverAttribute(age),
            SenderCategorical(gender, "M"), ReceiverCategorical(gender, "F")]
        # every exported concrete statistic type appears in the list, so a new
        # statistic cannot ship without a pin
        exported = [T for nm in names(REM) for T in (getfield(REM, nm),)
                    if (T isa DataType && T <: AbstractStatistic && isconcretetype(T)) ||
                       (T isa UnionAll && T.body isa DataType && T.body <: AbstractStatistic)]
        for T in exported
            @test any(s -> s isa T, every)
        end
        for (label, st) in (("decay", EventNetworkState(sq; decay=0.01)),
                            ("window", EventNetworkState(sq; window=200.0)),
                            ("plain", EventNetworkState(sq)))
            for e in sq; update!(st, e); end
            for stat in every
                a = alloc_compute(stat, st, 3, 7)
                @test a == 0
                a == 0 || println(stderr, "$(name(stat)) allocates $a B on the $label state")
                @test alloc_compute(stat, st, n, n - 1) == 0
            end
            ss = StatisticSet(every)
            dest = zeros(length(ss))
            compute_all!(dest, ss, st, 3, 7)
            @test @allocated(compute_all!(dest, ss, st, 3, 7)) == 0
        end
    end

    @testset "update! is allocation-free on a warmed state" begin
        sq = EventSequence(make_events(Random.Xoshiro(5), 20, 400; weights=true);
                           actors=ActorSet(1:20))
        function alloc_update(st)
            for e in sq; update!(st, e); end
            update!(st, sq[1])
            return @allocated update!(st, sq[1])
        end
        @test alloc_update(EventNetworkState(sq; keep_history=false)) == 0
        @test alloc_update(EventNetworkState(sq; keep_history=false, decay=0.01)) == 0
        @test alloc_update(EventNetworkState(sq; keep_history=true)) <= 64
        @test alloc_update(EventNetworkState(sq; keep_history=false, window=50.0)) <= 64
    end

    @testset "generate_observations streams in O(events) (criterion 4)" begin
        a_100_2000 = stream_allocs(100, 2000)
        a_2000_2000 = stream_allocs(2000, 2000)
        a_100_4000 = stream_allocs(100, 4000)
        @test a_2000_2000 <= 1.5 * a_100_2000   # 20× the actors: ≤ 1.5× the bytes
        @test a_100_4000 <= 2.2 * a_100_2000    # 2× the events: ≤ 2.2× the bytes

        ids = collect(1:2000)
        rs = RiskSet(1, ids, ids)
        nd(rs) = n_dyads(rs)
        nd(rs)
        @test @allocated(nd(rs)) == 0
        @test n_dyads(rs) == 2000 * 1999
    end

    @testset "fit_rem(::DataFrame) stratum validation allocates O(strata), not O(rows)" begin
        # `_stratum_counts` is the function barrier behind which the per-stratum
        # case/row counting runs on the DataFrame's `AbstractVector`-typed
        # columns; before it existed the loop dispatched per row (≈10
        # allocations and 250 B per row). Dict growth is O(log strata)
        # allocations: measured 47 at 42 000 rows and 59 at 84 000.
        function count_allocations(n_events)
            seq = EventSequence(make_events(Random.Xoshiro(3), 100, n_events);
                                actors=ActorSet(1:100))
            obs = generate_observations(seq, [Repetition(), Reciprocity()],
                                        CaseControlSampler(n_controls=20, seed=1))
            REM._stratum_counts(obs.stratum, obs.is_event)               # warm up
            return @allocations REM._stratum_counts(obs.stratum, obs.is_event)
        end
        n_2000 = count_allocations(2000)
        n_4000 = count_allocations(4000)
        @test n_2000 <= 64
        @test n_4000 <= n_2000 + 16
    end
end
