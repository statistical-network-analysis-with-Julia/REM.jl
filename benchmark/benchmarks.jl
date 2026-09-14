#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for REM.jl's hot loops.
#
# Three groups:
#
#   state         the lazy-decay `EventNetworkState`: absorbing an event stream
#                 via `update!` (each update touches only the event's own keys —
#                 counts decay lazily on read, so per-event cost is O(1)
#                 regardless of decay) and the decayed-on-read count lookups
#                 that sit inside every statistic evaluation;
#   observations  `generate_observations` — the case-control design build that
#                 must stream in O(events) and must NOT carry the actor
#                 universe per event (panel 2026-09, criterion 4);
#   fit           `fit_rem` end to end on the same streams (design build +
#                 Newton–Raphson on `Networks.newton_fit`).
#
# After the timings the suite asserts complexity and prints one `SCALING`
# line per assertion (consumed by the site repo's tools/run_benchmarks.jl);
# a violated bound exits non-zero:
#
#   events 2000 → 4000 (100 actors):   time ≤ 2.5×, bytes ≤ 2.2×   (linear in the stream)
#   actors 100 → 2000 (2000 events):   bytes ≤ 1.5×                (the O(actors) terms are one-time)
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite and prints one tab-separated `BENCHJL` line
# per benchmark. The allocation-regression @test blocks live next door in
# benchmark/regression_tests.jl (same environment).

using BenchmarkTools
using REM
using Random

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

const N_ACTORS = 100
const N_EVENTS = 5000

# Streams for the scaling assertions
const OBS_ACTORS_SMALL = 100
const OBS_ACTORS_LARGE = 2000
const OBS_EVENTS_SMALL = 2000
const OBS_EVENTS_LARGE = 4000
const N_CONTROLS = 20

# Tolerated ratios (see header)
const EVENTS_TIME_LIMIT = 2.5
const EVENTS_BYTES_LIMIT = 2.2
const ACTORS_BYTES_LIMIT = 1.5

"Random directed event stream: exponential gaps, uniform sender ≠ receiver."
function make_events(rng::AbstractRNG, n_actors::Int, n_events::Int)
    events = Event{Float64}[]
    t = 0.0
    for _ in 1:n_events
        t += -log(rand(rng))
        s = rand(rng, 1:n_actors)
        r = rand(rng, 1:(n_actors - 1))
        r >= s && (r += 1)
        push!(events, Event(s, r, t))
    end
    return events
end

make_seq(seed, n_actors, n_events) =
    EventSequence(make_events(Random.Xoshiro(seed), n_actors, n_events);
                  actors=ActorSet(1:n_actors))

const EVENTS = make_events(Random.Xoshiro(20260712), N_ACTORS, N_EVENTS)
const SEQ = EventSequence(EVENTS; actors=ActorSet(1:N_ACTORS))

# One stream per (actors, events) cell, the same seed so the small-actor
# streams are prefixes of nothing but share the draw discipline
const SEQ_A100_E2000 = make_seq(3, OBS_ACTORS_SMALL, OBS_EVENTS_SMALL)
const SEQ_A2000_E2000 = make_seq(3, OBS_ACTORS_LARGE, OBS_EVENTS_SMALL)
const SEQ_A100_E4000 = make_seq(3, OBS_ACTORS_SMALL, OBS_EVENTS_LARGE)

const STATS = StatisticSet([Repetition(), Reciprocity(), SenderActivity()])
const SAMPLER = CaseControlSampler(n_controls=N_CONTROLS, seed=1)

"Absorb the full event stream into a fresh state."
function absorb!(state, events)
    reset!(state)
    for e in events
        update!(state, e)
    end
    return state
end

"Sweep the lazily decayed dyad/degree counts (as statistics do on read)."
function read_sweep(state, n_actors)
    s = 0.0
    for i in 1:n_actors
        s += get_out_degree(state, i) + get_in_degree(state, i)
        for j in (i + 1):min(i + 10, n_actors)
            s += get_dyad_count(state, i, j)
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "state")
    # Lazy decay: per-event cost must be O(1) — no full-table decay pass
    g["update_stream_decay"] =
        @benchmarkable absorb!(state, $EVENTS) setup =
            (state = EventNetworkState($SEQ; decay=0.05))
    g["update_stream_nodecay"] =
        @benchmarkable absorb!(state, $EVENTS) setup =
            (state = EventNetworkState($SEQ))
    g["lazy_read_sweep"] =
        @benchmarkable read_sweep(state, $N_ACTORS) setup =
            (state = absorb!(EventNetworkState($SEQ; decay=0.05), $EVENTS))
end

let g = addgroup!(SUITE, "observations")
    # The design build: one case + N_CONTROLS controls per event, three
    # statistics, static risk set (its dyad count is computed once)
    g["a100_e2000"] = @benchmarkable generate_observations($SEQ_A100_E2000, $STATS, $SAMPLER)
    g["a2000_e2000"] = @benchmarkable generate_observations($SEQ_A2000_E2000, $STATS, $SAMPLER)
    g["a100_e4000"] = @benchmarkable generate_observations($SEQ_A100_E4000, $STATS, $SAMPLER)
end

let g = addgroup!(SUITE, "fit")
    # End to end: design + Newton–Raphson (inverse-Hessian standard errors)
    g["a100_e2000"] = @benchmarkable fit_rem($SEQ_A100_E2000, $STATS;
                                             n_controls=N_CONTROLS, seed=1)
    g["a100_e4000"] = @benchmarkable fit_rem($SEQ_A100_E4000, $STATS;
                                             n_controls=N_CONTROLS, seed=1)
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
end

_ratio(f, results, group, large, small) =
    f(median(results[group][large])) / f(median(results[group][small]))

"""
Assert the streaming complexity (see the header) and print one `SCALING`
line per bound: `SCALING\\t<label>\\t<comparison>\\t<ratio>`.
"""
function assert_scaling(results::BenchmarkGroup)
    ok = true
    checks = (
        # label, group, large, small, measure, limit, what
        ("observations_time", "observations", "a100_e4000", "a100_e2000",
         BenchmarkTools.time, EVENTS_TIME_LIMIT, "e4000/e2000",
         "generate_observations is no longer O(events)"),
        ("observations_bytes", "observations", "a100_e4000", "a100_e2000",
         BenchmarkTools.memory, EVENTS_BYTES_LIMIT, "e4000/e2000",
         "generate_observations allocates super-linearly in the stream"),
        ("observations_actor_bytes", "observations", "a2000_e2000", "a100_e2000",
         BenchmarkTools.memory, ACTORS_BYTES_LIMIT, "a2000/a100",
         "generate_observations carries the actor universe per event again"),
        ("fit_time", "fit", "a100_e4000", "a100_e2000",
         BenchmarkTools.time, EVENTS_TIME_LIMIT, "e4000/e2000",
         "fit_rem is no longer O(events)"),
        ("fit_bytes", "fit", "a100_e4000", "a100_e2000",
         BenchmarkTools.memory, EVENTS_BYTES_LIMIT, "e4000/e2000",
         "fit_rem allocates super-linearly in the stream"),
    )
    for (label, group, large, small, f, limit, cmp, what) in checks
        ratio = _ratio(f, results, group, large, small)
        println("SCALING\t", label, "\t", cmp, "\t", round(ratio, digits=2))
        if ratio > limit
            println(stderr, "SCALING FAILURE: $label ratio $cmp is ",
                    round(ratio, digits=2), "x (limit $(limit)x): $what.")
            ok = false
        end
    end
    return ok
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    assert_scaling(results) || exit(1)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
