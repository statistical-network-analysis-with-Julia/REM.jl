#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for REM.jl's hot loops.
#
# Covers the lazy-decay `EventNetworkState`: absorbing an event stream via
# `update!` (each update touches only the event's own keys — counts decay
# lazily on read, so per-event cost is O(1) regardless of decay), and the
# decayed-on-read count lookups that sit inside every statistic evaluation.
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite and prints one tab-separated `BENCHJL` line
# per benchmark (consumed by the site repo's tools/run_benchmarks.jl).

using BenchmarkTools
using REM
using Random

# ---------------------------------------------------------------------------
# Fixture: 5000 events among 100 actors with unit-ish exponential gaps
# ---------------------------------------------------------------------------

const N_ACTORS = 100
const N_EVENTS = 5000

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

const EVENTS = make_events(Random.Xoshiro(20260712), N_ACTORS, N_EVENTS)
const SEQ = EventSequence(EVENTS)

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

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
