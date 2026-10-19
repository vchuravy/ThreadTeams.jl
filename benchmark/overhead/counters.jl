# Hardware counters per loop on the calling thread (and summed over all threads) for a loop
# with 64 trivial iterations: the team vs Polyester.jl's `@batch` (if installed) vs a serial
# loop. Linux only (LinuxPerf.jl); `perf_event_paranoid` must allow user-space counting.
#
#     taskset -c 8-15 julia -t 8 --project=benchmark/overhead benchmark/overhead/counters.jl
using ThreadTeams
using LinuxPerf
using Printf
include("../aligned.jl")

const HAVE_POLYESTER = Base.find_package("Polyester") !== nothing
HAVE_POLYESTER && @eval using Polyester

const N = parse(Int, get(ENV, "OVERHEAD_N", "64"))
const TEAM = TeamScheduler()

f_team(x) = tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = TEAM)
f_team_reduce(x) = tmapreduce(i -> (@inbounds x[i]), +, eachindex(x); scheduler = TEAM)
f_serial(x) = for i in eachindex(x)
    @inbounds x[i] += 1.0
end
HAVE_POLYESTER && @eval f_batch(x) = @batch for i in eachindex(x)
    @inbounds x[i] += 1.0
end

function loops(f, x, n)
    for _ in 1:n
        f(x)
    end
    return
end

const EVENTS = "(cpu-cycles, instructions, branch-instructions, branch-misses), " *
    "(cache-references, cache-misses), (L1-dcache-loads, L1-dcache-load-misses)"

# Per-event totals over the measured threads, scaled for multiplexing
function totals(stats)
    order = String[]
    acc = Dict{String, Float64}()
    for t in stats.threads, g in t.groups, c in g
        c.enabled > 0 || continue
        name = string(c.event)
        name in order || push!(order, name)
        acc[name] = get(acc, name, 0.0) + LinuxPerf.scaledcount(c)
    end
    return order, acc
end

function report(name, f, x, n; threads)
    loops(f, x, 1000)
    stats = @pstats EVENTS threads = threads loops(f, x, n)
    order, acc = totals(stats)
    @printf("%-14s %s ", name, threads ? "all threads:   " : "calling thread:")
    for ev in order
        short = replace(
            ev, "hw:" => "", "cache:L1_data:read:" => "L1d_", "cpu-cycles" => "cycles",
            "instructions" => "instr", "branch_mispredicts" => "br_miss",
            "cache_access" => "LLC_acc", "cache_misses" => "LLC_miss"
        )
        @printf("%s %7.0f  ", short, acc[ev] / n)
    end
    return println()
end

x = aligned(zeros(N))   # cache-line aligned, as in overhead.jl
println(
    "Julia ", VERSION, ", threads ", Threads.nthreads(:default), ",",
    Threads.nthreads(:interactive), ", ", N, " iterations; counts per loop"
)
fs = Any[
    ("team", f_team, 200_000), ("team reduce", f_team_reduce, 200_000),
    ("serial", f_serial, 2_000_000),
]
HAVE_POLYESTER && insert!(fs, 3, ("Polyester", f_batch, 200_000))
for (name, f, n) in fs
    report(name, f, x, n; threads = false)
    report(name, f, x, n; threads = true)
end
