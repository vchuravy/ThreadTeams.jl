# Time of a loop with 64 trivial iterations after a serial gap, for one mechanism per process
# (`WAKEUP_SCHED` = `team`, `threads` for `Threads.@threads :dynamic`, or `polyester`), so
# that the workers of one do not spin while another is measured:
#
#   - `busy`: the caller spins for the gap, as a serial phase of the program would, and the
#     workers keep their default spin time (for the team 100 µs, then they sleep);
#   - `spin0` (team only): `set_spin_time!(0)`, so the workers sleep right after each loop,
#     and the caller sleeps (`Libc.systemsleep`) for the gap.
#
# The added latency of a gap is the median after it minus the median without a gap.
#
#     for s in team threads polyester
#         WAKEUP_SCHED=$s julia -t 8,0 --project=benchmark/sweep benchmark/sweep/wakeup.jl
#     end
using ThreadTeams
using Printf
using Statistics
include("../aligned.jl")
include("../record.jl")

const SCHED = get(ENV, "WAKEUP_SCHED", "team")
SCHED == "polyester" && @eval using Polyester
const SAMPLES = parse(Int, get(ENV, "WAKEUP_SAMPLES", "1000"))
const GAPS_US = parse.(Int, split(get(ENV, "WAKEUP_GAPS_US", "0,50,200,1000,10000"), ','))

const TEAM = TeamScheduler()
loop(::Val{:team}, x) = tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = TEAM)
loop(::Val{:threads}, x) = Threads.@threads :dynamic for i in eachindex(x)
    @inbounds x[i] += 1.0
end
SCHED == "polyester" && @eval loop(::Val{:polyester}, x) = @batch for i in eachindex(x)
    @inbounds x[i] += 1.0
end

@inline function spin_until(t)
    while time_ns() < t
    end
    return
end

function measure(s, x, gap_ns, mode, n)
    ts = Vector{Int}(undef, n)
    for i in 1:n
        if gap_ns > 0
            mode === :busy ? spin_until(time_ns() + gap_ns) : Libc.systemsleep(gap_ns / 1.0e9)
        end
        t0 = time_ns()
        loop(s, x)
        ts[i] = time_ns() - t0
    end
    return ts
end

function main(s)
    x = aligned(zeros(64))
    for _ in 1:10_000
        loop(s, x)
    end
    println("Julia ", VERSION, ", threads ", Threads.nthreads(:default), ",", Threads.nthreads(:interactive), ", ", SCHED)
    modes = SCHED == "team" ? (:busy, :spin0) : (:busy,)
    for mode in modes
        mode === :spin0 && ThreadTeams.set_spin_time!(0)
        base = 0.0
        for g in GAPS_US
            mode === :spin0 && g == 0 && continue
            # Fewer samples for the long gaps, to bound the time of the run
            n = g >= 10_000 ? max(SAMPLES ÷ 10, 50) : SAMPLES
            ts = measure(s, x, g * 1000, mode, n) ./ 1.0e3
            med = median(ts)
            g == 0 && (base = med)
            @printf(
                "%-5s gap %6d µs: median %9.2f µs  mean %9.2f µs  p90 %9.2f µs  added %9.2f µs\n",
                mode, g, med, mean(ts), quantile(ts, 0.9), med - base
            )
            mech = "$SCHED $mode gap=$(g)us"
            record("wakeup", mech, "median_us", med)
            record("wakeup", mech, "mean_us", mean(ts))
            record("wakeup", mech, "p90_us", quantile(ts, 0.9))
            mode === :busy && record("wakeup", mech, "added_median_us", med - base)
        end
    end
    return
end

main(Val(Symbol(SCHED)))
