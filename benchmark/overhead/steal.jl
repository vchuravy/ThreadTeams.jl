# Time per loop and rate of stolen chunks (see `ThreadTeams.stolen_chunks`) for a loop with
# one chunk per thread, each of which spins for `w` nanoseconds: the team, the greedy team
# and Polyester.jl's `@batch` (if installed). Run each scheduler in a process of its own,
# so that the workers of one do not compete with the other's for the threads:
#
#     for s in team greedy polyester
#         STEAL_SCHED=$s taskset -c 0,2,4,6,8,10,12,14 julia -t 8,0 \
#             --project=benchmark/overhead benchmark/overhead/steal.jl
#     end
#
# Report the mean as well as the median: when the OS deschedules a worker for a time slice
# (a few milliseconds), waiting for it costs one region a lot, while stealing its chunk
# costs many regions a little, so the median alone favours waiting.
using ThreadTeams
using Printf
using Statistics

const SCHED = get(ENV, "STEAL_SCHED", "team")
SCHED == "polyester" && @eval using Polyester

const NT = Threads.nthreads(:default)
const N = parse(Int, get(ENV, "STEAL_N", "20000"))

@inline function spin(w)
    t = time_ns() + w
    while time_ns() < t
    end
    return
end

const TEAM = TeamScheduler()
const GREEDY = GreedyTeamScheduler(nchunks = NT)
loop(::Val{:team}, w) = tforeach(_ -> spin(w), 1:NT; scheduler = TEAM)
loop(::Val{:greedy}, w) = tforeach(_ -> spin(w), 1:NT; scheduler = GREEDY)
SCHED == "polyester" && @eval loop(::Val{:polyester}, w) = @batch per = thread for _ in 1:NT
    spin(w)
end

stolen() = isdefined(ThreadTeams, :stolen_chunks) ? ThreadTeams.stolen_chunks() : 0

function measure(s, w)
    ts = Vector{Int}(undef, N)
    for _ in 1:1000
        loop(s, w)
    end
    s0 = stolen()
    for i in 1:N
        t0 = time_ns()
        loop(s, w)
        ts[i] = time_ns() - t0
    end
    return median(ts), mean(ts), (stolen() - s0) / (N * (NT - 1))
end

println("Julia ", VERSION, ", threads ", NT, ",", Threads.nthreads(:interactive), ", ", SCHED)
for w in (0, 100, 300, 1_000, 3_000, 10_000)
    med, avg, rate = Base.invokelatest(measure, Val(Symbol(SCHED)), w)
    @printf(
        "w = %6d ns: median %8.0f ns  mean %8.0f ns  stolen %5.1f%%\n", w, med, avg, 100rate
    )
end
