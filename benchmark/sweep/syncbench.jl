# A port of the PARALLEL FOR and REDUCTION tests of EPCC's OpenMP syncbench (v4.0,
# https://github.com/EPCCed/epcc-openmp-microbenchmarks), so that its numbers can be put
# next to those of the C benchmark that `run.sh` builds against libgomp and libomp.
#
# As in EPCC, a test runs `innerreps` parallel loops of `nthreads` iterations, each of which
# runs a delay loop calibrated to 0.1 µs, with `innerreps` doubled until a test takes 1 ms;
# the overhead is the time per loop minus that of the same delays run serially, as mean (±
# the 95 % confidence interval of EPCC's output) and as median over 20 outer repetitions.
#
#     julia -t 8,0 --project=benchmark/sweep benchmark/sweep/syncbench.jl
using ThreadTeams
using OhMyThreads: OhMyThreads
using Printf
using Statistics
include("../record.jl")

const HAVE_POLYESTER = Base.find_package("Polyester") !== nothing
HAVE_POLYESTER && @eval using Polyester

const NT = Threads.nthreads(:default)
const OUTERREPS = parse(Int, get(ENV, "SYNCBENCH_OUTER_REPS", "20"))
const TARGET_US = parse(Float64, get(ENV, "SYNCBENCH_TEST_TIME", "1000"))
const DELAY_US = parse(Float64, get(ENV, "SYNCBENCH_DELAY_TIME", "0.1"))
const CONF95 = 1.96

# EPCC's `delay`: a float accumulation that the compiler can neither vectorise nor remove
@noinline function delay(n::Int)
    a = 0.0f0
    for i in 0:(n - 1)
        a += i
    end
    a < 0 && println(a)
    return nothing
end

function delaylength_for(us)
    n = 0
    t = 0.0
    while t < us
        n = floor(Int, n * 1.1) + 1
        t0 = time_ns()
        for _ in 1:1000
            delay(n)
        end
        t = (time_ns() - t0) / 1.0e3 / 1000
    end
    return n
end

const TEAM = TeamScheduler()
const GREEDY = GreedyTeamScheduler(nchunks = NT)

# One parallel loop of `NT` iterations with a delay of `n` each, per mechanism
pfor(::Val{:team}, n) = tforeach(_ -> delay(n), 1:NT; scheduler = TEAM)
pfor(::Val{:greedyteam}, n) = tforeach(_ -> delay(n), 1:NT; scheduler = GREEDY)
pfor(::Val{:threads_static}, n) = Threads.@threads :static for _ in 1:NT
    delay(n)
end
pfor(::Val{:threads_dynamic}, n) = Threads.@threads :dynamic for _ in 1:NT
    delay(n)
end
pfor(::Val{:omt}, n) = OhMyThreads.tforeach(_ -> delay(n), 1:NT)
HAVE_POLYESTER && @eval pfor(::Val{:polyester}, n) = @batch per = thread for _ in 1:NT
    delay(n)
end

# A loop of `NT` iterations with a delay and a `+ 1` each, summed
pred(::Val{:team}, n) = tmapreduce(_ -> (delay(n); 1), +, 1:NT; scheduler = TEAM)
pred(::Val{:omt}, n) = OhMyThreads.tmapreduce(_ -> (delay(n); 1), +, 1:NT)
HAVE_POLYESTER && @eval function pred(::Val{:polyester}, n)
    a = 0
    @batch per = thread reduction = ((+, a),) for _ in 1:NT
        delay(n)
        a += 1
    end
    return a
end

# EPCC's reference: the delays of one thread, run serially
reference(n) = delay(n)
reference_red(n) = (delay(n); 1)

function run_reps(f, reps, n)
    for _ in 1:reps
        f(n)
    end
    return
end
function run_reps_red(f, reps, n)
    s = 0
    for _ in 1:reps
        s += f(n)
    end
    s == reps * NT || s == reps || error("wrong reduction result $s")
    return
end

# `innerreps` as EPCC computes it: doubled until one test takes the target time. Compiled
# and warmed up first (EPCC's C has nothing to compile): otherwise the first test already
# takes the target time, and the measured loops are too few to amortise waking the workers
function innerreps_for(run, f, n)
    run(f, 10, n)
    reps = 10
    t = 0.0
    while t < TARGET_US
        t0 = time_ns()
        run(f, reps, n)
        t = (time_ns() - t0) / 1.0e3
        reps *= 2
    end
    return reps
end

function measure(run, f, n)
    reps = innerreps_for(run, f, n)
    run(f, reps, n)   # EPCC ignores the first repetition
    times = map(1:OUTERREPS) do _
        t0 = time_ns()
        run(f, reps, n)
        (time_ns() - t0) / 1.0e3 / reps
    end
    return mean(times), std(times), median(times), reps
end
# With the mechanism as a type parameter, and `f` specialised on (Julia does not specialise
# on a function argument that is only passed on), so that the loop is not a dynamic dispatch
measure(run, f::F, ::Val{m}, n) where {F, m} = measure(run, k -> f(Val{m}(), k), n)

function main()
    n = delaylength_for(DELAY_US)
    println(
        "Julia ", VERSION, ", threads ", NT, ",", Threads.nthreads(:interactive),
        "; EPCC syncbench port, delay length $n ($(DELAY_US) µs), $OUTERREPS outer repetitions"
    )
    ref = measure(run_reps, reference, n)
    @printf("reference           mean %8.3f µs  median %8.3f µs\n", ref[1], ref[3])
    mechs = [:team, :greedyteam, :threads_static, :threads_dynamic, :omt]
    HAVE_POLYESTER && insert!(mechs, 3, :polyester)
    for (test, run, f, ref_f, ms) in (
            ("PARALLEL FOR", run_reps, pfor, reference, mechs),
            ("REDUCTION", run_reps_red, pred, reference_red, filter(in((:team, :polyester, :omt)), mechs)),
        )
        r = test == "PARALLEL FOR" ? ref : measure(run_reps_red, ref_f, n)
        for m in ms
            sleep(0.05)   # let the previous mechanism's threads go to sleep
            t = measure(run, f, Val(m), n)
            ovh, ci, med = t[1] - r[1], CONF95 * (t[2] + r[2]), t[3] - r[3]
            @printf(
                "%-12s %-16s reps %7d  time %8.3f µs  overhead %8.3f ± %6.3f µs  median overhead %8.3f µs\n",
                test, m, t[4], t[1], ovh, ci, med
            )
            record("syncbench", "$test $m", "overhead_mean_us", ovh)
            record("syncbench", "$test $m", "overhead_ci95_us", ci)
            record("syncbench", "$test $m", "overhead_median_us", med)
        end
    end
    return
end

main()
