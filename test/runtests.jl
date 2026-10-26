using ThreadTeams
using ParallelTestRunner

@info "Testing ThreadTeams" VERSION Threads.nthreads(:default) Threads.nthreads(:interactive)

testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")
# Enzyme does not support Julia nightly
VERSION >= v"1.14-" && delete!(testsuite, "enzyme")

# ParallelTestRunner starts its workers with a single thread; give them as many as we have
# (Julia 1.10 does not accept `--threads=N,0`)
ninteractive = Threads.nthreads(:interactive)
threads = "--threads=$(Threads.nthreads(:default))" * (ninteractive > 0 ? ",$ninteractive" : "")

const SETUP = joinpath(@__DIR__, "setup.jl")
init_code = quote
    using ThreadTeams
    include($SETUP)
end

# These measure timings or watch the workers of the pool, which is unreliable while other
# test processes compete for the CPU
serial = filter(in(keys(testsuite)), ["concurrency", "parallel_phase", "spin_time", "allocations", "foreach_chunk", "foreach_range"])

runtests(ThreadTeams, ARGS; testsuite, init_code, exeflags = [threads], serial)
