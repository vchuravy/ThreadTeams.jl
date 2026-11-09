# Pins Julia thread `i` to the `i`-th CPU of `ENV["BENCH_CPUS"]` (a comma-separated list, as
# `taskset -c` takes it), before the benchmark starts its workers. Loaded with `julia -L`
# by `run.sh`; Linux only (`sched_setaffinity`), does nothing elsewhere or if the variable
# is unset. `JULIA_EXCLUSIVE=1` would pin thread `i` to CPU `i - 1`, which on most
# machines with SMT puts two threads on one physical core.
if Sys.islinux() && !isempty(get(ENV, "BENCH_CPUS", ""))
    let cpus = parse.(Int, split(ENV["BENCH_CPUS"], ',')), nt = Threads.nthreads()
        length(cpus) >= nt ||
            error("BENCH_CPUS lists $(length(cpus)) CPUs for $nt threads")
        function pin_current(cpu)
            mask = zeros(UInt8, max(128, cld(cpu + 1, 8)))   # a `cpu_set_t`, 1024 bits
            mask[cpu ÷ 8 + 1] |= 0x01 << (cpu % 8)
            err = ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, length(mask), mask)
            err == 0 || error("sched_setaffinity(CPU $cpu) failed: errno $(Libc.errno())")
            return
        end
        for tid in 1:nt
            # A sticky task on thread `tid` (as ThreadPinning.jl's `@tspawnat` does)
            t = Task(() -> pin_current(cpus[tid]))
            t.sticky = true
            ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1)
            schedule(t)
            fetch(t)
        end
        cpus_now = [
            fetch(
                let t = Task(() -> ccall(:sched_getcpu, Cint, ()))
                    t.sticky = true
                    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1)
                    schedule(t)
                end
            ) for tid in 1:nt
        ]
        cpus_now == cpus[1:nt] || @warn "Threads run on CPUs $cpus_now after pinning to $(cpus[1:nt])"
    end
end
