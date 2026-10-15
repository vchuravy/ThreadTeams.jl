# Trade-offs and caveats

**CPU use while idle.** After each loop, idle workers busy-wait for `spin_time()` (default
100 µs) before they sleep. The time counts from the end of the loop: a worker that finished
early keeps spinning while the loop waits for the other chunks, so one late chunk does not
send the others to sleep (else the next loop would pay for waking them). A worker sleeps
at once when another task is ready to run on its thread (or in the pool's shared queue);
otherwise it keeps its core busy, and with loops back to back `top` shows all threads at
100%. With more Julia threads than CPUs (e.g. `-t N` on N cores with Julia ≥ 1.12, which
adds an interactive thread), spinning workers also yield their core to other OS threads
every 5 µs (`sched_yield`); on an idle machine that would only add context switches, so
they do not otherwise. Other processes on the same cores slow the team down and are
slowed down, though stealing limits the damage: with one core of eight taken by another
busy process, 16-element loops cost about 0.9 µs with stealing and 90–600 µs without
(`-t 8` on 8 cores). Polyester.jl makes the same trade-off. `ThreadTeams.set_spin_time!(0)`
makes workers sleep at once, at the cost of a wake-up (about 10 µs) per loop;
`ThreadTeams.shutdown!()` stops them. To keep loops fast in one part of a program with a
short (or zero) spin time, wrap it in `ThreadTeams.parallel_phase() do ... end` (like TBB's
`parallel_phase`): idle workers then spin until it returns, across serial parts longer
than the spin time.

**Hybrid CPUs.** On CPUs with performance and efficiency cores (recent Intel CPUs, Apple
silicon), spinning cores take power and turbo headroom from the busy ones. LLVM's and GCC's
OpenMP runtimes do not spin by default there; the team does not adapt its default.
Consider `set_spin_time!(0)` with `parallel_phase`. Do not combine the team with another
spinning team on the same threads (Polyester-threaded code, multithreaded BLAS in the
loop body).

**Long-running tasks on worker threads.** Workers are sticky to their threads. If another
task occupies a worker's thread without yielding, the calling task runs that worker's chunk
itself (see "Stealing") and does not wait for the task. A worker that the OS deschedules
*while* it runs a chunk still delays the loop.

**Automatic differentiation.** Enzyme.jl differentiates parallel operations on the team in
forward and reverse mode, through a package extension that loads with EnzymeCore. Enzyme
cannot differentiate the team itself (atomics, long-lived tasks), so the extension has
rules for the two functions that run a region: they differentiate the loop body chunk by
chunk and run the derivative chunks on the team with the same scheduler. In reverse mode
the augmented forward passes of all chunks run in parallel with one tape per chunk, then
the reverse passes; Enzyme's scheme for `Threads.@threads`. With more than one thread,
Enzyme updates shadow memory atomically, so chunks may read the same data (stencils).
Captured scalars (a coefficient) get derivatives too. Reductions (`tmapreduce`, `treduce`,
`@tasks` with `@set reducer`) need a constant operator and a number or tuple of numbers as
the result, and do not work in batched mode. `tmap` and `tcollect` throw (their chunk
results are vectors); write into a preallocated array with `tmap!` or `tforeach`. Other
AD tools that work on the loop's code cannot differentiate through the team; use a serial
version or `Threads.@threads` there.

**Captured variables (boxing).** The loop body is a closure (as with `Threads.@threads`).
If the enclosing function assigns a variable more than once (e.g. in both branches of an
`if`) and the body uses it, Julia captures it in a `Core.Box`: the loop is type-unstable
and allocates, and assigning the variable in the body is a data race. As with OhMyThreads'
own schedulers, such a loop throws a `BoxedVariableError` (opt out with
`@allow_boxed_captures`). Assign such variables once (one destructuring assignment works)
or move the loop into its own function.

**Bounds facts.** Code inside the closure does not see facts the caller established (e.g.
array sizes from a `size` check), and the body is always called through the closure, even
when the loop runs serially. Some kernels can be slower per element than in a plain serial
loop.

**Other spinning pools.** Polyester.jl (e.g., through FastBroadcast.jl's `Threaded()` in
OrdinaryDiffEq.jl) and the team both pin spinning tasks to the same threads. They block each
other, and the calling task runs most chunks itself. In Trixi.jl, threaded RK
stage updates via Polyester together with the team made a time step about 5x slower than
serial stage updates.

**Other.** Precompilation always runs loops serially.
