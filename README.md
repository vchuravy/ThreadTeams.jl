# ThreadTeams.jl

[![CI](https://github.com/vchuravy/ThreadTeams.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/vchuravy/ThreadTeams.jl/actions/workflows/CI.yml)
[![docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vchuravy.github.io/ThreadTeams.jl/dev)

A team of persistent worker tasks, one per thread, that runs the chunks of parallel loops
with low per-loop overhead. It is plain Julia on top of Base tasks, and it is the pool
behind the `TeamScheduler` (`:team`) and `GreedyTeamScheduler` (`:greedyteam`) of
[OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl) and the CPU back-end of
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

One sticky worker task runs per thread of the `:default` thread pool. A parallel loop
splits its range into one contiguous chunk per thread, runs chunk 1 on the calling task and
hands the rest to the workers. After a loop, idle workers spin (100 µs by default) before
they sleep, so back-to-back loops do not pay for creating, scheduling and waking tasks.
Starting a loop costs about 0.3 µs and allocates nothing after its first run, against
2–4 µs and about 3 KB for
`Threads.@threads` or OhMyThreads' default scheduler (see the [overhead](https://vchuravy.github.io/ThreadTeams.jl/dev/overhead/) page).

## Installation

The package is registered, so you can install it as usual:

```julia
pkg> add ThreadTeams
```

It needs Julia 1.10 or newer and has no dependencies besides ScopedValues.jl; the Enzyme
rules load with EnzymeCore.jl. To run `tforeach`, `tmapreduce`, `@tasks`, ... on the team,
install OhMyThreads.jl, which depends on this package.

> **Version 2** only provides the pool. Version 1 depended on OhMyThreads.jl, re-exported
> its API and defined the schedulers; these now live in OhMyThreads.jl. Replace
> `using ThreadTeams` by `using OhMyThreads` (and keep `using ThreadTeams` for the pool
> controls). `ThreadTeams.nchunks` is gone.

## When to use it

- **Use it** for many small to medium parallel loops in a row, where the 2–4 µs per loop of
  `Threads.@threads` is a noticeable part of the run time. A typical case is the right-hand
  side of a PDE solver: 10–20 loops of a few µs each per call.
- **Use OhMyThreads' own schedulers** (`DynamicScheduler`, the default) for large loops,
  where a few µs per loop do not matter, for loop bodies that are unbalanced beyond what
  `GreedyTeamScheduler` handles or that yield (I/O, locks, `wait`), and for composable
  task-based parallelism or `@barrier`. Switching is one keyword argument.
- **Compared to Polyester.jl's `@batch`**: the same design (contiguous chunks, persistent
  spinning workers, nested loops run serially) at a slightly higher cost per loop (a few
  tens of nanoseconds). No LLVM-level threading runtime, no conversion of arrays to
  `PtrArray`s; exceptions propagate, and the loop body is an ordinary closure. `@batch`
  wraps the loop body in `@inbounds`; the team keeps bounds checks
  unless the body has its own `@inbounds`. Compare the two with `@inbounds` in the body.

## Usage

With OhMyThreads.jl, pass one of its team schedulers:

```julia
using OhMyThreads   # start Julia with threads, e.g. `julia -t auto`

x = rand(10_000)
y = similar(x)

const team = TeamScheduler()
tforeach(eachindex(x); scheduler = team) do i
    y[i] = sqrt(x[i])
end
tmapreduce(abs2, +, x; scheduler = team)                         # parallel mapreduce
tmap(sqrt, x; scheduler = GreedyTeamScheduler(chunksize = 256))  # load balancing
@tasks for i in eachindex(x)
    @set scheduler = team
    y[i] = 2x[i]
end
```

With this package alone, split the work yourself:

```julia
using ThreadTeams

ThreadTeams.foreach_range(length(x)) do lo, hi     # one contiguous range per thread
    for i in lo:hi
        y[i] = sqrt(x[i])
    end
end
ThreadTeams.foreach_chunk(c -> work(c), 100)       # chunks 1:100, taken as threads become free
ThreadTeams.run_fold(c -> sum(abs2, view(x, c:8:length(x))), +, 8, ThreadTeams.GreedySchedule())
```

### API

The package exports nothing.

| | |
|---|---|
| `ThreadTeams.foreach_chunk(f, m, [p])` | call `f(c)` for `c in 1:m` on at most `p` threads (serially when the team is busy) |
| `ThreadTeams.foreach_range(f, n; threads, schedule, grain)` | call `f(first, last)` over `1:n` with fixed block owners; static or stealing schedule |
| `ThreadTeams.run_chunks(f, n, [schedule])` | call `f(c)` for `c in 1:n`; on tasks or serially when the team is busy, as the schedule says |
| `ThreadTeams.run_fold(f, op, n, [schedule])` | fold the `f(c)` with `op`, in chunk order |
| `ThreadTeams.StaticSchedule(; busy)` | chunk `c` on worker `c` (at most `nworkers()` chunks); `busy = :serial` runs loops that find the team busy serially instead of on tasks |
| `ThreadTeams.GreedySchedule(; busy)` | any number of chunks, taken by the caller and the workers as they become free |
| `ThreadTeams.runs_serially(schedule)` | whether a loop started now would run serially (nested, busy, one thread) |
| `ThreadTeams.nworkers()` | size of the team, `Threads.nthreads(:default)` |
| `ThreadTeams.set_spin_time!(seconds)` / `spin_time()` | how long idle workers spin after a loop before they sleep (default 100 µs) |
| `ThreadTeams.parallel_phase(f)` | run `f()` with idle workers spinning until it returns (no sleep between loops) |
| `ThreadTeams.shutdown!()` | stop the workers; the next loop starts them again |

### Fixed placement: `foreach_range`

`ThreadTeams.foreach_range(f, n)` calls `f(first, last)` on inclusive ranges covering
`1:n`. It assigns balanced contiguous blocks to fixed pool workers, matching
`Threads.@threads :static`, independent of the caller's thread. A caller on a participating
worker's thread runs that block; otherwise it waits. Late workers' blocks may be rescued.

Use this for NUMA first-touch locality across loops with the same size and thread count.
Pin threads with ThreadPinning.jl (e.g. `pinthreads(:numa)`; use `threadpool = :all` to
include the interactive thread, or start Julia with `-t N,0`). Rescue, stealing and caller
migration after a yield can change execution placement.

```julia
ThreadTeams.foreach_range(length(x)) do lo, hi          # static first touch
    @views x[lo:hi] .= 0.0
end
ThreadTeams.foreach_range(length(x); schedule = :steal) do lo, hi
    for i in lo:hi
        x[i] = work(i)
    end
end
```

- `schedule = :static` (default): one call per block, for balanced work.
- `schedule = :steal`: claims ranges of at most `grain` indices and steals remaining work
  from other blocks' tails. The default grain is currently an eighth of a block.
  Unstarted owners get time to wake; started blocks may lose work to thieves.
  Inputs of `2^31` indices or more use static mode.

Nonempty input runs as one serial `f(1, n)` call when using one participant, the team is
busy, the call is nested, a `Threads.@threads` loop is active, or during precompilation.
After failure, started calls finish before the first recorded exception is rethrown.
See the [`foreach_range` reference](https://vchuravy.github.io/ThreadTeams.jl/dev/api/#ThreadTeams.foreach_range)
for limits, grain and exception semantics.

The [documentation](https://vchuravy.github.io/ThreadTeams.jl/dev) covers the design
(chunking, stealing, serial fallback, exceptions, dynamic scope, reductions), the measured
overheads, and the trade-offs and caveats (CPU use while idle, hybrid CPUs, automatic
differentiation, captured variables).

## Development

CI checks that the code is formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl)
(`julia -m Runic --inplace src test ext docs/make.jl benchmark`). A pre-commit hook in
`.githooks` checks the staged files; enable it with `git config core.hooksPath .githooks`.

## License

MIT, see [LICENSE](LICENSE).

## AI Usage
This package was developed with Claude Code Opus 5.5 and Fable 5.1
