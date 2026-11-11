# ThreadTeams.jl

A team of persistent worker tasks, one per thread, that runs the chunks of parallel loops
with low per-loop overhead. It is the pool behind the `TeamScheduler` (`:team`) and
`GreedyTeamScheduler` (`:greedyteam`) of
[OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl) and the CPU back-end of
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

!!! note "Version 2"
    Version 2 only provides the pool. Version 1 depended on OhMyThreads.jl, re-exported
    its API and defined the schedulers; these now live in OhMyThreads.jl. Replace
    `using ThreadTeams` by `using OhMyThreads` (and keep `using ThreadTeams` for the pool
    controls). `ThreadTeams.nchunks` is gone.

One sticky worker task runs per thread of the `:default` thread pool. A parallel loop
splits its range into one contiguous chunk per thread, runs chunk 1 on the calling task and
hands the rest to the workers. After a loop, idle workers spin (100 µs by default) before
they sleep, so back-to-back loops do not pay for creating, scheduling and waking tasks.
Starting a loop costs about 0.3 µs and allocates nothing after its first run, against
2–4 µs and about 3 KB for
`Threads.@threads` or OhMyThreads' default scheduler (see [Overhead](@ref)).

## Installation

The package is registered, so you can install it as usual:

```julia
pkg> add ThreadTeams
```

It needs Julia 1.10 or newer and has no dependencies besides ScopedValues.jl; the Enzyme
rules load with EnzymeCore.jl. To run `tforeach`, `tmapreduce`, `@tasks`, ... on the team,
install OhMyThreads.jl, which depends on this package.

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

Pass the scheduler as an object: OhMyThreads resolves a symbol (`scheduler = :team`) with a
dynamic dispatch, about 0.3 µs and 64 bytes per loop.

With this package alone, split the work yourself, with
[`ThreadTeams.foreach_range`](@ref), [`ThreadTeams.foreach_chunk`](@ref), or
[`ThreadTeams.run_chunks`](@ref) and [`ThreadTeams.run_fold`](@ref):

```julia
using ThreadTeams

ThreadTeams.foreach_range(length(x)) do lo, hi     # one contiguous range per thread
    for i in lo:hi
        y[i] = sqrt(x[i])
    end
end
ThreadTeams.foreach_chunk(c -> work(c), 100)       # chunks 1:100, taken as threads become free
```

## When to use it

- **Use it** for many small to medium parallel loops in a row, where the per-loop overhead
  of `Threads.@threads` (2–4 µs, or about 100 µs with oversubscription) is a noticeable part
  of the run time. A typical case is the right-hand side of a PDE solver: 10–20 loops of a
  few µs each per call.
- **Use OhMyThreads' own schedulers** (`DynamicScheduler`, the default) for large loops,
  where a few µs per loop do not matter, for loop bodies that are unbalanced beyond what
  `GreedyTeamScheduler` handles or that yield (I/O, locks, `wait`), and for
  composable task-based parallelism or `@barrier`. Switching is one keyword argument.

With Enzyme.jl loaded, operations on the team can be differentiated in forward and reverse
mode.

See [Design](@ref) for the semantics of the team (chunking, stealing, serial fallback,
exceptions, dynamic scope, reductions), [Overhead](@ref) for measurements, and
[Trade-offs and caveats](@ref) for CPU use while idle, hybrid CPUs, automatic
differentiation and captured variables.
