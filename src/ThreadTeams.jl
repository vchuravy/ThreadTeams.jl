"""
    ThreadTeams

A pool of persistent ("parked") worker tasks that runs OhMyThreads.jl's parallel functions
(`tforeach`, `tmapreduce`, `tmap`, `@tasks`, ...) with low per-loop overhead. The package
re-exports the OhMyThreads.jl API and adds two schedulers:

- [`TeamScheduler`](@ref) (`scheduler = :team`): one contiguous chunk per thread of the
  `:default` thread pool, chunk 1 on the calling task, the others on the workers;
- [`GreedyTeamScheduler`](@ref) (`scheduler = :greedyteam`): more chunks than
  threads, taken by the caller and the workers as they become free (load balancing).

One sticky worker task runs per thread of the `:default` thread pool. After a loop, idle
workers spin for a short while (see [`ThreadTeams.set_spin_time!`](@ref)) before they
sleep, so back-to-back loops pay for no task creation, scheduling or wake-up.

Pool controls: [`ThreadTeams.nworkers`](@ref), [`ThreadTeams.set_spin_time!`](@ref),
[`ThreadTeams.spin_time`](@ref), [`ThreadTeams.parallel_phase`](@ref),
[`ThreadTeams.shutdown!`](@ref).

With Enzyme.jl loaded, parallel operations on the team can be differentiated in forward
and reverse mode (a package extension; see the README).
"""
module ThreadTeams

using OhMyThreads
using OhMyThreads: Scheduler
import ScopedValues

# Re-export the OhMyThreads.jl API
for name in names(OhMyThreads)
    name === :OhMyThreads && continue
    @eval export $name
end
export TeamScheduler, GreedyTeamScheduler

include("pool.jl")
include("chunks.jl")
include("schedulers.jl")
include("precompile.jl")

end # module
