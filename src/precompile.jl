# Precompiles the loop-body-independent parts of the team, so a session's first parallel
# loop does not pay for them (about 0.6 s of 0.9 s on Julia 1.12; the rest is the
# specialization on the loop body).
#
# Parallel loops run serially during precompilation (`generating_output`), so a workload
# alone never reaches the pool: the worker tasks, command hand-off and waiting loops need
# explicit statements. `run_tasks` is not specialized on the loop body, so one statement
# covers it. The region functions (`run_chunks`, `run_fold`, `dispatch!`, ...) are
# specialized on the chunk function: packages that build loops on them precompile them for
# sample loop bodies (as OhMyThreads.jl does for its team schedulers).

precompile(run_tasks, (Any, Int))
precompile(Pool, ())
precompile(new_pool, ())
precompile(Tuple{WorkerCode})
precompile(worker_loop, (Pool, Int))
precompile(wait_for_command, (WorkerState, SleepFlag, Base.Event, Bool, Bool, RangeState))
precompile(wait_idle, (Pool, Int))
precompile(is_oversubscribed, ())
precompile(shutdown!, ())
precompile(set_spin_time!, (Float64,))
precompile(spin_time, ())
precompile(nworkers, ())

precompile(StaticSchedule, ())
precompile(GreedySchedule, ())
precompile(runs_serially, (StaticSchedule,))
precompile(runs_serially, (GreedySchedule,))
precompile(run_off_pool, (Function, Int, GreedySchedule))
precompile(throw_greedy_errors, (Vector{Any}, Vector{Int}, Int))

if ccall(:jl_generating_output, Cint, ()) == 1
    parallel_phase(() -> nothing)
end
