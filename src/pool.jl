# The worker pool.
#
# There is one sticky worker task per thread of the `:default` thread pool. Worker `k`
# runs on thread `tids[k]` and waits for commands in its own `WorkerState` (on a cache
# line of its own, see `CACHE_LINE`). A command is the index of the chunk to run (packed
# with the region data the worker needs, see `cmd_base`), `0` means idle, `RUNNING` that
# the worker claimed its chunk and `STOP` asks the worker to exit.
#
# A parallel region is started by `dispatch!`:
#
# 1. The chunk function `f::F` is stored into pre-allocated per-type `Slot{F}`s (one
#    element of a `Memory{F}` each, emptied after the region with `unsetindex!`): the
#    caller's, and one per worker unless `f` is large (`COPY_BYTES`). No boxing of `f`.
#    Workers run it through a closure per type `F`, created in the caller's world age
#    (`job_closure`), which finds the worker's slot from the worker's index.
# 2. The caller publishes its dynamic scope (if any), then stores the closure and the
#    command (chunk index, number of chunks, see `cmd_base`) into the `WorkerState` of the
#    workers of chunks `2:n` (release stores, then one fence), in decreasing chunk order.
#    The worker on the caller's thread (if any) is skipped.
# 3. Sleeping workers are woken (`Event`) in a binary tree, as in the tree barrier of
#    Mellor-Crummey & Scott (1991): whoever claims chunk `c` wakes the workers of chunks
#    `2c` and `2c + 1` if they sleep, so the caller wakes at most two itself.
# 4. Workers run their chunk in the caller's dynamic scope (`ScopedValue`s, the logger)
#    and world age.
# 5. The caller runs chunk 1, then "steals" every chunk its worker has not claimed (a
#    compare-and-swap of the command from `chunk` to `0`; a worker claims with a swap to
#    `RUNNING`) and runs it itself (`steal_chunks`), so the region does not wait for a
#    worker that is asleep or not scheduled by the OS. A greedy region only takes such
#    chunks back (`take_back_chunks`). Finally it waits until every worker it used is idle,
#    also when a chunk threw.
#
# Only one region is active at a time (the `busy` flag, taken with a compare-and-swap). A
# loop nested in a region runs serially; one started concurrently by another task runs on
# `Threads.@spawn`ed tasks.
#
# Memory orderings (checked by the model in tla/README.md):
# - The region data (the closure, the scope, the workers' slots) is stored with relaxed
#   stores and published by the release store of each command; the worker's sequentially
#   consistent claim (an acquire) makes it visible. Relaxed loads suffice in the spin loop.
# - A worker's results (`errs`, the partials buffer) are published by its release store
#   `cmd = 0` and acquired by the load of `wait_idle`.
# - The sleep handshake (Dekker): the worker stores `sleeping = true`, then loads `cmd`,
#   both sequentially consistent; the dispatcher's command stores are followed by an sc
#   fence and the loads of `wake_chunk` are sc. Either the worker sees its command, or the
#   waker sees that it sleeps.
# - Everything else (the steal, the re-notify of `wait_idle`, `sleeping = false`) is
#   relaxed: a stale value costs a spurious notification or a chunk run by the caller.
#
# With the `GreedyTeamScheduler` the loop has more chunks than workers: the region runs a
# `GreedyJob` on `p` participants (the caller and `p - 1` workers), each claiming chunk
# indices from a shared counter until none are left.

const STOP = -1
const RUNNING = -2

# A command packs the region data a worker needs before it runs its chunk, so that one
# cache line brings all of it: the chunk index, the number of chunks and the skipped worker
# (for `wake_children`), and whether the caller has a dynamic scope (then loaded from the
# pool). Commands are positive; `0`, `STOP` and `RUNNING` are not commands.
const CMD_BITS = Sys.WORD_SIZE == 64 ? 20 : 9
const CMD_MASK = (1 << CMD_BITS) - 1
const SCOPE_BIT = 1 << (Sys.WORD_SIZE - 2)
@inline cmd_base(n::Int, skip::Int, has_scope::Bool) =
    (n << CMD_BITS) | (skip << (2 * CMD_BITS)) | (has_scope ? SCOPE_BIT : 0)
@inline cmd_chunk(cmd::Int) = cmd & CMD_MASK
@inline cmd_nchunks(cmd::Int) = (cmd >> CMD_BITS) & CMD_MASK
@inline cmd_skip(cmd::Int) = (cmd >> (2 * CMD_BITS)) & CMD_MASK
@inline cmd_has_scope(cmd::Int) = (cmd & SCOPE_BIT) != 0

"""
    TeamScheduler()

OhMyThreads.jl scheduler (also `scheduler = :team`) that runs a parallel operation on the
worker pool: one contiguous chunk per thread of the `:default` pool, chunk 1 on the
calling task and chunk `c` on worker `c`. The calling task also runs every chunk whose
worker has not picked it up in time (asleep, or not scheduled by the OS), so a loop never
waits for an idle worker. Lowest overhead; best for balanced iterations.

Only one parallel operation runs on the pool at a time. One started while another task
uses the pool runs on `Threads.@spawn`ed tasks; one started inside a chunk (nested) runs
serially. Chunks run in the caller's dynamic scope (`ScopedValue`s, logger) and world
age. If chunks throw, the exception of the first failing chunk is rethrown once all
started chunks have finished.

The task index (`OhMyThreads.WithTaskIndex`, `@task_index`) is the chunk index; for
`chunks`/`index_chunks` inputs the position of the chunk, as with the other schedulers.
Task-local values (`@local`) live in the persistent worker tasks: initialized once per
worker, reused across loops.

No keyword arguments. `ntasks` in particular is not supported: chunks are not guaranteed
to run at the same time, so `@barrier` can not be used.
"""
struct TeamScheduler <: Scheduler
    function TeamScheduler(; kwargs...)
        isempty(kwargs) || unsupported_kwargs("TeamScheduler", kwargs)
        return new()
    end
end

"""
    GreedyTeamScheduler(; nchunks = 0, chunksize = 0)

OhMyThreads.jl scheduler (also `scheduler = :greedyteam`) that runs a parallel operation
on the worker pool with load balancing: `nchunks` contiguous chunks (default
`4 * nworkers()`), or chunks of `chunksize` elements, which the calling task and the
workers take from a shared counter until none are left. For loops whose iterations take
very different times. See [`TeamScheduler`](@ref) for the semantics both share.

Chunks are contiguous and reductions combine the chunk results in chunk order, so the
operator only has to be associative and the result is deterministic for a given number of
chunks. The task index is the chunk index.
"""
struct GreedyTeamScheduler <: Scheduler
    nchunks::Int
    chunksize::Int
    function GreedyTeamScheduler(;
            nchunks::Integer = 0, chunksize::Integer = 0,
            kwargs...
        )
        isempty(kwargs) || unsupported_kwargs("GreedyTeamScheduler", kwargs)
        (nchunks >= 0 && chunksize >= 0) ||
            throw(ArgumentError("nchunks and chunksize must be non-negative"))
        (nchunks > 0 && chunksize > 0) &&
            throw(ArgumentError("nchunks and chunksize are mutually exclusive"))
        return new(nchunks, chunksize)
    end
end

const TeamSchedulers = Union{TeamScheduler, GreedyTeamScheduler}

@noinline function unsupported_kwargs(name, kwargs)
    ks = join(("`$k`" for k in keys(kwargs)), ", ")
    if :ntasks in keys(kwargs)
        throw(
            ArgumentError(
                "$name does not take `ntasks`: chunks run on a fixed pool " *
                    "of workers and the calling task runs chunks that no worker " *
                    "has picked up, so the number of tasks running at the same " *
                    "time is not fixed (`@barrier` is not supported)"
            )
        )
    end
    throw(ArgumentError("$name does not take the keyword argument$(length(kwargs) > 1 ? "s" : "") $ks"))
end

# Padding around the atomic fields the workers and the dispatcher hammer (a worker's
# command, its sleep flag, the greedy counter), so that each has a cache line no other
# field or neighbouring object shares: the field sits `CACHE_LINE - 8` bytes into the
# object with `CACHE_LINE` bytes after it, so its line lies within the object at any
# 16-byte aligned address. (A worker's command and closure form a 16-byte pair at
# `CACHE_LINE - 16`.) Apple silicon has 128-byte lines, x86 and other AArch64 cores 64;
# padding to 128 on every AArch64 CPU costs a few hundred bytes.
const CACHE_LINE = Sys.ARCH === :aarch64 ? 128 : 64
const PAD_BEFORE = CACHE_LINE ÷ sizeof(Int) - 1
const PAD_AFTER = CACHE_LINE ÷ sizeof(Int)
const PadBefore = NTuple{PAD_BEFORE, Int}
const PadAfter = NTuple{PAD_AFTER, Int}
pad_before() = ntuple(_ -> 0, Val(PAD_BEFORE))
pad_after() = ntuple(_ -> 0, Val(PAD_AFTER))

# Shared chunk counter of a `GreedyTeamScheduler` region, on its own cache line
mutable struct ChunkCounter
    const _pad0::PadBefore
    @atomic next::Int
    const _pad1::PadAfter
end
ChunkCounter() = ChunkCounter(pad_before(), 0, pad_after())

# The dispatcher stores the chunk function into a slot of each worker (`worker_slots`)
# with a typed store (so with the GC's write barrier: the slots are old, the captures may
# be young), as libgomp's master writes `thr->fn`/`thr->data` per thread, if it takes at
# most `COPY_BYTES`; a larger one is run from the dispatcher's slot by every worker.
# Reading the dispatcher's slot, rewritten in every region, costs every worker a second
# cache miss after the one that brings the command (about 40 ns with 4 threads); its own
# slot it reads from the dispatcher's cache in parallel with the command. (Loading the
# lines of the slots used last while waiting, to have them arrive with the command, was
# measured at no gain: the loads delay the claim by about as much as they save.)
const COPY_BYTES = 256

# A slot for a chunk function of type `F` (`job_slot`, `worker_slots`): one element of a
# `Memory{F}` (a `Vector{F}` on Julia 1.10) with a cache line of unused elements on each
# side, since it is written in every region. Stored with a typed `setindex!` (so with the
# GC's write barrier) and emptied with `unsetindex!` (`clear_slot!`), after which
# `isassigned(slot)` is false. On Julia ≥ 1.11 the slot holds a `MemoryRef` to the
# element, one load away; on 1.10 the vector and the index.
@static if isdefined(Core, :Memory)
    struct Slot{F}
        ref::MemoryRef{F}
    end
    function Slot{F}() where {F}
        pad = slot_padding(F)
        return Slot{F}(Core.memoryref(Memory{F}(undef, 2 * pad + 1), pad + 1))
    end
    @inline Base.getindex(slot::Slot) = slot.ref[]
    @inline Base.setindex!(slot::Slot{F}, f) where {F} =
        (Core.memoryrefset!(slot.ref, f, :not_atomic, false); slot)
    Base.isassigned(slot::Slot) = Core.memoryref_isassigned(slot.ref, :not_atomic, false)
    @inline unset_slot!(slot::Slot) = unsetindex!(slot.ref)
    slot_memory(slot::Slot) = slot.ref.mem
    slot_index(slot::Slot) = Base.memoryrefoffset(slot.ref)
else
    struct Slot{F}
        mem::Vector{F}
        i::Int
    end
    function Slot{F}() where {F}
        pad = slot_padding(F)
        return Slot{F}(Vector{F}(undef, 2 * pad + 1), pad + 1)
    end
    @inline Base.getindex(slot::Slot) = @inbounds slot.mem[slot.i]
    @inline Base.setindex!(slot::Slot{F}, f) where {F} = (@inbounds slot.mem[slot.i] = f; slot)
    Base.isassigned(slot::Slot) = isassigned(slot.mem, slot.i)
    @inline unset_slot!(slot::Slot) = unsetindex!(slot.mem, slot.i)
    slot_memory(slot::Slot) = slot.mem
    slot_index(slot::Slot) = slot.i
end
# Elements of type `F` per cache line, plus one
slot_padding(::Type{F}) where {F} = cld(CACHE_LINE, max(Base.aligned_sizeof(F), 1)) + 1
# `Base.unsetindex!` is public since Julia 1.14; before, the internal `_unsetindex!` is
# the same function
const unsetindex! = isdefined(Base, :unsetindex!) ? Base.unsetindex! : Base._unsetindex!
# Bytes of an element of type `F`: the value if stored inline, else a reference
@inline field_bytes(::Type{F}) where {F} = sizeof(Base.RefValue{F})
# Whether chunk functions of type `F` go into the workers' own slots (`COPY_BYTES`)
@inline own_slots(::Type{F}) where {F} = field_bytes(F) <= COPY_BYTES
# A worker's command and the closure of the region (`job_closure`), a 16-byte pair at a
# multiple of 16 bytes on a cache line of their own: one line transfer brings the worker
# both, and the rest of the region data inside the command (`cmd_base`).
mutable struct WorkerState
    const _pad0::NTuple{PAD_BEFORE - 1, Int}
    @atomic cmd::Int
    @atomic job::Any
    const _pad1::PadAfter
end
WorkerState() = WorkerState(ntuple(_ -> 0, Val(PAD_BEFORE - 1)), 0, nothing, pad_after())

# Whether a worker is asleep (or about to be), on a line of its own apart from its command:
# written only when the worker sleeps or wakes, so the two loads per chunk in `wake_chunk`
# (on every worker's critical path) hit a line that stays shared, not the line the worker
# is claiming its chunk on.
mutable struct SleepFlag
    const _pad0::PadBefore
    @atomic sleeping::Bool
    const _pad1::PadAfter
end
SleepFlag() = SleepFlag(pad_before(), false, pad_after())

mutable struct Pool
    # Region data that does not travel with the commands (`cmd_base`): the dispatcher's
    # dynamic scope, `nothing` when the caller has none. Written only when it changes, so
    # the line stays shared. (The world age travels with the closure, `job_closure`.)
    @atomic scope::Any
    const workers::Vector{WorkerState}
    const events::Vector{Base.Event}
    const tasks::Vector{Task}
    # `errs[k]`: `CapturedException` thrown by the chunk worker `k` ran, or `nothing`
    const errs::Vector{Any}
    # `tid2worker[tid]`: index of the worker running on thread `tid`, or `0`
    const tid2worker::Vector{Int}
    const sleepflags::Vector{SleepFlag}
    # Greedy regions: the chunk counter, and per participant the exception it caught and
    # the chunk that threw it
    const counter::ChunkCounter
    const greedy_errs::Vector{Any}
    const greedy_errchunks::Vector{Int}
    # more Julia threads than CPUs available to the process: spinning workers then give
    # their CPU to other OS threads now and then
    const oversubscribed::Bool
    # Chunks the dispatcher took over from their worker (`steal_chunks`); written only by
    # the dispatcher, which holds `busy` (relaxed, no read-modify-write)
    @atomic stolen::Int
end

# Global state. `busy` is held by the task running a parallel region (and by `shutdown!`);
# `pool` is only created, replaced or used while holding `busy`.
mutable struct PoolHolder
    @atomic busy::Bool
    # Task that holds `busy` (to detect `shutdown!` calls from inside a region)
    @atomic owner::Any
    @atomic pool::Union{Nothing, Pool}
end
const POOL = PoolHolder(false, nothing, nothing)

# Spin time of an idle worker before it goes to sleep, in nanoseconds
const SPIN_NS = Threads.Atomic{Int}(100_000)
# Number of active `parallel_phase`s; while > 0, idle workers do not go to sleep on time
const PHASE = Threads.Atomic{Int}(0)

"""
    ThreadTeams.set_spin_time!(seconds::Real)

Set how long an idle worker busy-waits for the next parallel loop before it sleeps;
default `100e-6` (100 µs). The time counts from the end of the loop: a worker that
finished early keeps spinning while the loop waits for the other chunks. Takes effect the
next time a worker starts waiting.

Longer spinning makes loops that follow each other with short serial gaps cheaper (waking
a sleeping worker costs about 10 µs), but idle workers keep their CPU busy for that long
after every loop. With `0`, workers sleep as soon as they notice the pool is idle, which
they check every 64 spins (about 100 ns), so loops closer than that keep them awake
(lowest CPU use, highest latency). To spin only during a part of the program, use
[`ThreadTeams.parallel_phase`](@ref).

On CPUs with performance and efficiency cores (recent Intel, Apple silicon), spinning
cores take power and turbo headroom from the busy ones; LLVM's and GCC's OpenMP runtimes
do not spin there by default. Consider `set_spin_time!(0)`.
"""
function set_spin_time!(seconds::Real)
    seconds >= 0 || throw(ArgumentError("spin time must be non-negative"))
    SPIN_NS[] = round(Int, seconds * 1.0e9)
    return seconds
end

"""
    ThreadTeams.spin_time() -> Float64

Current spin time of idle workers in seconds, see [`set_spin_time!`](@ref).
"""
spin_time() = SPIN_NS[] / 1.0e9

"""
    ThreadTeams.nworkers() -> Int

Maximum number of chunks a parallel loop is split into: the number of threads of the
`:default` thread pool (`Threads.nthreads(:default)`).
"""
nworkers() = Threads.threadpoolsize(:default)

@inline cpu_pause() = ccall(:jl_cpu_pause, Cvoid, ())

@inline function os_yield()
    @static if Sys.iswindows()
        ccall(:SwitchToThread, stdcall, Cint, ())
    else
        ccall(:sched_yield, Cint, ())
    end
    return nothing
end

# Does this thread have another task to run? Then an idle worker sleeps at once so that
# the task gets the thread. (Scheduler internals; without them, workers sleep only after
# the spin time.)
@inline function other_work()
    @static if isdefined(Base, :workqueue_for) && isdefined(Base, :Partr) &&
            isdefined(Base.Partr, :multiq_check_empty)
        return !isempty(Base.workqueue_for(Threads.threadid())) ||
            !Base.Partr.multiq_check_empty()
    else
        return false
    end
end

# More Julia threads than CPUs this process may run on (affinity mask), e.g. `-t N` on N
# cores with Julia ≥ 1.12, which adds an interactive thread
function is_oversubscribed()
    nt = Threads.threadpoolsize(:default) + Threads.threadpoolsize(:interactive)
    return nt > Int(ccall(:uv_available_parallelism, Cuint, ()))
end

# The caller's dynamic scope, and a macro to run code in it. Julia ≥ 1.11 keeps
# `ScopedValue`s and the logger in the task's scope; on 1.10, ScopedValues.jl and
# `with_logger` use the task's `logstate`. (A running task's scope can not be set
# directly; the `tryfinally` form installs a handler, which `run_job_in_scope` skips when
# there is no scope to enter.)
@static if isdefined(Core, :current_scope)
    @inline current_scope() = Core.current_scope()
    macro enter_scope(scope, ex)
        return Expr(:tryfinally, esc(ex), nothing, esc(scope))
    end
else
    @inline current_scope() = current_task().logstate
    macro enter_scope(scope, ex)
        return quote
            local ct = current_task()
            local old = ct.logstate
            ct.logstate = $(esc(scope))
            try
                $(esc(ex))
            finally
                ct.logstate = old
            end
        end
    end
end

@static if isdefined(Base, :tls_world_age)
    @inline world_age() = Base.tls_world_age()
else
    @inline world_age() = ccall(:jl_get_tls_world_age, UInt, ())
end

@static if isdefined(Base, :generating_output)
    @inline generating_output() = Base.generating_output()
else
    @inline generating_output() = ccall(:jl_generating_output, Cint, ()) != 0
end

# Run chunk `chunk` with the chunk function in `slot`, for the dispatcher and (in the
# closure body) the workers. Not inlined, so that every chunk runs the same machine code:
# inlined into `dispatch!` and the closure, the dispatcher's copy of the loop body can
# differ in layout and speed by a few percent, and `dispatch!` compiles slower.
@noinline function run_chunk(slot::Slot{F}, chunk::Int) where {F}
    slot[](chunk)
    return nothing
end

# Generic call (`jl_f_opaque_closure_call`, valid from the worker's older world) of a
# closure that is not a `JobClosure` (`new_job_closure`); `run_job_in_scope` calls the
# usual type directly.
@inline function run_job(job, k::Int, chunk::Int)
    job(k, chunk)
    return nothing
end

# Run the chunk of command `cmd`, claimed by worker `k` with state `ws`, in the
# dispatcher's dynamic scope and world age. The closure and scope are read here, not
# passed: `Any`-typed arguments would make the call dynamic and box them.
function run_job_in_scope(pool::Pool, ws::WorkerState, k::Int, cmd::Int)
    job = @atomic :monotonic ws.job
    chunk = cmd_chunk(cmd)
    # The type check is here: `run_job` called with the `Any`-typed closure is itself a
    # dynamic call (`ijl_apply_generic`) ahead of the direct call of the closure's body
    if job isa JobClosure
        if !cmd_has_scope(cmd)
            # The common case: no scope to enter, no exception handler for it
            job(k, chunk)
        else
            scope = @atomic :monotonic pool.scope
            @enter_scope scope job(k, chunk)
        end
    elseif !cmd_has_scope(cmd)
        run_job(job, k, chunk)
    else
        scope = @atomic :monotonic pool.scope
        @enter_scope scope run_job(job, k, chunk)
    end
    return nothing
end

# Run chunk `chunk` from the worker's own slot and drop the chunk function from it
# afterwards (when the chunk throws, the dispatcher does, see `dispatch!`)
@inline function run_job_own(slot::Slot{F}, chunk::Int) where {F}
    run_chunk(slot, chunk)
    clear_slot!(slot)
    return nothing
end

# Wait until `ws.cmd` is non-zero: spin for `SPIN_NS` after the end of the region (or while
# a `parallel_phase` is active), then sleep on `ev`; sleep at once if another task could
# run on this thread.
function wait_for_command(
        ws::WorkerState, flag::SleepFlag, ev::Base.Event, oversubscribed::Bool
    )
    spins = 0
    t0 = time_ns()
    tyield = tevents = t0
    while true
        # Relaxed: the claim below acquires the region data (published by the release store
        # of the command)
        cmd = @atomic :monotonic ws.cmd
        if cmd > 0
            # Claim the chunk; fails if the dispatcher stole it in the meantime.
            # Sequentially consistent: a full barrier before the loads of `wake_children`
            # (the Dekker handshake with the children of the chunk, see `wake_children`)
            (@atomicreplace :sequentially_consistent :monotonic ws.cmd cmd => RUNNING).success &&
                return cmd
        elseif cmd == STOP
            return cmd
        end
        spins += 1
        if spins & 63 == 0
            now = time_ns()
            pool_busy() && (t0 = now)
            spun_out = now - t0 > SPIN_NS[]
            if (spun_out && PHASE[] == 0) || other_work()
                # Re-check after announcing the sleep: either we see the command here, or
                # the waker sees `sleeping` and notifies `ev`
                @atomic flag.sleeping = true
                if (@atomic ws.cmd) <= 0
                    wait(ev)
                end
                # Relaxed: a waker that still sees `true` only notifies once too often
                @atomic :monotonic flag.sleeping = false
                spins = 0
                t0 = tyield = tevents = time_ns()
                continue
            end
            if spun_out && now - tevents > 20_000
                # Spinning longer than usual (a `parallel_phase`): run the libuv event loop
                # now and then, as the scheduler would on an idle thread; otherwise, with a
                # worker on every thread, timers and I/O would never complete
                Base.process_events()
                tevents = now
            end
            if oversubscribed && now - tyield > 5_000
                # Let the OS run the other threads on this core now and then (only with
                # more threads than CPUs: otherwise it only adds context switches)
                os_yield()
                tyield = now
            end
        end
        cpu_pause()
        GC.safepoint()
    end
    return
end

# The code of a worker task; also used to recognize worker tasks
struct WorkerCode
    pool::Any
    k::Int
end
(w::WorkerCode)() = worker_loop(w.pool::Pool, w.k)

function worker_loop(pool::Pool, k::Int)
    while !run_commands(pool, k)
    end
    return
end

# Run commands until `STOP` (returns `true`) or a chunk throws (`false`). One exception
# handler per stretch of commands, not per chunk (an `enter` costs about as much as a
# small chunk). Not inlined: on Julia 1.10 the frame that caught the exception would keep
# it, and the arrays it refers to, alive while the worker waits for the next command.
@noinline function run_commands(pool::Pool, k::Int)
    ws = pool.workers[k]
    ev = pool.events[k]
    flag = pool.sleepflags[k]
    try
        while true
            cmd = wait_for_command(ws, flag, ev, pool.oversubscribed)
            if cmd == STOP
                @atomic :release ws.cmd = 0
                return true
            end
            wake_children(pool, cmd)
            run_job_in_scope(pool, ws, k, cmd)
            @atomic :release ws.cmd = 0
        end
    catch err
        # (The dispatcher clears this worker's slot, see `dispatch!`)
        pool.errs[k] = CapturedException(err, catch_backtrace())
        @atomic :release ws.cmd = 0
    end
    return false
end

function Pool()
    tids = Threads.threadpooltids(:default)
    nw = length(tids)
    tid2worker = zeros(Int, maximum(tids))
    for (k, tid) in enumerate(tids)
        tid2worker[tid] = k
    end
    nw <= CMD_MASK || error("ThreadTeams supports at most $CMD_MASK threads")
    pool = Pool(
        nothing, [WorkerState() for _ in 1:nw],
        [Base.Event(true) for _ in 1:nw], Task[], Any[nothing for _ in 1:nw],
        tid2worker, [SleepFlag() for _ in 1:nw], ChunkCounter(),
        Any[nothing for _ in 1:nw], zeros(Int, nw), is_oversubscribed(), 0
    )
    # A new task inherits its creator's dynamic scope. The workers get none, so they
    # neither keep the first caller's scope alive nor miss the fast path of
    # `run_job_in_scope` for callers without one.
    @enter_scope nothing for (k, tid) in enumerate(tids)
        task = Task(WorkerCode(pool, k))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1)
        push!(pool.tasks, task)
        schedule(task)
    end
    return pool
end

# Is a parallel region active (in this task or any other)? Inlined into every parallel
# loop ahead of the serial fallback, so it must use a relaxed load: LLVM's
# MemoryDependenceAnalysis treats an acquire load as a clobber of all memory, even for
# `!invariant.load` queries ("FIXME: This is overly conservative"), which stops GVN from
# merging loads across it (array sizes checked by the caller) and costs the serial loop
# its constant strides (Trixi.jl: +4 to +8% in several kernels on Julia 1.13). A stale
# value only makes us try the pool, or run one loop serially. Data is published with
# acquire/release in the non-inlined `run_chunks`, `run_fold` and `dispatch!` only.
@inline pool_busy() = @atomic :monotonic POOL.busy

# Set while a task runs a chunk of a region on spawned tasks (`run_tasks`), inherited by
# tasks spawned from there (ScopedValues.jl, `Base.ScopedValues` on Julia ≥ 1.11)
const IN_TASKS_REGION = ScopedValues.ScopedValue(false)

# Is the current task running a chunk (as dispatcher, worker, or `run_tasks` task)? Only
# consulted while the pool is busy (`pool_busy`); cheap loads first, the scope lookup last.
@inline function in_region()
    ct = current_task()
    (@atomic :monotonic POOL.owner) === ct && return true
    (isdefined(ct, :code) && ct.code isa WorkerCode) && return true
    return IN_TASKS_REGION[]
end

# Take the pool for a region (starting it if needed); `nothing` if the loop can not use it.
# Only called from non-inlined functions.
@inline function acquire_pool()
    generating_output() && return nothing
    # Acquire: pairs with the release store of `release_pool`
    (@atomicreplace :acquire :monotonic POOL.busy false => true).success || return nothing
    @atomic :monotonic POOL.owner = current_task()
    pool = @atomic :acquire POOL.pool
    if pool === nothing
        pool = try
            new_pool()
        catch
            release_pool()
            rethrow()
        end
    end
    return pool
end

@noinline function new_pool()
    pool = Pool()
    @atomic :release POOL.pool = pool
    return pool
end

@inline function release_pool()
    @atomic :monotonic POOL.owner = nothing
    @atomic :release POOL.busy = false
    return nothing
end

# The dispatcher's slot for chunk functions of type `F`: pre-allocated, so storing one
# does not box it. Its cache is created when the generated function is first specialized
# for `F` and embedded in its body as a constant, so that getting it costs nothing (an
# `IdDict` lookup costs about 25 ns); the slot itself is created at the first region of a
# session, since a `MemoryRef` embedded in a package image comes out broken (Julia 1.12:
# `unsetindex!` on it segfaults). Loops run serially during precompilation, so no image
# holds a slot. Used only while holding `POOL.busy`, cleared after every region.
@generated function job_slot_cache(::Type{F}) where {F}
    return :($(Base.RefValue{Union{Nothing, Slot{F}}}(nothing)))
end
@inline function job_slot(::Type{F}) where {F}
    cache = job_slot_cache(F)
    slot = cache[]
    slot === nothing && (slot = new_job_slot(F, cache))
    return slot
end
@noinline function new_job_slot(::Type{F}, cache::Base.RefValue) where {F}
    slot = Slot{F}()
    cache[] = slot
    return slot
end

# The workers' slots for chunk functions of type `F` (`COPY_BYTES`): slot `k` is used by
# worker `k` only, so it stays in that worker's cache. The vector is created once per type
# like `job_slot`'s cache, and grown when a region first needs the slots (so an image
# holds it empty).
@generated function worker_slot_vector(::Type{F}) where {F}
    return :($(Slot{F}[]))
end
@inline function worker_slots(::Type{F}, nw::Int) where {F}
    v = worker_slot_vector(F)
    length(v) < nw && grow_slots!(v, nw)
    return v
end
@noinline function grow_slots!(v::Vector{Slot{F}}, nw::Int) where {F}
    while length(v) < nw
        push!(v, Slot{F}())
    end
    return v
end

# The closure the workers call as `job(k, chunk)`: worker `k` runs the chunk from its
# own slot `worker_slot_vector(F)[k]` (`run_job_own`), or from the dispatcher's slot
# `job_slot(F)` for a chunk function larger than `COPY_BYTES` (`run_chunk`).
#
# An opaque closure runs its body in the world age it was created in, so dynamic calls in
# the chunk function dispatch as in a task the dispatcher would have spawned. The workers'
# tasks are older than most chunk functions; `invoke_in_world` instead costs a method
# lookup plus a boxed call per chunk, about 20 ns on each worker's critical path (Julia
# 1.10 to 1.13), against 3 ns for the closure call. `run_job_in_scope` checks the closure
# type and calls its compiled body directly; another type (`new_job_closure`) goes through
# the generic call in `run_job`, whose arguments box without allocating.
#
# One closure per type `F`, cached like `job_slot` and reused while the dispatcher's world
# age is the one it was created in; the first loop after a method definition (`include`,
# `eval`, Revise) creates a new one, about 120 ns with the body compiled. Used only while
# holding `POOL.busy`. It holds the slots of its type, which are empty outside of regions.
const JobClosure = Core.OpaqueClosure{Tuple{Int, Int}, Nothing}

@generated function job_closure_cache(::Type{F}) where {F}
    return :($(Base.RefValue{Any}(nothing)))
end

@inline function job_closure(slot::Slot{F}) where {F}
    cache = job_closure_cache(F)
    job = cache[]
    world = world_age()
    if !(job isa JobClosure && job.world == world)
        # A dynamic call (same world), so that the closure construction is not part of
        # `dispatch!`'s code: Enzyme compiles the primal of the loops its rules run on the
        # team into its own module, and crashes on the construction (Julia 1.10)
        job = Core._call_in_world(world, new_job_closure, slot, worker_slot_vector(F))
        cache[] = job
    end
    return job
end

# The body of the closure (see `job_closure`)
@inline function run_job_closure(
        slot::Slot{F}, wslots::Vector{Slot{F}}, k::Int, chunk::Int
    ) where {F}
    if own_slots(F)
        run_job_own(@inbounds(wslots[k]), chunk)
    else
        run_chunk(slot, chunk)
    end
    return nothing
end

# The closure is built by the runtime (`jl_new_opaque_closure`), its argument types hidden
# from inference so that codegen does not build it: the runtime uses the compiled body as
# the entry point, compiling it on the first loop with a chunk function type. Built by
# codegen, the body lands in the module of `dispatch!`, which Enzyme's transforms do not
# handle (segmentation fault, Julia 1.10), and on 1.12 and 1.13 the closure only gets a
# stub that boxes the arguments and goes through `jl_invoke_oc` (13 ns and 48 bytes per
# call). The return type can be given on Julia ≥ 1.12 only; on 1.10 and 1.11 it is the
# inferred one, `Union{}` for a chunk function that always throws (then not a
# `JobClosure`, see `run_job`).
@static if VERSION >= v"1.12"
    @noinline function new_job_closure(slot::Slot{F}, wslots::Vector{Slot{F}}) where {F}
        return Base.Experimental.@opaque Base.inferencebarrier(Tuple{Int, Int}) -> Nothing (k, chunk) -> run_job_closure(slot, wslots, k, chunk)
    end
else
    @noinline function new_job_closure(slot::Slot{F}, wslots::Vector{Slot{F}}) where {F}
        return Base.Experimental.@opaque Base.inferencebarrier(Tuple{Int, Int}) (k, chunk) -> run_job_closure(slot, wslots, k, chunk)
    end
end

# Drop the chunk function from its slot after the region, so that it and its captures can
# be collected: `unsetindex!` nulls the element's references (removing references needs
# no write barrier); afterwards `isassigned(slot)` is false. Nothing reads a slot outside
# a region.
@inline function clear_slot!(slot::Slot{F}) where {F}
    unset_slot!(slot)
    return nothing
end

# The buffer for the partial results of a reduction, emptied after the region (`empty!`
# nulls boxed elements but keeps the capacity): resizing allocates only when a region
# needs more chunks than any before.
@generated function static_buffer(::Type{T}) where {T}
    return :($(Vector{T}()))
end
@inline function partials_buffer(::Type{T}, n::Int) where {T}
    buf = static_buffer(T)
    length(buf) < n && resize!(buf, n)
    return buf
end

# Spin until worker `k` is idle, yielding now and then so that a worker or chunk sharing
# this thread can make progress. Returns `false` if the worker died with its command set
# (`dead_worker_error`); the caller then waits for the other workers before it throws.
@inline function wait_idle(pool::Pool, k::Int)
    ws = @inbounds pool.workers[k]
    spins = 0
    while (@atomic :acquire ws.cmd) != 0
        spins += 1
        if spins & 1023 == 0
            # Fallback of the handshake in `wake_chunk` (relaxed load): wake the worker if
            # it sleeps with a pending command
            @atomic(:monotonic, (@inbounds pool.sleepflags[k]).sleeping) &&
                notify(@inbounds pool.events[k])
            # A worker task dies only by an exception thrown into it from outside (an
            # interrupt); with its command still set, the region would wait forever
            istaskdone(@inbounds pool.tasks[k]) && return false
            yield()
        else
            cpu_pause()
            GC.safepoint()
        end
    end
    return true
end

@noinline function dead_worker_error(k::Int)
    throw(
        ErrorException(
            "ThreadTeams: worker $k died while running a chunk (its task " *
                "failed); call ThreadTeams.shutdown!() to start a new team"
        )
    )
end

# Index of the worker that runs chunk `chunk` (≥ 2) when worker `skip` is not used
@inline worker_for_chunk(chunk, skip) = (k = chunk - 1; (skip != 0 && k >= skip) ? k + 1 : k)

# Wake the worker of chunk `chunk` if it sleeps with this chunk still pending (`base`: the
# command without the chunk index, see `cmd_base`)
@inline function wake_chunk(pool::Pool, chunk::Int, skip::Int, base::Int)
    k = worker_for_chunk(chunk, skip)
    if (@atomic (@inbounds pool.sleepflags[k]).sleeping) &&
            (@atomic (@inbounds pool.workers[k]).cmd) == base | chunk
        notify(@inbounds pool.events[k])
    end
    return nothing
end

# Called by whoever claims chunk `chunk`, before running it: wake the workers of chunks
# `2chunk` and `2chunk + 1` (a binary tree: `n` sleeping workers wake in O(log n) steps).
# The dispatcher stores the commands in decreasing order, so a child's command precedes
# its parent's. For the dispatcher's wake-ups, the fence after its stores and the sc loads
# here guarantee that either the child sees its command before it sleeps, or we see that
# it sleeps; for a worker, the full barrier of its sc claim does the same on x86 and
# AArch64, though not in the C11 model (tla/README.md). A lost wake-up costs no
# correctness: the dispatcher steals the chunk, and `wait_idle` re-notifies.
@inline function wake_children(pool::Pool, cmd::Int)
    chunk = cmd_chunk(cmd)
    n = cmd_nchunks(cmd)
    skip = cmd_skip(cmd)
    base = cmd & ~CMD_MASK
    c = 2chunk
    c <= n && wake_chunk(pool, c, skip, base)
    c + 1 <= n && wake_chunk(pool, c + 1, skip, base)
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n` in parallel; the caller holds `POOL.busy`.
function dispatch!(pool::Pool, f::F, n::Int) where {F}
    slot = job_slot(F)
    slot[] = f
    job = job_closure(slot)
    tid = Threads.threadid()
    skip = tid <= length(pool.tid2worker) ? @inbounds(pool.tid2worker[tid]) : 0
    # Published by the release stores of the commands; stored only when present, so the
    # line stays shared
    scope = current_scope()
    has_scope = scope !== nothing
    has_scope && (@atomic :monotonic pool.scope = scope)
    # The workers' slots are used, and need to exist, only for a small `f` (`COPY_BYTES`)
    copies = own_slots(F)
    wslots = copies ? worker_slots(F, length(pool.workers)) : worker_slot_vector(F)
    base = cmd_base(n, skip, has_scope)
    try
        # Decreasing chunk order: see `wake_children`
        for chunk in n:-1:2
            k = worker_for_chunk(chunk, skip)
            # A worker can still be busy if a previous region was interrupted
            if !wait_idle(pool, k)
                # Take back the chunks already handed out before giving up
                cancel_chunks(pool, chunk + 1, n, skip, base, wslots, copies)
                wait_workers(pool, chunk + 1, n, skip)
                dead_worker_error(k)
            end
            @inbounds pool.errs[k] = nothing
            # Release stores: their line transfers overlap in the store buffer, where sc
            # stores (`xchg` on x86) would each wait for the line (20% of the dispatcher's
            # time with 8 threads). The fence below orders them before the `sleeping` loads
            # of `wake_children` (Dekker with the worker, which stores `sleeping` and loads
            # `cmd`). A wake-up still missed on a weakly ordered machine is repaired by the
            # re-notify of `wait_idle`; the region only takes longer.
            ws = @inbounds pool.workers[k]
            copies && (@inbounds(wslots[k])[] = f)
            @atomic :monotonic ws.job = job
            @atomic :release ws.cmd = base | chunk
        end
        Threads.atomic_fence()
        # Chunk 1 is ours: wake the workers of chunks 2 and 3, they wake the others
        wake_children(pool, base | 1)
        try
            run_chunk(slot, 1)   # the workers' code, see `run_chunk`
        catch
            # Cancel the chunks that were not started and wait for the others: never leave
            # workers running when returning from the region
            cancel_chunks(pool, 2, n, skip, base, wslots, copies)
            wait_workers(pool, 2, n, skip)
            rethrow()
        end
        # The chunks that their worker has not claimed yet: a greedy region only takes them
        # back, a static one runs them
        if f isa GreedyJob
            take_back_chunks(pool, n, skip, base, wslots, copies)
        else
            steal_chunks(pool, slot, n, skip, base, wslots, copies)
        end
        wait_workers(pool, 2, n, skip)
        for chunk in 2:n
            k = worker_for_chunk(chunk, skip)
            err = @inbounds pool.errs[k]
            err === nothing || throw_worker_error(err::CapturedException)
        end
    finally
        # Keep neither the chunk function (`clear_slot!`), nor the scope, nor the chunks'
        # exceptions alive (every used worker is idle here, also when a chunk threw)
        clear_slot!(slot)
        has_scope && (@atomic :monotonic pool.scope = nothing)
        for chunk in 2:n
            k = worker_for_chunk(chunk, skip)
            # A worker whose chunk threw, or that died with its command set
            # (`dead_worker_error`), left the chunk function in its slot
            if (@inbounds pool.errs[k]) !== nothing ||
                    (@atomic :monotonic (@inbounds pool.workers[k]).cmd) != 0
                copies && clear_slot!(@inbounds wslots[k])
                @inbounds pool.errs[k] = nothing
            end
        end
    end
    return nothing
end

# Run the chunks of a static region their worker has not claimed (libomp's `static_steal`
# schedule, with the dispatcher as the only thief). A spinning worker claims within a
# cache-line transfer, so one that has not is asleep or not running: waiting would cost a
# wake-up or an OS time slice. Locality: a stolen chunk is shorter than that wake-up (the
# dispatcher ran its own chunk first), so it moves at most the data a core touches in that
# time, and fetching it back costs its worker about one chunk.
#
# A load first: a compare-and-swap on a line the worker just wrote takes it from the
# worker's cache (and its final `cmd = 0` takes it back); a load only shares it. Relaxed:
# nothing of the worker's is read after a steal; `wait_workers` acquires the results of
# the workers that did claim. The worker of a stolen chunk never runs it, so the
# dispatcher clears its slot (still in the dispatcher's cache) right away.
@inline function steal_chunks(
        pool::Pool, slot, n::Int, skip::Int, base::Int, wslots, copies::Bool
    )
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        ws = @inbounds pool.workers[k]
        cmd = base | chunk
        if (@atomic :monotonic ws.cmd) == cmd &&
                (@atomicreplace :monotonic ws.cmd cmd => 0).success
            copies && clear_slot!(@inbounds wslots[k])
            @atomic :monotonic pool.stolen = (@atomic :monotonic pool.stolen) + 1
            # We took over the chunk, and with it the wake-ups of its children
            wake_children(pool, cmd)
            try
                run_chunk(slot, chunk)
            catch err
                @inbounds pool.errs[k] = CapturedException(err, catch_backtrace())
            end
        end
    end
    return nothing
end

# Take back the participants of a greedy region their worker has not claimed: when the
# dispatcher's own participant returns, the counter has handed out every chunk (or a chunk
# threw), so they would find nothing to do, and running them or waking their children
# would be wasted. The children are taken back too unless their worker claims first; a
# worker not woken stays asleep, so a burst of regions shorter than a wake-up recruits one
# level of the wake-up tree per region.
@inline function take_back_chunks(
        pool::Pool, n::Int, skip::Int, base::Int, wslots, copies::Bool
    )
    taken = cancel_chunks(pool, 2, n, skip, base, wslots, copies)
    @atomic :monotonic pool.stolen = (@atomic :monotonic pool.stolen) + taken
    return nothing
end

"""
    ThreadTeams.stolen_chunks() -> Int

Number of chunks the calling task ran (static regions) or took back (greedy regions)
because their worker had not picked them up in time, since the pool started. A high rate
per region means workers are often asleep or not running (an oversubscribed machine). For
diagnosis and benchmarks.
"""
function stolen_chunks()
    pool = @atomic POOL.pool
    return pool === nothing ? 0 : @atomic :monotonic pool.stolen
end

# Take the chunks `from:n` away from the workers that have not claimed them yet; returns
# how many. A load first, relaxed, and the slots cleared: see `steal_chunks`.
@inline function cancel_chunks(pool::Pool, from, n, skip, base, wslots, copies::Bool)
    taken = 0
    for chunk in from:n
        k = worker_for_chunk(chunk, skip)
        ws = @inbounds pool.workers[k]
        cmd = base | chunk
        if (@atomic :monotonic ws.cmd) == cmd &&
                (@atomicreplace :monotonic ws.cmd cmd => 0).success
            copies && clear_slot!(@inbounds wslots[k])
            taken += 1
        end
    end
    return taken
end

# Wait until the workers of chunks `from:n` are idle. A dead worker is reported only once
# the others have finished, since the region's slot and buffers are released afterwards.
@inline function wait_workers(pool::Pool, from, n, skip)
    dead = 0
    for chunk in from:n
        k = worker_for_chunk(chunk, skip)
        wait_idle(pool, k) || (dead = k)
    end
    dead == 0 || dead_worker_error(dead)
    return nothing
end

# Rethrow the original exception of a worker chunk; the worker's backtrace stays on the
# exception stack (`current_exceptions()`) as "caused by: ..."
@noinline function throw_worker_error(err::CapturedException)
    return try
        throw(err)
    catch
        throw(err.ex)
    end
end

# How a loop runs when it can not use the pool: serially when nested in a region (or
# during precompilation), on tasks when another task holds the pool
@inline function no_pool_mode()
    (generating_output() || in_region()) && return :serial
    return :tasks
end

# Run `f(chunk)` for `chunk in 1:n` on `Threads.@spawn`ed tasks (chunk 1 on the calling
# task), for loops started while another task runs a region. Exceptions as for a region:
# that of the lowest failing chunk is rethrown after all chunks finished. Not specialized
# on `f`: compiled once (precompile.jl) instead of once per loop body (about 20 ms each),
# for a dynamic call per chunk (well under a microsecond, next to the `@spawn`s).
@noinline function run_tasks(@nospecialize(f), n::Int)
    tasks = Vector{Task}(undef, n - 1)
    # Loops nested in the chunks run serially while the pool is busy (once it is released,
    # the first nested loop to arrive takes it)
    err1 = ScopedValues.with(IN_TASKS_REGION => true) do
        for chunk in 2:n
            tasks[chunk - 1] = Threads.@spawn :default f($chunk)
        end
        try
            f(1)
            nothing
        catch err
            CapturedException(err, catch_backtrace())
        end
    end
    for t in tasks
        Base._wait(t)
    end
    err1 === nothing || throw_worker_error(err1)
    for t in tasks
        istaskfailed(t) && throw_worker_error(CapturedException(t.result, t.backtrace))
    end
    return nothing
end

# The chunks of a greedy region: participant `p` claims chunk indices from the shared
# counter and runs them until all `m` are taken. Exceptions are stored with the chunk that
# threw, and a failing chunk stops the hand-out; chunks are claimed in increasing order,
# so every chunk before it runs, as in a serial loop.
struct GreedyJob{F}
    f::F
    counter::ChunkCounter
    m::Int
    errs::Vector{Any}
    errchunks::Vector{Int}
end
function (g::GreedyJob)(p::Int)
    chunk = 0
    try
        while true
            # Relaxed: the counter only claims indices; the job is published with the
            # commands, results with the workers' release stores
            chunk = @atomic :monotonic g.counter.next += 1
            chunk > g.m && break
            g.f(chunk)
        end
    catch err
        @inbounds g.errs[p] = CapturedException(err, catch_backtrace())
        @inbounds g.errchunks[p] = chunk
        @atomic :monotonic g.counter.next = typemax(Int) >> 1
    end
    return nothing
end

# Rethrow the exception of the lowest failing chunk of a greedy region, if any
@inline function throw_greedy_errors(errs::Vector{Any}, errchunks::Vector{Int}, p::Int)
    first = 0
    for k in 1:p
        if @inbounds(errs[k]) !== nothing &&
                (first == 0 || @inbounds(errchunks[k] < errchunks[first]))
            first = k
        end
    end
    first == 0 || throw_worker_error(@inbounds(errs[first])::CapturedException)
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n` on the pool, which the caller holds
@inline run_on_pool(pool::Pool, f::F, n::Int, ::TeamScheduler) where {F} = dispatch!(pool, f, n)
function run_on_pool(pool::Pool, f::F, m::Int, ::GreedyTeamScheduler) where {F}
    p = min(m, length(pool.workers))
    errs = pool.greedy_errs
    for k in 1:p
        @inbounds errs[k] = nothing
    end
    # Published to the workers by the (sequentially consistent) stores of the commands
    @atomic :monotonic pool.counter.next = 0
    try
        dispatch!(pool, GreedyJob(f, pool.counter, m, errs, pool.greedy_errchunks), p)
        throw_greedy_errors(errs, pool.greedy_errchunks, p)
    finally
        for k in 1:p   # do not keep the exceptions alive
            @inbounds errs[k] = nothing
        end
    end
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n` on spawned tasks (another task holds the pool)
run_off_pool(f::F, n::Int, ::TeamScheduler) where {F} = run_tasks(f, n)
function run_off_pool(f::F, m::Int, ::GreedyTeamScheduler) where {F}
    p = min(m, nworkers())
    errs = Any[nothing for _ in 1:p]
    errchunks = zeros(Int, p)
    run_tasks(GreedyJob(f, ChunkCounter(), m, errs, errchunks), p)
    throw_greedy_errors(errs, errchunks, p)
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n`: in parallel if the pool is free, serially if nested,
# on tasks if another task holds the pool. With the `TeamScheduler`, `n ≤ nworkers()`.
# Not inlined: its atomics must stay out of the caller (`pool_busy`). `run_chunks` and
# `run_fold` have Enzyme rules (ext/ThreadTeamsEnzymeCoreExt.jl): Enzyme can not
# differentiate the team itself (atomics, long-lived tasks), so the rules differentiate
# the chunk function and run the derivative chunks on the team.
@noinline function run_chunks(f::F, n::Int, sched::TeamSchedulers) where {F}
    pool = acquire_pool()
    if pool === nothing
        if no_pool_mode() === :tasks
            run_off_pool(f, n, sched)
        else
            for chunk in 1:n
                f(chunk)
            end
        end
        return nothing
    end
    try
        run_on_pool(pool, f, n, sched)
    finally
        release_pool()
    end
    return nothing
end

struct StoreChunk{F, T}
    f::F
    buf::Vector{T}
end
# No `@inbounds`: should a worker outlive its region (an interrupted caller), it must not
# write past the end of the emptied buffer
@inline (s::StoreChunk)(chunk) = (s.buf[chunk] = s.f(chunk); nothing)

@inline function fold_partials(op::Op, buf::Vector, n::Int) where {Op}
    acc = @inbounds buf[1]
    for chunk in 2:n
        acc = op(acc, @inbounds buf[chunk])
    end
    return acc
end

# `op(op(f(1), f(2)), ...)` over `1:n` with the `f(chunk)` evaluated in parallel (`n ≥ 2`),
# combined in chunk order, so deterministic for a given `n`. Not inlined: its atomics must
# stay out of the caller (`pool_busy`).
@noinline function run_fold(f::F, op::Op, n::Int, sched::TeamSchedulers) where {F, Op}
    T = Base.promote_op(f, Int)
    T === Union{} && (T = Any)
    pool = acquire_pool()
    if pool === nothing
        if no_pool_mode() === :tasks
            buf = Vector{T}(undef, n)
            run_off_pool(StoreChunk{F, T}(f, buf), n, sched)
            return fold_partials(op, buf, n)
        end
        acc = f(1)
        for chunk in 2:n
            acc = op(acc, f(chunk))
        end
        return acc
    end
    buf = partials_buffer(T, n)
    try
        run_on_pool(pool, StoreChunk{F, T}(f, buf), n, sched)
        return fold_partials(op, buf, n)
    finally
        empty!(buf)   # let the partial results be collected
        release_pool()
    end
end

"""
    ThreadTeams.shutdown!()

Stop all worker tasks and drop the pool. Waits for a parallel region running in another
task to finish first. The next parallel loop starts a new pool.

Throws an error when called from inside a parallel loop.
"""
function shutdown!()
    while !(@atomicreplace :acquire :monotonic POOL.busy false => true).success
        # The region can not end while we wait in one of its chunks
        if @atomic(POOL.owner) === current_task() || in_worker()
            error("ThreadTeams.shutdown! called from inside a parallel loop")
        end
        yield()
    end
    @atomic :monotonic POOL.owner = current_task()
    try
        pool = @atomic POOL.pool
        pool === nothing && return nothing
        @atomic POOL.pool = nothing
        for (k, ws) in enumerate(pool.workers)
            istaskdone(pool.tasks[k]) && continue   # a worker that died (see `wait_idle`)
            wait_idle(pool, k)
            @atomic ws.cmd = STOP
            notify(pool.events[k])
        end
        for t in pool.tasks
            istaskfailed(t) || wait(t)
        end
    finally
        release_pool()
    end
    return nothing
end

function in_worker()
    ct = current_task()
    return isdefined(ct, :code) && ct.code isa WorkerCode
end

"""
    ThreadTeams.parallel_phase(f)

Call `f()` and return its result. While `f` runs, idle workers keep spinning instead of
sleeping after [`spin_time()`](@ref ThreadTeams.spin_time), and sleeping workers are
woken at the start. Use it around a part of the program with many parallel loops
separated by serial work longer than the spin time, so that the loops do not pay for
waking the workers (about 10 µs each). Idle workers still sleep when their thread has
another task to run, but otherwise keep their CPU busy until `f` returns.

Phases may be nested and may be active in several tasks at once.

```julia
ThreadTeams.parallel_phase() do
    for step in 1:nsteps
        serial_part!(state)       # e.g. 1 ms
        tforeach(eachindex(u); scheduler = TeamScheduler()) do i
            ...
        end
    end
end
```
"""
function parallel_phase(f)
    Threads.atomic_add!(PHASE, 1)
    try
        if nworkers() > 1 && !generating_output()
            # Start the workers if needed, and wake up the sleeping ones
            pool = @atomic :acquire POOL.pool
            if pool === nothing &&
                    (@atomicreplace :acquire :monotonic POOL.busy false => true).success
                try
                    pool = @atomic :acquire POOL.pool
                    pool === nothing && (pool = new_pool())
                finally
                    @atomic :release POOL.busy = false
                end
            end
            if pool !== nothing
                for (k, flag) in enumerate(pool.sleepflags)
                    @atomic(flag.sleeping) && notify(pool.events[k])
                end
            end
        end
        return f()
    finally
        Threads.atomic_sub!(PHASE, 1)
    end
end
