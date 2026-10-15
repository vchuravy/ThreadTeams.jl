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
# 1. The chunk function `f::F` is stored in a per-type `Slot{F}` owned by the pool, which
#    is already heap-allocated, so the hand-off does not allocate (no boxing of `f`).
#    Workers receive a copy of its bytes in a buffer next to their command, which they
#    copy into a slot of their own (or the slot itself, if `f` is large; see
#    `PAYLOAD_BYTES`). They run it through a closure (one per type `F`, created in the
#    caller's world age, see `job_closure`) that is called with the worker's index and
#    buffer, and so finds the worker's own slot.
# 2. The caller publishes the region (its dynamic scope, if it has one), then gives
#    workers chunks `2:n` by storing the closure and the command (the chunk index and the
#    number of chunks) into their `WorkerState` (release stores, then one fence),
#    in decreasing chunk order. The worker running on the caller's thread (if any) is
#    skipped, because the caller occupies that thread.
# 3. Sleeping workers are woken up (with their `Event`) in a binary tree: whoever claims
#    chunk `c` first wakes the workers of chunks `2c` and `2c + 1` if they sleep, so the
#    caller wakes at most two workers itself.
# 4. Workers run their chunk in the caller's dynamic scope (`ScopedValue`s, the logger)
#    and world age.
# 5. The caller runs chunk 1 itself. Then it "steals" every chunk that its worker has not
#    claimed yet (a compare-and-swap of the command from `chunk` to `0`; the worker claims
#    it with a swap to `RUNNING`) and runs it itself (see `steal_chunks`). This keeps the
#    region from waiting for a worker that is asleep or whose thread the OS does not
#    schedule (oversubscribed cores). A greedy region only takes such chunks back (see
#    `take_back_chunks`). Finally it waits until every worker it used is idle again, also
#    when a chunk threw.
#
# Only one parallel region can be active at a time. The `busy` flag is taken with a
# compare-and-swap. A loop nested in a region (started by a chunk) runs serially; a loop
# started concurrently by another task runs on ordinary tasks (`Threads.@spawn`) instead.
#
# Memory orderings (tla/README.md has the model that checks them):
# - The region data (the closure, the scope, the bytes of `f`) is stored with relaxed
#   stores and published by the release store of each command; the worker's sequentially
#   consistent claim of the command (an acquire) makes it visible to it. Relaxed loads
#   suffice in the spin loop.
# - A worker's results (`errs`, the partials buffer) are published by its release store
#   `cmd = 0` and acquired by the load of `wait_idle`.
# - The sleep handshake (Dekker): the worker stores `sleeping = true` and then loads `cmd`,
#   both sequentially consistent; the dispatcher's command stores are followed by one
#   sequentially consistent fence and the loads of `wake_chunk` are sequentially
#   consistent. Either the worker sees its command, or the waker sees that it sleeps.
# - Everything else (the steal, the re-notify of `wait_idle`, `sleeping = false`) is
#   relaxed: a stale value only costs a spurious notification or a chunk run by the caller.
#
# With the `GreedyTeamScheduler`, the loop is split into more chunks than workers. The
# region then runs a `GreedyJob` on `p` participants (the caller and `p - 1` workers), each
# of which claims chunk indices from a shared counter until all chunks are taken.

const STOP = -1
const RUNNING = -2

# A command for a chunk carries the region data that the worker needs before it can run
# the chunk, so that the worker finds everything on the cache line of its command (the
# chunk function is stored next to it, see `WorkerState`): the chunk index, the number of
# chunks and the worker that is skipped (for `wake_children`), and whether the caller has a
# dynamic scope (then the worker loads it from the pool). Commands are positive; `0`,
# `STOP` and `RUNNING` are not commands.
const CMD_BITS = Sys.WORD_SIZE == 64 ? 20 : 9
const CMD_MASK = (1 << CMD_BITS) - 1
const SCOPE_BIT = 1 << (Sys.WORD_SIZE - 2)
const PAYLOAD_BIT = 1 << (Sys.WORD_SIZE - 3)
@inline cmd_base(n::Int, skip::Int, has_scope::Bool, payload::Bool) =
    (n << CMD_BITS) | (skip << (2 * CMD_BITS)) | (has_scope ? SCOPE_BIT : 0) |
    (payload ? PAYLOAD_BIT : 0)
@inline cmd_chunk(cmd::Int) = cmd & CMD_MASK
@inline cmd_nchunks(cmd::Int) = (cmd >> CMD_BITS) & CMD_MASK
@inline cmd_skip(cmd::Int) = (cmd >> (2 * CMD_BITS)) & CMD_MASK
@inline cmd_has_scope(cmd::Int) = (cmd & SCOPE_BIT) != 0
@inline cmd_has_payload(cmd::Int) = (cmd & PAYLOAD_BIT) != 0

"""
    TeamScheduler()

OhMyThreads.jl scheduler (also `scheduler = :team`) that runs a parallel operation on the
worker pool: the input is split into one contiguous chunk per thread of the `:default`
thread pool, chunk 1 runs on the calling task and chunk `c` on worker `c`. The calling task
also runs every chunk whose worker has not picked it up in time (a sleeping worker, or a
thread that the OS does not schedule), so a loop never waits for an idle worker. Lowest
overhead; best for loops with balanced iterations.

Only one parallel operation runs on the pool at a time. One started while another task is
using the pool runs on `Threads.@spawn`ed tasks instead, and one started from inside a
chunk (nested) runs serially. Chunks run in the dynamic scope (`ScopedValue`s, logger) and
world age of the caller. If chunks throw, the exception of the first failing chunk is
rethrown after all started chunks have finished.

The task index (`OhMyThreads.WithTaskIndex`, `@task_index`) is the chunk index. For
`chunks`/`index_chunks` inputs it is the position of the chunk, as with the other schedulers.
Task-local values (`@local`) live in the worker tasks, which persist: they are initialized
once per worker and reused across loops.

The scheduler takes no keyword arguments. In particular `ntasks` is not supported: chunks
are not guaranteed to run at the same time, so `@barrier` can not be used.
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
on the worker pool with load balancing: the input is split into `nchunks` contiguous chunks
(default `4 * nworkers()`), or into chunks of `chunksize` elements, and the calling task and
the workers take chunks from a shared counter until none are left. Use it for loops whose
iterations take very different amounts of time. See [`TeamScheduler`](@ref) for the
semantics shared by both schedulers.

Chunks are contiguous, and reductions combine the chunk results in chunk order, so the
reducing operator only has to be associative (not commutative), and the result is
deterministic for a given number of chunks. The task index is the chunk index.
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

# Padding around the atomic fields that the workers and the dispatcher hammer (a worker's
# command, its sleep flag, the greedy chunk counter), so that each lies on a cache line
# that no other field of the pool, and no object allocated next to it, shares. The field
# sits at `CACHE_LINE - 8` bytes from the start of the object and `CACHE_LINE` bytes of
# padding follow it, so its line lies within the object for any 16-byte aligned address.
# (A worker's command and its chunk function form a 16-byte pair at `CACHE_LINE - 16`.)
# Apple silicon has 128-byte lines; x86 and the other AArch64 cores have 64-byte lines.
# Padding to 128 bytes on every AArch64 CPU costs a few hundred bytes.
const CACHE_LINE = Sys.ARCH === :aarch64 ? 128 : 64
const PAD_BEFORE = CACHE_LINE ÷ sizeof(Int) - 1
const PAD_AFTER = CACHE_LINE ÷ sizeof(Int)
const PadBefore = NTuple{PAD_BEFORE, Int}
const PadAfter = NTuple{PAD_AFTER, Int}
pad_before() = ntuple(_ -> 0, Val(PAD_BEFORE))
pad_after() = ntuple(_ -> 0, Val(PAD_AFTER))

# Shared chunk counter of a `GreedyTeamScheduler` region, on its own cache line
mutable struct ChunkCounter
    _pad0::PadBefore
    @atomic next::Int
    _pad1::PadAfter
end
ChunkCounter() = ChunkCounter(pad_before(), 0, pad_after())

# The chunk function is handed to a worker as a copy of its bytes (the field of the
# dispatcher's slot, see `job_slot`) in a buffer of the worker, if it fits into
# `PAYLOAD_BYTES`; the worker copies them into a slot of its own (see `worker_slots`).
# Reading the dispatcher's slot instead, which the dispatcher writes in every region, cost
# every worker a cache miss after the one that brings it the command (about 40 ns of the
# hand-off with 4 threads). The buffer is written in every region as well, but the worker
# also loads its lines while it waits for a command, so that the stores of the buffer and
# of the command reach it together. The dispatcher's slot is then read by the dispatcher
# only, and stays in its cache.
const PAYLOAD_BYTES = 256
const PAYLOAD_LINES = PAYLOAD_BYTES ÷ CACHE_LINE
const PAYLOAD_WORDS = PAYLOAD_BYTES ÷ 8

# The buffer of a worker: `PAYLOAD_BYTES` aligned to the cache line, so that it shares no
# line with anything else (the object is 16-byte aligned; `CACHE_LINE` bytes of slack)
mutable struct PayloadBuffer
    words::NTuple{PAYLOAD_WORDS + CACHE_LINE ÷ 8, UInt64}
end
function payload_pointer(buf::PayloadBuffer)
    p = UInt(pointer_from_objref(buf))
    return Ptr{UInt64}((p + CACHE_LINE - 1) & ~UInt(CACHE_LINE - 1))
end

# A slot for a chunk function of type `F`: the dispatcher's (see `job_slot`) and the
# workers' (see `worker_slots`). It is written in every region, so it is padded to keep
# it off the cache lines of anything else (in particular of another worker's slot). The
# slot is empty (`isdefined(slot, :x)` is false) outside of regions, see `clear_slot!`.
mutable struct Slot{F}
    _pad0::PadAfter
    x::F
    _pad1::PadAfter
    Slot{F}() where {F} = new{F}()
end
Base.getindex(slot::Slot) = slot.x
Base.setindex!(slot::Slot{F}, f) where {F} = (slot.x = f; slot)
# The number of bytes of a field of type `F`: the value itself if it is stored inline,
# else a reference
@inline field_bytes(::Type{F}) where {F} = sizeof(Base.RefValue{F})
@inline slot_pointer(slot::Slot{F}) where {F} =
    pointer_from_objref(slot) + fieldoffset(Slot{F}, 2)
# The command of a worker and the closure that runs the chunk function of the region (see
# `job_closure`), on a cache line of their own: the 16-byte pair starts at a multiple of
# 16 bytes, so it never straddles two lines. The worker gets both, and with the command
# the rest of the region data (see `cmd_base`), with the one cache line transfer that
# brings it the command; the region data stored elsewhere (the scope) changes rarely.
mutable struct WorkerState
    _pad0::NTuple{PAD_BEFORE - 1, Int}
    @atomic cmd::Int
    @atomic job::Any
    _pad1::PadAfter
    # the buffer for the bytes of the chunk function (see `PAYLOAD_BYTES`) and its aligned
    # start; the object keeps the buffer alive
    const payload::PayloadBuffer
    const payload_ptr::Ptr{UInt64}
end
function WorkerState()
    buf = PayloadBuffer(ntuple(_ -> UInt64(0), Val(PAYLOAD_WORDS + CACHE_LINE ÷ 8)))
    return WorkerState(
        ntuple(_ -> 0, Val(PAD_BEFORE - 1)), 0, nothing, pad_after(), buf,
        payload_pointer(buf)
    )
end

# Whether a worker is asleep (or about to be), on a cache line of its own, apart from its
# command: it is written only when the worker goes to sleep or wakes up, so the loads of
# `wake_chunk` (two per chunk, on the critical path of every worker) hit a line that stays
# shared between all cores, instead of the line that the worker is claiming its chunk on.
mutable struct SleepFlag
    _pad0::PadBefore
    @atomic sleeping::Bool
    _pad1::PadAfter
end
SleepFlag() = SleepFlag(pad_before(), false, pad_after())

mutable struct Pool
    # The region data that does not travel with the commands (see `cmd_base`), written by
    # the dispatcher before it hands out chunks, and only when it changes, so that the
    # line stays shared by all cores: the dispatcher's dynamic scope (`nothing` outside of
    # regions and in regions whose caller has none). (The world age travels with the
    # closure, see `job_closure`.)
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
    # Number of chunks the dispatcher took over from their worker (see `steal_chunks`).
    # Written only by the dispatcher, which holds `busy` (relaxed, no read-modify-write).
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

Set how long an idle worker busy-waits for the next parallel loop before it goes to
sleep. The default is `100e-6` (100 µs). The time counts from the end of the parallel
loop: a worker that finished its chunk early keeps spinning while the loop waits for the
other chunks. Takes effect the next time a worker starts waiting.

Longer spinning makes parallel loops that follow each other with short serial gaps cheaper
(waking up a sleeping worker costs about 10 µs), but idle workers keep their CPU busy for
that long after every loop. `0` makes workers go to sleep as soon as they notice that the
pool is idle, which they check every 64 spins (about 100 ns): loops that follow each other
within that time keep them awake (lowest CPU use, highest latency). To keep the workers
spinning only during a part of the program, use [`ThreadTeams.parallel_phase`](@ref).

On CPUs with performance and efficiency cores (recent Intel CPUs, Apple silicon), spinning
cores take power and turbo headroom from the busy ones. LLVM's and GCC's OpenMP runtimes
do not spin by default on such CPUs; consider `set_spin_time!(0)` there.
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

# Does this thread have another task to run? Then an idle worker goes to sleep at once
# instead of spinning, so that the task gets the thread. (Uses scheduler internals; if they
# are missing, idle workers only go to sleep after the spin time.)
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

# The dynamic scope of the caller, and a macro to run code in it. Julia ≥ 1.11 keeps
# `ScopedValue`s and the logger in the task's scope; on Julia 1.10, ScopedValues.jl and
# `with_logger` keep them in the task's `logstate`. (The scope of a started task can not be
# set directly; the `tryfinally` form installs an exception handler, which `run_job_in_scope`
# avoids when the scope is already the current one.)
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

# Runs chunk `chunk` with the chunk function in `slot`. The dispatcher runs its chunks
# through this function, and so do the workers (in the body of the closure, see
# `job_closure`). It is not inlined: then every chunk runs the same machine code. Otherwise
# the chunk function is inlined into `dispatch!` (twice: for chunk 1 and for stolen chunks)
# and into the closure, and the dispatcher runs its own copy of the loop body, whose code
# layout and thus speed can differ from the workers' copy by a few percent (and which makes
# `dispatch!` larger and slower to compile).
@noinline function run_chunk(slot::Slot{F}, chunk::Int) where {F}
    slot.x(chunk)
    return nothing
end

# Call the closure of the current region, which runs the chunk in the world age of the
# dispatcher. A direct call of its compiled body for the usual closure type; a generic
# call (`jl_f_opaque_closure_call`, which works from the older world of the worker too)
# for a closure with another return type, see `new_job_closure`. (Julia 1.12 emits a
# generic call for both, see `JobClosure`.)
@inline function run_job(job, ws::Union{WorkerState, Nothing}, k::Int, chunk::Int)
    if job isa JobClosure
        job(ws, k, chunk)
    else
        job(ws, k, chunk)
    end
    return nothing
end

# Run the chunk of command `cmd` (claimed by worker `k`, whose state is `ws`) in the
# dispatcher's dynamic scope and world age. The closure is read here: passing the
# `Any`-typed closure and scope as arguments can make the call dynamic, which boxes its
# arguments.
function run_job_in_scope(pool::Pool, ws::WorkerState, k::Int, cmd::Int)
    job = @atomic :monotonic ws.job
    # The bytes of the chunk function are in the worker's buffer, or else the closure runs
    # the dispatcher's slot (see `run_job_closure`)
    wsp = cmd_has_payload(cmd) ? ws : nothing
    if !cmd_has_scope(cmd)
        # The common case (no `ScopedValue` or logger set by the caller, and the worker runs
        # without a scope): no scope to enter, and no exception handler for it
        run_job(job, wsp, k, cmd_chunk(cmd))
    else
        scope = @atomic :monotonic pool.scope
        @enter_scope scope run_job(job, wsp, k, cmd_chunk(cmd))
    end
    return nothing
end

# Copy `nbytes` bytes (rounded up to whole words: within both allocations, which are
# padded to 8 bytes at least) from `src` to `dst`
@inline function copy_words!(dst::Ptr, src::Ptr, nbytes::Int)
    d = Ptr{UInt64}(dst)
    s = Ptr{UInt64}(src)
    for i in 1:cld(nbytes, 8)
        unsafe_store!(d, unsafe_load(s, i), i)
    end
    return nothing
end

# Run chunk `chunk` with the chunk function whose bytes are at `src` (the buffer of the
# worker): copied into the worker's slot `slot` and dropped from it afterwards (when the
# chunk throws, by the dispatcher at the end of the region, see `dispatch!`). The references
# the bytes may hold are kept alive by the dispatcher's slot during the region. They are
# written into the slot without the write barrier of the GC: it does not need one, since
# the slot never holds anything that the dispatcher's slot (stored with the barrier) does
# not hold too.
@inline function run_job_copy(slot::Slot{F}, src::Ptr{UInt64}, chunk::Int) where {F}
    GC.@preserve slot copy_words!(slot_pointer(slot), src, field_bytes(F))
    run_chunk(slot, chunk)
    clear_slot!(slot)
    return nothing
end

# Wait until `ws.cmd` is non-zero: spin for `SPIN_NS` nanoseconds after the end of the
# parallel region (or for as long as a `parallel_phase` is active), then sleep on `ev`. Go
# to sleep right away when another task could run on this thread.
function wait_for_command(
        ws::WorkerState, flag::SleepFlag, ev::Base.Event, oversubscribed::Bool
    )
    spins = 0
    t0 = time_ns()
    tyield = tevents = t0
    payload = ws.payload_ptr
    while true
        # Fetch the lines of the payload buffer together with the command (see
        # `PAYLOAD_BYTES`). Only to warm the cache: the values are not used (and atomic
        # loads are not removed by the compiler).
        for i in 0:(PAYLOAD_LINES - 1)
            unsafe_load(payload + i * CACHE_LINE, :monotonic)
        end
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
                # Re-check after announcing that we sleep: either we see the command
                # here, or the dispatcher sees `sleeping` and notifies `ev`.
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
                # Spinning longer than usual (in a `parallel_phase`): run the libuv event
                # loop now and then (only does something on the thread that runs it, when
                # it is free), as the scheduler would do on an idle thread. Otherwise, with
                # a worker on every thread, timers and I/O would never complete.
                Base.process_events()
                tevents = now
            end
            if oversubscribed && now - tyield > 5_000
                # Let the OS run the other threads on this core now and then (only when
                # there are more threads than CPUs: on an idle machine this would only add
                # context switches)
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

# Runs commands until `STOP` (returns `true`) or until a chunk throws (returns `false`).
# One exception handler per stretch of commands, not one per chunk (an `enter` costs about
# as much as a small chunk). Not inlined: the frame that caught the exception must be gone
# before the worker waits for the next command, else on Julia 1.10 it keeps the exception
# (and what it refers to, such as the arrays of the loop body) alive.
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
        # (The dispatcher drops the copy of the chunk function, see `run_job_copy`)
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
    # A new task inherits the dynamic scope of the task that creates it. The workers are
    # created without one, so that they neither keep the scope of the first caller alive
    # nor miss the fast path of `run_job_in_scope` for callers without a scope.
    @enter_scope nothing for (k, tid) in enumerate(tids)
        task = Task(WorkerCode(pool, k))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1)
        push!(pool.tasks, task)
        schedule(task)
    end
    return pool
end

# Is a parallel region active (in this task or any other)?
#
# This check is inlined into every parallel loop in front of the serial fallback loop, so
# it must only use relaxed (`:monotonic`) loads. LLVM's MemoryDependenceAnalysis treats any
# acquire (or stronger) load as a clobber of all memory, even for `!invariant.load` queries
# ("FIXME: This is overly conservative" in MemoryDependenceAnalysis.cpp). An acquire load
# here stops GVN from merging loads after it with loads before it (e.g., array sizes
# checked by the caller), so the serial loop would lose constant strides and other facts
# (Trixi.jl: +4 to +8% time in several kernels on Julia 1.13). A stale value is harmless:
# it only makes us try the pool (and fall back to serial in `acquire_pool`) or run one loop
# serially. Data is only published with acquire/release in the non-inlined `run_chunks`,
# `run_fold` and `dispatch!`.
@inline pool_busy() = @atomic :monotonic POOL.busy

# Set while a task runs a chunk of a region that runs on spawned tasks (`run_tasks`), and
# inherited by tasks spawned from there. A `ScopedValue` of ScopedValues.jl, which is
# `Base.ScopedValues` on Julia ≥ 1.11.
const IN_TASKS_REGION = ScopedValues.ScopedValue(false)

# Is the current task running a chunk of a parallel region (as the dispatcher, as a worker,
# or as a task of `run_tasks`)? Only consulted while the pool is busy (see `pool_busy`);
# plain and relaxed loads, then the scope lookup.
@inline function in_region()
    ct = current_task()
    (@atomic :monotonic POOL.owner) === ct && return true
    (isdefined(ct, :code) && ct.code isa WorkerCode) && return true
    return IN_TASKS_REGION[]
end

# Take the pool for a parallel region (starting it if needed); `nothing` if the loop has to
# run serially. Only called from non-inlined functions.
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

# The slot of the dispatcher for the chunk function of type `F`: already allocated, so
# storing the function does not allocate (no boxing). The workers get a copy of its bytes
# (see `PAYLOAD_BYTES`), or the slot itself if the function is too large. One per type `F`,
# created when the generated function is first specialized for `F` and embedded in its
# body as a constant, so that getting it costs nothing (a dictionary lookup keyed by the
# type costs about 25 ns). Only used while holding `POOL.busy`, and cleared after every
# region.
@generated function job_slot(::Type{F}) where {F}
    return :($(Slot{F}()))
end

# The slots of the workers for chunk functions of type `F` (see `run_job_copy`): slot `k`
# is only used by worker `k`, so that it stays in the cache of the worker. Padded, so that
# the slots of two workers never share a cache line. Created, as `job_slot`, once per type;
# grown when a region first needs them.
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

# The closure that the workers call to run a chunk, as `job(ws, k, chunk)`: worker `k`
# copies the bytes of the chunk function from the buffer of its `WorkerState` `ws` into its
# own slot `worker_slot_vector(F)[k]` and runs the chunk from there (`run_job_copy`), or,
# if `ws` is `nothing` (a chunk function larger than `PAYLOAD_BYTES`), runs it from the
# dispatcher's slot `job_slot(F)` (`run_chunk`).
#
# An opaque closure runs its body in the world age it was created in (set on entry and
# restored on return, so dynamic calls in the chunk function dispatch in that world, as in
# a task the dispatcher would have spawned). That is what the workers need: their tasks are
# older than most chunk functions, and `invoke_in_world` (`Core._call_in_world`) costs a
# generic method lookup in a world other than the current one plus a boxed call on every
# chunk of every worker, about 20 ns on the critical path of each worker (Julia 1.10 to
# 1.13), against 3 ns for the closure call. The call of the closure in `run_job` is a
# generic one (`jl_apply_generic`, Julia 1.12) that boxes its arguments, so they are ones
# whose boxing does not allocate: an object (or `nothing`) and two small integers
# (`jl_box_int64` returns preallocated boxes for them).
#
# One closure per type `F`, kept in a per-type cache like `job_slot` and reused while the
# dispatcher's world age is the one it was created in. The first loop after a method
# definition (`include`, `eval`, Revise) creates a new one, about 120 ns when the body is
# compiled already. Only used while holding `POOL.busy`. The closure holds the slots of
# its type, which hold no chunk function outside of regions.
const JobClosure = Core.OpaqueClosure{Tuple{Any, Int, Int}, Nothing}

@generated function job_closure_cache(::Type{F}) where {F}
    return :($(Base.RefValue{Any}(nothing)))
end

@inline function job_closure(slot::Slot{F}) where {F}
    cache = job_closure_cache(F)
    job = cache[]
    world = world_age()
    if !(job isa JobClosure && job.world == world)
        # A dynamic call (in the same world), so that the construction of the closure is
        # not part of the code of `dispatch!`: Enzyme compiles the primal code of the
        # loops that its rules (the EnzymeCore extension) run on the team into its own
        # module, and its transforms crash on the closure construction (Julia 1.10).
        job = Core._call_in_world(world, new_job_closure, slot, worker_slot_vector(F))
        cache[] = job
    end
    return job
end

# The body of the closure (see `job_closure`)
@inline function run_job_closure(
        slot::Slot{F}, wslots::Vector{Slot{F}}, ws, k::Int, chunk::Int
    ) where {F}
    if ws === nothing
        run_chunk(slot, chunk)
    else
        run_job_copy(@inbounds(wslots[k]), (ws::WorkerState).payload_ptr, chunk)
    end
    return nothing
end

# The closure is built by the runtime (`jl_new_opaque_closure`): its argument types are
# hidden from inference, so that codegen does not build it. The runtime gives it the
# compiled code of its body as entry point, compiling the body on the first loop with a
# chunk function type (as the first call of a worker did before). Built by codegen, the
# closure's body is emitted into the module of `dispatch!`, which Enzyme's transforms do
# not handle (segmentation fault, Julia 1.10), and on Julia 1.12 and 1.13 the closure only
# gets a stub that boxes the arguments and goes through `jl_invoke_oc` (about 13 ns and
# 48 bytes per call). The return type can be given on Julia ≥ 1.12 only; on 1.10 and 1.11
# it is the inferred one, `Union{}` for a chunk function that always throws (the closure is
# then not a `JobClosure`, see `run_job`).
@static if VERSION >= v"1.12"
    @noinline function new_job_closure(slot::Slot{F}, wslots::Vector{Slot{F}}) where {F}
        return Base.Experimental.@opaque Base.inferencebarrier(Tuple{Any, Int, Int}) -> Nothing (ws, k, chunk) -> run_job_closure(slot, wslots, ws, k, chunk)
    end
else
    @noinline function new_job_closure(slot::Slot{F}, wslots::Vector{Slot{F}}) where {F}
        return Base.Experimental.@opaque Base.inferencebarrier(Tuple{Any, Int, Int}) (ws, k, chunk) -> run_job_closure(slot, wslots, ws, k, chunk)
    end
end

# Drop the chunk function from its slot after the region, so that it (and the arrays it
# captures) can be collected. `setfield!` can not make the field undefined again, and
# storing a struct with null pointer fields would trip the GC write barrier, so the bytes
# of the field are zeroed directly. That is the state `Slot{F}()` creates: the field may be
# undefined (`Slot` has a `new()` constructor), and for an inline immutable with pointer
# fields Julia represents "undefined" by a null first pointer, which is what the zeroed
# bytes are. Afterwards `isdefined(slot, :x)` is false and `slot[]` throws an
# `UndefRefError`; for an isbits `F` the zero bytes are a value, for a mutable `F` a null
# (undefined) reference. The GC skips null fields, and removing references needs no write
# barrier. Nothing reads the slot outside a region. (The field is followed by padding, so
# rounding up to whole words stays within the object.)
@inline function clear_slot!(slot::Slot{F}) where {F}
    GC.@preserve slot zero_words!(slot_pointer(slot), field_bytes(F))
    return nothing
end
@inline function zero_words!(dst::Ptr, nbytes::Int)
    d = Ptr{UInt64}(dst)
    for i in 1:cld(nbytes, 8)
        unsafe_store!(d, UInt64(0), i)
    end
    return nothing
end

# The buffer for the partial results of a reduction. It is emptied after the region
# (`empty!` nulls the elements of a buffer of boxed values but keeps the capacity), so
# resizing it here allocates only when a region needs more chunks than any before.
@generated function static_buffer(::Type{T}) where {T}
    return :($(Vector{T}()))
end
@inline function partials_buffer(::Type{T}, n::Int) where {T}
    buf = static_buffer(T)
    length(buf) < n && resize!(buf, n)
    return buf
end

# Spin until worker `k` is idle. Yields now and then, so that a worker (or a chunk that
# yields) sharing the thread of the waiting task can make progress. Returns `false` if the
# worker died with its command still set (see `dead_worker_error`); the caller then waits
# for the other workers before it throws, so that no worker is left running a chunk.
@inline function wait_idle(pool::Pool, k::Int)
    ws = @inbounds pool.workers[k]
    spins = 0
    while (@atomic :acquire ws.cmd) != 0
        spins += 1
        if spins & 1023 == 0
            # Defensive: wake the worker if it went to sleep with a pending command (a
            # relaxed load: this is the fallback of the handshake in `wake_chunk`)
            @atomic(:monotonic, (@inbounds pool.sleepflags[k]).sleeping) &&
                notify(@inbounds pool.events[k])
            # A worker task can only die through an exception thrown into it from outside
            # (e.g. an interrupt); with its command still set, the region would wait forever
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

# Called by whoever claims chunk `chunk` of `n`, before running it: wake the workers of
# chunks `2chunk` and `2chunk + 1` (a binary tree, so that waking `n` sleeping workers
# takes O(log n) steps). The dispatcher stores the commands in decreasing chunk order, so a
# child's command is stored before its parent's. For the dispatcher's own wake-ups, the
# fence after its stores and the sequentially consistent loads here guarantee that either
# the child sees its command before it sleeps, or we see that it sleeps. For a worker, the
# full barrier of its sequentially consistent claim does the same on x86 and AArch64;
# the C11 model alone does not promise it (see tla/README.md). A wake-up that is lost
# this way costs no correctness: the dispatcher runs the chunk itself (stealing), and
# `wait_idle` re-notifies sleeping workers with pending commands.
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
    # Published to the workers by the release stores of the commands. The scope is stored
    # only when it is not the usual one: the line stays shared.
    scope = current_scope()
    has_scope = scope !== nothing
    has_scope && (@atomic :monotonic pool.scope = scope)
    # Hand the chunk function to the workers as bytes if it fits (see `PAYLOAD_BYTES`)
    nbytes = field_bytes(F)
    payload = cld(nbytes, 8) * 8 <= PAYLOAD_BYTES
    wslots = worker_slots(F, length(pool.workers))
    base = cmd_base(n, skip, has_scope, payload)
    try
        # Decreasing chunk order: see `wake_children`
        for chunk in n:-1:2
            k = worker_for_chunk(chunk, skip)
            # A worker can still be busy if a previous region was interrupted
            if !wait_idle(pool, k)
                # Take back the chunks already handed out before giving up
                cancel_chunks(pool, chunk + 1, n, skip, base)
                wait_workers(pool, chunk + 1, n, skip)
                dead_worker_error(k)
            end
            @inbounds pool.errs[k] = nothing
            # Release stores: their cache-line transfers overlap in the store buffer, where
            # sequentially consistent stores (`xchg` on x86) would each wait for the line (about
            # 20% of the dispatcher's time with 8 threads). The fence below orders them before
            # the loads of `sleeping` in `wake_children` (Dekker-style with the worker, which
            # stores `sleeping` and then loads `cmd`). Should a wake-up still be missed on a
            # weakly ordered machine, `wait_idle` re-notifies sleeping workers with pending
            # commands, so the region only takes longer.
            ws = @inbounds pool.workers[k]
            payload && GC.@preserve slot copy_words!(ws.payload_ptr, slot_pointer(slot), nbytes)
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
            cancel_chunks(pool, 2, n, skip, base)
            wait_workers(pool, 2, n, skip)
            rethrow()
        end
        # The chunks that their worker has not claimed yet: a greedy region only takes them
        # back, a static one runs them
        if f isa GreedyJob
            take_back_chunks(pool, n, skip, base)
        else
            steal_chunks(pool, slot, n, skip, base)
        end
        wait_workers(pool, 2, n, skip)
        for chunk in 2:n
            k = worker_for_chunk(chunk, skip)
            err = @inbounds pool.errs[k]
            err === nothing || throw_worker_error(err::CapturedException)
        end
    finally
        # Keep neither the chunk function (see `clear_slot!`), nor the scope, nor the
        # exceptions of the chunks alive (every used worker is idle again here, also when
        # a chunk threw)
        clear_slot!(slot)
        has_scope && (@atomic :monotonic pool.scope = nothing)
        for chunk in 2:n
            k = worker_for_chunk(chunk, skip)
            # A worker whose chunk threw left its copy of the chunk function in its slot
            # (see `run_job_copy`)
            if (@inbounds pool.errs[k]) !== nothing
                payload && clear_slot!(@inbounds wslots[k])
                @inbounds pool.errs[k] = nothing
            end
        end
    end
    return nothing
end

# Run the chunks of a static region that their worker has not claimed yet. A worker that
# spins claims its chunk within a cache-line transfer, so one that has not claimed it by
# now is asleep or not running (the OS gave its CPU to another thread): waiting for it
# would cost a wake-up or a time slice of the OS. Cache locality: a stolen chunk is shorter
# than that wake-up (the dispatcher ran its own chunk first), so it moves at most the data
# that a core touches in that time, and fetching it back costs its worker about one chunk.
#
# A load first: a compare-and-swap on a line that the worker has just written takes it from
# the worker's cache (and the worker's final store `cmd = 0` takes it back); a load only
# shares it. Relaxed: nothing of the worker's is read after a successful steal, and
# `wait_workers` acquires the results of the workers that did claim their chunk.
@inline function steal_chunks(pool::Pool, slot, n::Int, skip::Int, base::Int)
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        ws = @inbounds pool.workers[k]
        cmd = base | chunk
        if (@atomic :monotonic ws.cmd) == cmd &&
                (@atomicreplace :monotonic ws.cmd cmd => 0).success
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

# Take back the participants of a greedy region that their worker has not claimed. When the
# dispatcher's own participant returns, the counter has handed out every chunk (or a chunk
# threw), so these participants would find no chunk left: running them, or waking the
# workers of their children, would be wasted work. (The children are taken back as well,
# unless their worker claims first. A worker that is not woken stays asleep, so a burst of
# regions shorter than a wake-up recruits one level of the wake-up tree per region.)
@inline function take_back_chunks(pool::Pool, n::Int, skip::Int, base::Int)
    taken = cancel_chunks(pool, 2, n, skip, base)
    @atomic :monotonic pool.stolen = (@atomic :monotonic pool.stolen) + taken
    return nothing
end

"""
    ThreadTeams.stolen_chunks() -> Int

Number of chunks that the calling task ran (static regions) or took back (greedy regions)
because their worker had not picked them up in time, since the pool was started. A high
rate compared to the number of regions means that workers are often asleep or not running
(an oversubscribed machine); for diagnosis and benchmarks.
"""
function stolen_chunks()
    pool = @atomic POOL.pool
    return pool === nothing ? 0 : @atomic :monotonic pool.stolen
end

# Take the chunks `from:n` away from the workers that have not claimed them yet; returns
# how many. A load first, relaxed: see `steal_chunks`.
@inline function cancel_chunks(pool::Pool, from, n, skip, base)
    taken = 0
    for chunk in from:n
        ws = @inbounds pool.workers[worker_for_chunk(chunk, skip)]
        cmd = base | chunk
        if (@atomic :monotonic ws.cmd) == cmd &&
                (@atomicreplace :monotonic ws.cmd cmd => 0).success
            taken += 1
        end
    end
    return taken
end

# Wait until the workers of chunks `from:n` are idle. A dead worker is reported only after
# the others have finished: the region's slot and buffers are released afterwards.
@inline function wait_workers(pool::Pool, from, n, skip)
    dead = 0
    for chunk in from:n
        k = worker_for_chunk(chunk, skip)
        wait_idle(pool, k) || (dead = k)
    end
    dead == 0 || dead_worker_error(dead)
    return nothing
end

# Rethrow the original exception of a worker chunk. The worker's backtrace is kept on the
# exception stack (`current_exceptions()`), so it is shown as "caused by: ...".
@noinline function throw_worker_error(err::CapturedException)
    return try
        throw(err)
    catch
        throw(err.ex)
    end
end

# `nothing`, or `:serial` / `:tasks` if the loop can not use the pool: serially when
# nested in a region (or during precompilation), on tasks when another task holds the pool
@inline function no_pool_mode()
    (generating_output() || in_region()) && return :serial
    return :tasks
end

# Run `f(chunk)` for `chunk in 1:n` on `Threads.@spawn`ed tasks (chunk 1 on the calling
# task), for loops started while another task runs a parallel region. Exceptions as for a
# region: the one of the lowest failing chunk is rethrown after all chunks finished.
# Not specialized on `f`: this path is compiled once (see precompile.jl) instead of once
# per loop body (about 20 ms each), at the cost of a dynamic call per chunk (well under a
# microsecond, next to the `@spawn`s).
@noinline function run_tasks(@nospecialize(f), n::Int)
    tasks = Vector{Task}(undef, n - 1)
    # Loops nested in the chunks run serially while the pool is busy (once the other task
    # has released it, the first nested loop to arrive takes the pool)
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

# Runs the chunks of a greedy region: participant `p` claims chunk indices from the
# shared counter and runs them until all `m` chunks are taken. Exceptions are caught and
# stored with the chunk that threw. A failing chunk stops the hand-out of further chunks;
# since chunks are claimed in increasing order, every chunk before it was claimed, so it
# runs (as in a serial loop).
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
            # `:monotonic` suffices: the counter only claims indices; the job is published
            # with the commands, and results with the workers' release stores
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

# Run `f(chunk)` for `chunk in 1:n`, in parallel if the pool is free, else serially (nested)
# or on tasks (another task holds the pool). With the `TeamScheduler`, `n` must not
# exceed `nworkers()`.
# Not inlined: the atomics in here must stay out of the caller (see `pool_busy`).
# `run_chunks` and `run_fold` have Enzyme rules (ext/ThreadTeamsEnzymeCoreExt.jl): Enzyme can
# not differentiate the team itself (atomics, long-lived tasks), so the rules differentiate
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

# Compute `op(op(f(1), f(2)), ...)` over `1:n` with the `f(chunk)` evaluated in parallel
# (`n ≥ 2`). The partial results are combined in chunk order, so the result is
# deterministic for a given `n`.
# Not inlined: the atomics in here must stay out of the caller (see `pool_busy`).
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
going to sleep after [`spin_time()`](@ref ThreadTeams.spin_time), and sleeping workers
are woken up at the start. Use it around a part of the program with many parallel loops
separated by serial work that takes longer than the spin time, so that the loops do not
pay for waking up the workers (about 10 µs each). Idle workers still go to sleep when
their thread has another task to run, but otherwise keep their CPU busy until `f`
returns.

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
