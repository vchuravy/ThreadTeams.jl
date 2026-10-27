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
#    worker that is asleep or not scheduled by the OS. Finally it waits until every worker
#    it used is idle, also when a chunk threw.
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
# `GreedyJob` on `p` participants (the caller and `p - 1` workers), each running the chunk
# with its own index, then claiming the chunks after `p` from a shared counter until none
# are left.
#
# `foreach_range` (src/ranges.jl) also dispatches one chunk per participant, with jobs that
# place the blocks of its range by worker rather than by chunk.

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
    TeamScheduler(; busy = :tasks)

OhMyThreads.jl scheduler (also `scheduler = :team`) that runs a parallel operation on the
worker pool: one contiguous chunk per thread of the `:default` pool, chunk 1 on the
calling task and chunk `c` on worker `c`. The calling task also runs every chunk whose
worker has not picked it up in time (asleep, or not scheduled by the OS), so a loop never
waits for an idle worker. Lowest overhead; best for balanced iterations.

Only one parallel operation runs on the pool at a time. One started inside a chunk
(nested) runs serially. One started while another task uses the pool runs on
`Threads.@spawn`ed tasks with `busy = :tasks` (the default), and serially on the calling
task with `busy = :serial`, which also runs the operation serially while a
`Threads.@threads` loop runs (in any task: the check is process-wide). Use `:serial` in
code that is itself run in parallel by tasks or `@threads`: spreading its loops over
threads that are busy already only oversubscribes them. Chunks run in the caller's dynamic
scope (`ScopedValue`s, logger) and world age. If chunks throw, the exception of the first
failing chunk is rethrown once all started chunks have finished.

The task index (`OhMyThreads.WithTaskIndex`, `@task_index`) is the chunk index; for
`chunks`/`index_chunks` inputs the position of the chunk, as with the other schedulers.
Task-local values (`@local`) live in the persistent worker tasks: initialized once per
worker, reused across loops.

`ntasks` is not supported: chunks are not guaranteed to run at the same time, so
`@barrier` can not be used.
"""
struct TeamScheduler <: Scheduler
    serial_when_busy::Bool
    function TeamScheduler(; busy::Symbol = :tasks, kwargs...)
        isempty(kwargs) || unsupported_kwargs("TeamScheduler", kwargs)
        return new(serial_when_busy(busy))
    end
end

"""
    GreedyTeamScheduler(; nchunks = 0, chunksize = 0, busy = :tasks)

OhMyThreads.jl scheduler (also `scheduler = :greedyteam`) that runs a parallel operation
on the worker pool with load balancing: `nchunks` contiguous chunks (default
`4 * nworkers()`), or chunks of `chunksize` elements. The calling task and the workers
start with one chunk each, placed as with [`TeamScheduler`](@ref), then take the others
from a shared counter until none are left. For loops whose iterations take very
different times. See [`TeamScheduler`](@ref) for the semantics both share.

Chunks are contiguous and reductions combine the chunk results in chunk order, so the
operator only has to be associative and the result is deterministic for a given number of
chunks. The task index is the chunk index.
"""
struct GreedyTeamScheduler <: Scheduler
    nchunks::Int
    chunksize::Int
    serial_when_busy::Bool
    function GreedyTeamScheduler(;
            nchunks::Integer = 0, chunksize::Integer = 0, busy::Symbol = :tasks,
            kwargs...
        )
        isempty(kwargs) || unsupported_kwargs("GreedyTeamScheduler", kwargs)
        (nchunks >= 0 && chunksize >= 0) ||
            throw(ArgumentError("nchunks and chunksize must be non-negative"))
        (nchunks > 0 && chunksize > 0) &&
            throw(ArgumentError("nchunks and chunksize are mutually exclusive"))
        return new(nchunks, chunksize, serial_when_busy(busy))
    end
end

const TeamSchedulers = Union{TeamScheduler, GreedyTeamScheduler}

function serial_when_busy(busy::Symbol)
    busy === :tasks && return false
    busy === :serial && return true
    throw(ArgumentError("busy must be :tasks or :serial, not $(repr(busy))"))
end

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
# field or neighbouring object shares: the field sits one word short of `CACHE_LINE` bytes
# into the object with `CACHE_LINE` bytes after it, so its line lies within the object at
# any 16-byte aligned address. (A worker's command and closure form a two-word pair two
# words short of `CACHE_LINE`.) Apple silicon has 128-byte lines, x86 and other AArch64
# cores 64; padding to 128 on every AArch64 CPU costs a few hundred bytes.
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

# Cache-line-separated packed range (see `pack_range`). UInt64 also on 32-bit systems:
# Julia aligns atomic fields to their size. i686 uses SSE2 loads/stores and lock cmpxchg8b
# (checked on Julia 1.10/1.12); ARMv7 should use ldrexd/strexd (not verified).
mutable struct RangeBlock
    const _pad0::PadBefore
    @atomic word::UInt64
    const _pad1::PadAfter
end
RangeBlock() = RangeBlock(pad_before(), UInt64(0), pad_after())

# Per-pool range state: error/stop share a read-only line until failure; `active` has
# its own line, read by idle workers and set/cleared by the dispatcher. Steal blocks
# and rescue timestamps are reused between regions.
mutable struct RangeState
    const _pad0::NTuple{PAD_BEFORE - 1, Int}
    @atomic err::Any
    @atomic stop::Bool
    const _pad1::PadAfter
    @atomic active::Bool
    const _pad2::PadAfter
    const blocks::Vector{RangeBlock}
    # Dispatcher-only first-awake timestamps; zero means not observed awake.
    const seen::Vector{UInt64}
end
RangeState(nw::Int) = RangeState(
    ntuple(_ -> 0, Val(PAD_BEFORE - 1)), nothing, false, pad_after(), false, pad_after(),
    [RangeBlock() for _ in 1:nw], zeros(UInt64, nw)
)

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

# Who empties a worker's slot after the region (`worker_slot_vector`). If the worker does,
# the line is modified in its cache, and the dispatcher's next store into the slot has to
# fetch it back (a read-for-ownership that hits a modified line, waited for at the fence
# of `dispatch!`). If the dispatcher does (while waiting for the workers, `wait_workers`),
# the worker's copy stays clean and the next store only invalidates it, but the
# dispatcher spends a store per worker. Which is faster depends on the CPU: on Intel the
# dispatcher's (#46: Broadwell-EP 2-3 fewer RFO HITM per loop with 4-6 threads, Cascade
# Lake about 15% faster with 4 threads); on Zen 3, the NVIDIA GB10 (Cortex-X925) and the
# Apple M1 the worker's (5-20% faster with 4-16 threads; on Zen 3 a read of a modified
# line moves it to the reader, so the dispatcher's store misses either way). Set once per
# process from `dispatcher_clears_slots()` when the pool starts, before its workers.
const DISPATCHER_CLEARS = Ref(false)

"""
    ThreadTeams.dispatcher_clears_slots(cpu = Sys.CPU_NAME, arch = Sys.ARCH,
        setting = get(ENV, "THREADTEAMS_SLOT_CLEARING", "")) -> Bool

Whether the dispatcher, rather than each worker, empties the workers' chunk-function slots
after a region: `true` on Intel x86 CPUs, `false` on AMD, ARM and unknown CPUs. The
environment variable `THREADTEAMS_SLOT_CLEARING` (`dispatcher` or `worker`) overrides it.
Read once per process, when the pool starts.
"""
function dispatcher_clears_slots(
        cpu::AbstractString = Sys.CPU_NAME, arch::Symbol = Sys.ARCH,
        setting::AbstractString = get(ENV, "THREADTEAMS_SLOT_CLEARING", "")
    )
    setting == "dispatcher" && return true
    setting == "worker" && return false
    isempty(setting) || @warn "THREADTEAMS_SLOT_CLEARING should be `dispatcher` or `worker`, not `$setting`; using the default"
    (arch === :x86_64 || arch === :i686) || return false
    # LLVM's names of AMD's x86 CPUs, and generic targets (unknown vendor)
    amd = ("znver", "bdver", "btver", "amdfam", "barcelona", "k8", "opteron", "athlon")
    return !(any(p -> startswith(cpu, p), amd) || cpu == "generic" || startswith(cpu, "x86-64"))
end

# Computed once per process (`Base.OncePerProcess`, Julia >= 1.12); before, when a pool
# starts, which happens once per process too unless the pool is shut down
@static if isdefined(Base, :OncePerProcess)
    const SLOT_CLEARING = Base.OncePerProcess{Bool}(dispatcher_clears_slots)
else
    const SLOT_CLEARING = dispatcher_clears_slots
end

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
# A worker's command and the closure of the region (`job_closure`), a two-word pair at a
# multiple of two words on a cache line of their own: one line transfer brings the worker
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
    # `errs[k]`: `ChunkFailedException` of the chunk worker `k` ran, or `nothing`
    const errs::Vector{Any}
    # Set when a chunk of the region stored its exception into `errs` (by the worker
    # before its release store of `cmd = 0`, by the dispatcher for a chunk it stole) or,
    # in a greedy region, into `greedy_errs` (`GreedyJob`), so that after a region the
    # dispatcher checks one line instead of every worker's `errs`; reset with `errs` by the
    # dispatcher. `errs` is therefore all `nothing` when a region starts. On a cache line
    # of its own (see `PadBefore`).
    const _pad0::PadBefore
    @atomic failed::Bool
    const _pad1::PadAfter
    # `tid2worker[tid]`: index of the worker running on thread `tid`, or `0`
    const tid2worker::Vector{Int}
    const sleepflags::Vector{SleepFlag}
    # Greedy regions: the chunk counter, and per participant the exception it caught and
    # the chunk that threw it
    const counter::ChunkCounter
    const greedy_errs::Vector{Any}
    const greedy_errchunks::Vector{Int}
    # Julia threads have to share CPUs (`is_oversubscribed`): spinning workers then give
    # their CPU to other OS threads now and then
    const oversubscribed::Bool
    # Chunks the dispatcher took over from their worker (`steal_chunks`); written only by
    # the dispatcher, which holds `busy` (relaxed, no read-modify-write)
    @atomic stolen::Int
    # Reusable state for `foreach_range`.
    const ranges::RangeState
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
# How long an idle worker spins before it runs the event loop (every 20 µs): independent
# of the spin time, so that long spin times do not hold up timers and I/O (#35)
const EVENTS_NS = 100_000
# Number of active `parallel_phase`s; while > 0, idle workers do not go to sleep on time
const PHASE = Threads.Atomic{Int}(0)

"""
    ThreadTeams.set_spin_time!(seconds::Real)

Set how long an idle worker busy-waits for the next parallel loop before it sleeps;
default `100e-6` (100 µs). The time counts from the end of the loop: a worker that
finished early keeps spinning while the loop waits for the other chunks. A worker that the
loops do not use counts from the end of its own last chunk, also while other loops run.
Takes effect the next time a worker starts waiting.

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

# A task sticky to this thread (`local_work`) needs this worker to sleep. A shared task
# (`queued_work`) needs only one worker to run it (`run_queued_task`); the others keep
# spinning, avoiding wake-ups in the next region. Without these scheduler internals,
# workers sleep only after the spin time.
const HAS_WORKQUEUE = isdefined(Base, :workqueue_for)
const HAS_MULTIQ = isdefined(Base, :Partr) && isdefined(Base.Partr, :multiq_check_empty) &&
    isdefined(Base.Partr, :multiq_deletemin)

@inline local_work() =
    @static HAS_WORKQUEUE ? !isempty(Base.workqueue_for(Threads.threadid())) : false

@inline queued_work() = @static HAS_MULTIQ ? !Base.Partr.multiq_check_empty() : false

# Serialize queue pops so workers do not all race for the same task.
mutable struct TaskTaker
    const _pad0::PadBefore
    @atomic taking::Bool
    const _pad1::PadAfter
end
const TAKER = TaskTaker(pad_before(), false, pad_after())

# Run a queued task on this thread until it yields, waits or ends; meanwhile, the worker
# waits in this thread's sticky queue. Return whether a task ran. Like `trypoptask`, skip
# tasks that are no longer runnable (abandoned tasks, Julia ≥ 1.14).
@noinline function run_queued_task()
    @static if HAS_MULTIQ
        (@atomicreplace :acquire :monotonic TAKER.taking false => true).success || return false
        t = try
            pop_runnable_task()
        finally
            @atomic :release TAKER.taking = false
        end
        t === nothing && return false
        # Pass the value supplied by `schedule(t, val)`, e.g. by a timer.
        yield(t, t.result)
        return true
    else
        return false
    end
end

function pop_runnable_task()
    while true
        t = Base.Partr.multiq_deletemin()
        t === nothing && return nothing
        t = t::Task
        t._state === Base.task_state_runnable && return t
        @static if isdefined(Base, :discard_stale_workqueue_task)
            Base.discard_stale_workqueue_task(t)
        end
    end
    return
end

# More Julia threads than CPUs they may run on, e.g. `-t N` on N cores with Julia ≥ 1.12
# (which adds an interactive thread), `taskset`, a cgroup CPU quota, or threads pinned
# (e.g. with ThreadPinning.jl) so that several of them share a CPU. Checked when the pool is
# created (`ThreadTeams.shutdown!()` makes the next loop check again).
function is_oversubscribed()
    tids = vcat(Threads.threadpooltids(:interactive), Threads.threadpooltids(:default))
    nt = length(tids)
    # CPUs of the calling thread's affinity mask, limited by the cgroup CPU quota
    navail = Int(ccall(:uv_available_parallelism, Cuint, ()))
    masks = thread_affinity_masks(tids)
    masks === nothing && return nt > navail
    oversubscribed_masks(masks) && return true
    # The quota is only known through `uv_available_parallelism`, which also counts the
    # affinity mask of the calling thread; use it only when that thread may run on all the
    # CPUs of the other threads (else a pinned caller would count as oversubscription)
    k = findfirst(==(Threads.threadid()), tids)
    return k !== nothing && masks[k] == reduce(.|, masks) && nt > navail
end

# Affinity mask of each thread in `tids` as a `BitVector` (Julia ≥ 1.11, where the runtime
# can query the affinity of another thread), or `nothing` if it is not available (Julia
# 1.10, or an OS where libuv does not support thread affinity, e.g. macOS)
function thread_affinity_masks(tids)
    @static if VERSION >= v"1.11-"
        sz = Int(ccall(:uv_cpumask_size, Cint, ()))
        sz > 0 || return nothing
        buf = zeros(Cchar, sz)
        masks = BitVector[]
        for tid in tids
            ret = ccall(:jl_getaffinity, Int32, (Int16, Ptr{Cchar}, Int32), tid - 1, buf, sz)
            ret == 0 || return nothing
            push!(masks, BitVector(!iszero(c) for c in buf))
        end
        return masks
    else
        return nothing
    end
end

# Do these threads (one affinity mask each) have to share CPUs? Hall's condition for a
# thread-to-CPU matching, checked only for the union of the masks and for each thread's own
# mask: e.g. two threads pinned to the same CPU, or more threads than CPUs in a common mask.
# Exact for the layouts pinning produces (one CPU per thread, a shared mask, nested or
# disjoint masks); masks that overlap otherwise can hide a shortage, e.g. 4 threads on
# `{0,1}, {0,1}, {1,2}, {0,2}`.
function oversubscribed_masks(masks::AbstractVector{<:AbstractVector{Bool}})
    isempty(masks) && return false
    count(reduce(.|, masks)) < length(masks) && return true
    for m in masks
        nsub = count(s -> all(s .<= m), masks)
        nsub > count(m) && return true
    end
    return false
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

# Run chunk `chunk` from the worker's own slot, and empty the slot afterwards unless the
# dispatcher does (`DISPATCHER_CLEARS`; when the chunk throws, the dispatcher does in any
# case, see `dispatch!`)
@inline function run_job_own(slot::Slot{F}, chunk::Int) where {F}
    run_chunk(slot, chunk)
    DISPATCHER_CLEARS[] || clear_slot!(slot)
    return nothing
end

# Wait until `ws.cmd` is non-zero: spin for `SPIN_NS` after the end of the region (or while
# a `parallel_phase` is active), then sleep on `ev`; sleep at once if a runnable task is
# sticky to this thread. `ran_chunk`: the worker comes from a chunk, whose region may still
# run; only then does a busy pool restart the spin time. A worker that no region uses
# counts from the end of its last chunk or wake-up: with loops back to back the pool is
# nearly always busy, and the worker the loops leave out (the one of the caller's thread
# when the caller runs on the interactive thread, or the ones beyond the number of chunks)
# would never sleep. A worker that sees the pool idle once knows its region has ended; one
# that misses the gap before the next region takes that one for its own. `ranges`: whether
# a range region runs (`RangeState.active`), which makes the worker yield its CPU.
function wait_for_command(
        ws::WorkerState, flag::SleepFlag, ev::Base.Event, oversubscribed::Bool,
        ran_chunk::Bool, ranges::RangeState
    )
    spins = 0
    # `t0`: start of the spin time; `tstart`: start of the wait, which a busy pool does not
    # restart
    t0 = tstart = time_ns()
    tyield = tevents = t0
    own_region = ran_chunk
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
            if own_region
                pool_busy() ? (t0 = now) : (own_region = false)
            end
            spun_out = now - t0 > SPIN_NS[]
            sleepy = spun_out && PHASE[] == 0
            lw = !sleepy && local_work()
            if !sleepy && !lw && queued_work()
                if run_queued_task()
                    # Preserve `own_region`: this worker's region may still be running.
                    spins = 0
                    t0 = tstart = tyield = tevents = time_ns()
                    continue
                end
            elseif sleepy || lw
                # Re-check after announcing the sleep: either we see the command here, or
                # the waker sees `sleeping` and notifies `ev`
                @atomic flag.sleeping = true
                if (@atomic ws.cmd) <= 0
                    wait(ev)
                end
                # Relaxed: a waker that still sees `true` only notifies once too often
                @atomic :monotonic flag.sleeping = false
                spins = 0
                t0 = tstart = tyield = tevents = time_ns()
                own_region = false
                continue
            end
            if now - tstart > EVENTS_NS && now - tevents > 20_000
                # Spinning longer than usual (a `parallel_phase`, a long spin time, a region
                # of its own that runs on): run the libuv event loop now and then, as the
                # scheduler would on an idle thread. Otherwise, with a worker on every
                # thread, timers and I/O would wait for the spinning to end, and a task
                # that holds the pool until a timer fires would never get it back (#73)
                Base.process_events()
                tevents = now
            end
            if (oversubscribed || (own_region && (@atomic :monotonic ranges.active))) &&
                    now - tyield > 5_000
                # Yield on oversubscribed CPUs and after a chunk while a range region runs,
                # so spinning workers do not starve its dispatcher. On M1, only 4 of 8
                # runnable threads ran at once; waiting a 10 ms time slice made cold range
                # launches take 4-13 ms instead of 0.3 ms.
                # Clear `active` before releasing the pool: workers that miss the idle gap
                # must not extend this policy to the next region. Stale reads cost one
                # extra yield or spin. Other regions retain their existing policy.
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

# Run commands until `STOP` (returns `true`) or an exception (`false`). One exception
# handler per stretch of commands, not per chunk (an `enter` costs about as much as a
# small chunk). Not inlined: on Julia 1.10 the frame that caught the exception would keep
# it, and the arrays it refers to, alive while the worker waits for the next command.
@noinline function run_commands(pool::Pool, k::Int)
    ws = pool.workers[k]
    ev = pool.events[k]
    flag = pool.sleepflags[k]
    ran_chunk = false
    try
        while true
            cmd = wait_for_command(ws, flag, ev, pool.oversubscribed, ran_chunk, pool.ranges)
            if cmd == STOP
                @atomic :release ws.cmd = 0
                return true
            end
            wake_children(pool, cmd)
            run_job_in_scope(pool, ws, k, cmd)
            @atomic :release ws.cmd = 0
            ran_chunk = true
        end
    catch err
        # Thrown by the chunk, or into the task while it waited (an interrupt on its
        # thread): then a command that arrived meanwhile, or `STOP`, is still to be read
        if (@atomic :monotonic ws.cmd) == RUNNING
            # (The dispatcher clears this worker's slot, see `dispatch!`)
            pool.errs[k] = ChunkFailedException(err, catch_backtrace())
            @atomic :monotonic pool.failed = true
            @atomic :release ws.cmd = 0
        end
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
        pad_before(), false, pad_after(),
        tid2worker, [SleepFlag() for _ in 1:nw], ChunkCounter(),
        Any[nothing for _ in 1:nw], zeros(Int, nw), is_oversubscribed(), 0, RangeState(nw)
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
# acquire/release in the non-inlined `run_chunks`, `run_fold` and `run_foreach_chunk`.
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
    # Before the workers start, which read it
    DISPATCHER_CLEARS[] = SLOT_CLEARING()
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
# worker `k` only. The dispatcher stores the chunk function; the worker or the dispatcher
# empties the slot after the chunk, depending on the CPU (`DISPATCHER_CLEARS`). The vector
# is created once per type like `job_slot`'s cache, and grown when a region first needs the
# slots (so an image holds it empty).
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
# `give_cpu`: yield CPU time each spin on oversubscribed range regions (`wait_workers`).
@inline function wait_idle(pool::Pool, k::Int, give_cpu::Bool = false)
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
            give_cpu ? os_yield() : cpu_pause()
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

# Calling thread's pool worker, or zero if it has none.
@inline function caller_worker(pool::Pool)
    tid = Threads.threadid()
    return tid <= length(pool.tid2worker) ? @inbounds(pool.tid2worker[tid]) : 0
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

# Called for chunk 1, and by the worker that claims chunk `chunk` before running it (a
# chunk the dispatcher steals has no children left to wake, see `steal_chunks`): wake the
# workers of chunks `2chunk` and `2chunk + 1` (a binary tree: `n` sleeping workers wake in
# O(log n) steps).
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
# `release`: release the pool at the end (`release_pool`), so that the region needs no
# exception handler but this one (`run_chunks`); a caller with work to do before the
# release (`run_fold` empties its buffer) passes `false` and releases it itself.
# `skip`: range jobs pass their captured caller-worker index so job and dispatch agree
# even after task migration. Other jobs sample it here through the default argument.
# Inlined into its (non-inlined) callers: as a function of its own under its own
# handler, nested in the caller's, it spilled about 20 more values per worker to the
# stack (Cascade Lake: 40 ns per loop with 4 threads, 85 with 8); merging the handlers
# without inlining gained nothing.
@inline function dispatch!(
        pool::Pool, f::F, n::Int, release::Bool = false, skip::Int = caller_worker(pool)
    ) where {F}
    scope = current_scope()
    has_scope = scope !== nothing
    copies = own_slots(F)
    clears = copies && DISPATCHER_CLEARS[]
    base = cmd_base(n, skip, has_scope)
    # Whether every worker of the region is idle (also when a chunk threw), and whether the
    # workers' slots are empty: both until the first command goes out. `stored`: whether
    # the chunk function is in `slot`.
    drained = cleared = true
    stored = false
    # The `finally` reads the slots only through what is assigned here, before the
    # handler. Neither can throw: both are per-type constants (`job_slot` fills the cache,
    # `worker_slots` grows the vector).
    slot_cache = job_slot_cache(F)
    wslots = worker_slot_vector(F)
    try
        # Inside the handler, so that the pool is released whatever throws (also an
        # interrupt during the first region of a type, which creates the slots)
        slot = job_slot(F)
        slot[] = f
        stored = true
        job = job_closure(slot)
        # The workers' slots are used, and need to exist, only for a small `f`
        # (`COPY_BYTES`)
        copies && worker_slots(F, length(pool.workers))
        # Not alone in a block (here after the join of `copies && ...`): Julia 1.13 runs
        # `cfg_simplify!` on a const-propagated callee before inlining it, and that drops
        # the handler's copies (`UpsilonNode`s) of the assignments of a block it merges
        # into its successor. The `finally` would then see `drained` from before the `try`
        # and return while workers still run chunks (test/exceptions.jl, "interrupted
        # while waiting for the workers"). JuliaLang/julia#63994, fixed by #63995: the
        # workaround can go once the minimum 1.13.x has the fix
        drained = cleared = false
        # Published by the release stores of the commands; stored only when present, so
        # the line stays shared
        has_scope && (@atomic :monotonic pool.scope = scope)
        # (Published by the release stores of the commands; see `wait_for_command`)
        f isa RangeJob && (@atomic :monotonic pool.ranges.active = true)
        # Decreasing chunk order: see `wake_children`
        for chunk in n:-1:2
            k = worker_for_chunk(chunk, skip)
            # Normally idle (an interrupted region waits for its workers); detects a dead one.
            # Relaxed first: the last region's wait for the worker already acquired its
            # results, so only a worker still busy needs `wait_idle`'s acquire loop
            if (@atomic :monotonic (@inbounds pool.workers[k]).cmd) != 0 && !wait_idle(pool, k)
                # Take back the chunks already handed out before giving up
                cancel_chunks(pool, chunk + 1, n, skip, base, wslots, copies)
                wait_workers(pool, chunk + 1, n, skip)
                drained = true
                dead_worker_error(k)
            end
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
        # Range jobs wake all workers; other jobs use the tree. If chunk 1 throws,
        # `finally` cancels pending chunks and drains started ones. RangeJob branches
        # resolve at compile time, preserving other jobs' dispatch paths.
        f isa RangeJob ? wake_all(pool, n, skip, base) : wake_children(pool, base | 1)
        run_chunk(slot, 1)   # the workers' code, see `run_chunk`
        # Rescue unclaimed participants, allowing range owners time to wake.
        if f isa RangeJob
            rescue_ranges(pool, f, slot, n, skip, base, wslots, copies)
        else
            steal_chunks(pool, slot, n, skip, base, wslots, copies)
        end
        # An oversubscribed range dispatcher must yield CPU time to its workers.
        wait_workers(pool, 2, n, skip, wslots, clears, f isa RangeJob && pool.oversubscribed)
        drained = cleared = true
        # Acquired by `wait_workers`. Check job-level failures, then participant failures
        # recorded by `run_commands`.
        if @atomic :monotonic pool.failed
            f isa GreedyJob && throw_greedy_errors(f.errs, f.errchunks, n)
            f isa RangeJob && throw_range_error(f.state)
            throw_first_error(pool, n, skip)
        end
    finally
        # Thrown outside the chunks, e.g. an interrupt while waiting for the workers:
        # running chunks use the region's slots, buffers and greedy counter, which the next
        # region reuses, so wait for them before returning
        drained || drain_workers(pool, f, n, skip, base, wslots, copies)
        # Keep neither the chunk function (`clear_slot!`), nor the scope, nor the chunks'
        # exceptions alive (every used worker is idle here, also when a chunk threw)
        stored && clear_slot!(slot_cache[]::Slot{F})
        has_scope && (@atomic :monotonic pool.scope = nothing)
        # After a region that ended normally there is nothing left to clean up: the
        # workers' slots are empty, no chunk failed
        if !(cleared && !(@atomic :monotonic pool.failed))
            clean_up_region(pool, n, skip, wslots, copies, clears, cleared)
            # Drop a range error superseded by an interrupt.
            f isa RangeJob && clear_range_error!(f.state)
        end
        # Clear before releasing the pool, including after failure or interruption.
        f isa RangeJob && (@atomic :monotonic pool.ranges.active = false)
        release && release_pool()
    end
    return nothing
end

# The first exception of the region's chunks, in chunk order (`failed` is set)
@noinline function throw_first_error(pool::Pool, n, skip)
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        err = @inbounds pool.errs[k]
        err === nothing || throw_worker_error(err::ChunkFailedException)
    end
    return nothing
end

# After a region that failed or was cut short: drop the chunks' exceptions and the chunk
# functions left in the workers' slots
@noinline function clean_up_region(pool::Pool, n, skip, wslots, copies, clears, cleared)
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        # If the dispatcher empties the slots: emptied while waiting for the workers,
        # else here
        clears && !cleared && clear_slot!(@inbounds wslots[k])
        # A worker whose chunk threw, or that died with its command set
        # (`dead_worker_error`), left the chunk function in its slot
        if (@inbounds pool.errs[k]) !== nothing ||
                (@atomic :monotonic (@inbounds pool.workers[k]).cmd) != 0
            copies && !clears && clear_slot!(@inbounds wslots[k])
            @inbounds pool.errs[k] = nothing
        end
    end
    # Also when the region was interrupted before it reported them
    fill!(pool.greedy_errs, nothing)
    @atomic :monotonic pool.failed = false
    return nothing
end

# Run the chunks (greedy participants) of a region their worker has not claimed (libomp's `static_steal`
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
#
# Decreasing chunk order: the children of a chunk in the wake-up tree (`2chunk`,
# `2chunk + 1`) are stolen or claimed before it, so stealing it wakes no one. In increasing
# order the dispatcher woke the children of each chunk it stole, and then stole their
# chunks too: a short loop after the workers fell asleep woke every worker for nothing.
@inline function steal_chunks(
        pool::Pool, slot, n::Int, skip::Int, base::Int, wslots, copies::Bool
    )
    for chunk in n:-1:2
        k = worker_for_chunk(chunk, skip)
        ws = @inbounds pool.workers[k]
        cmd = base | chunk
        if (@atomic :monotonic ws.cmd) == cmd &&
                (@atomicreplace :monotonic ws.cmd cmd => 0).success
            copies && clear_slot!(@inbounds wslots[k])
            @atomic :monotonic pool.stolen = (@atomic :monotonic pool.stolen) + 1
            try
                run_chunk(slot, chunk)
            catch err
                @inbounds pool.errs[k] = ChunkFailedException(err, catch_backtrace())
                @atomic :monotonic pool.failed = true
            end
        end
    end
    return nothing
end

"""
    ThreadTeams.stolen_chunks() -> Int

Number of chunks (with [`GreedyTeamScheduler`](@ref), participants) the calling task ran
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
    return wait_workers(pool, from, n, skip, worker_slot_vector(Nothing), false)
end
# The same, emptying the slot of each worker (`copies`) as soon as the worker is idle: the
# stores overlap the wait for the others, so that only the last worker's is on the
# dispatcher's way out of the region (`worker_slot_vector`)
# `give_cpu` prevents an oversubscribed dispatcher from starving a worker: on Orin
# with `-t 6,1`, spinning stretched a 20 µs range to 2 ms.
@inline function wait_workers(
        pool::Pool, from, n, skip, wslots, copies::Bool, give_cpu::Bool = false
    )
    dead = 0
    for chunk in from:n
        k = worker_for_chunk(chunk, skip)
        if wait_idle(pool, k, give_cpu)
            copies && clear_slot!(@inbounds wslots[k])
        else
            dead = k
        end
    end
    dead == 0 || dead_worker_error(dead)
    return nothing
end

# Cancel the chunks of a region no worker has claimed and wait for the others, also when
# interrupted again meanwhile (the region's exception is already on its way)
@noinline function drain_workers(pool::Pool, f, n, skip, base, wslots, copies::Bool)
    f isa GreedyJob && (@atomic :monotonic f.counter.next = GREEDY_FAILED)
    f isa RangeJob && (@atomic :monotonic f.state.stop = true)
    cancel_chunks(pool, 2, n, skip, base, wslots, copies)
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        while true
            try
                wait_idle(pool, k)
                break
            catch
            end
        end
    end
    return nothing
end

"""
    ThreadTeams.ChunkFailedException

The exception of a chunk that did not run on the calling task: `exception`, and the raw
`backtrace` of the task that ran it. Like `TaskFailedException`, it keeps the raw backtrace
and only looks up its stack frames when it is shown (`CapturedException` looks them up
when it is created: milliseconds on Julia 1.10 and 1.11).

A loop rethrows the chunk's original exception; the `ChunkFailedException` is on the
exception stack below it (`current_exceptions()`), shown as "caused by: ...".
"""
struct ChunkFailedException <: Exception
    exception::Any
    backtrace::Any
end

function Base.showerror(io::IO, ex::ChunkFailedException, bt = nothing; backtrace = true)
    print(io, "ChunkFailedException")
    bt !== nothing && backtrace && Base.show_backtrace(io, bt)
    println(io)
    printstyled(io, "\n    nested chunk error: "; color = Base.error_color())
    b = IOBuffer()
    showerror(IOContext(b, io), ex.exception, ex.backtrace; backtrace)
    print(io, replace(String(take!(b)), "\n" => "\n    "))
    return nothing
end

# Rethrow the original exception of a chunk; the backtrace of the task that ran it stays
# on the exception stack (`current_exceptions()`) as "caused by: ..."
@noinline function throw_worker_error(err::ChunkFailedException)
    return try
        throw(err)
    catch
        throw(err.exception)
    end
end

# Is a `Threads.@threads` loop running, in any task? (A process-wide counter, which
# `@spawn` does not touch.) An opaque call: only in non-inlined functions (`pool_busy`).
@inline in_threaded_region() = ccall(:jl_in_threaded_region, Cint, ()) != 0

# Take the pool for a loop with scheduler `sched`, or `nothing` if it can not have it.
# With `busy = :serial`, not inside a `@threads` loop: its iterations occupy the threads,
# and taking the pool from one of them would wake workers whose threads are running the
# others.
@inline function take_pool(sched::TeamSchedulers)
    sched.serial_when_busy && in_threaded_region() && return nothing
    return acquire_pool()
end

# How a loop runs when it can not use the pool: serially when nested in a region (or
# during precompilation, or with `busy = :serial`), on tasks when another task holds the
# pool
@inline function no_pool_mode(sched::TeamSchedulers)
    (sched.serial_when_busy || generating_output() || in_region()) && return :serial
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
            ChunkFailedException(err, catch_backtrace())
        end
    end
    for t in tasks
        Base._wait(t)
    end
    err1 === nothing || throw_worker_error(err1)
    for t in tasks
        istaskfailed(t) && throw_worker_error(ChunkFailedException(t.result, t.backtrace))
    end
    return nothing
end

# The chunks of a greedy region of `p` participants: participant `k` runs chunk `k`, then
# claims chunk indices from the shared counter (which starts at `p`) and runs them until
# all `m` are taken. The first chunks are placed as in a static region, so that loops
# over the same data, back to back, find much of it in the cache of the core that
# touched it last. Exceptions are stored with the chunk that threw, and a failing chunk
# stops the hand-out; the counter hands out chunks in increasing order and the first
# chunks always run (a participant whose worker has not started is run by the
# dispatcher, see `steal_chunks`), so every chunk before it runs, as in a serial loop.
#
# On the pool, the exception handler of each participant is the only one besides
# `dispatch!`'s, which releases the pool: a chunk's failure also sets the pool's `failed`
# flag, which the dispatcher checks after every region anyway, and `errs` is empty
# between regions (cleared with the rest after a failure, `clean_up_region`), so a region
# that ended normally neither resets nor scans it.
struct GreedyJob{F}
    f::F
    counter::ChunkCounter
    m::Int
    errs::Vector{Any}
    errchunks::Vector{Int}
    pool::Union{Pool, Nothing}   # `nothing` on spawned tasks (`run_off_pool`)
    nparts::Int
end
function (g::GreedyJob)(p::Int)
    chunk = p
    try
        g.f(chunk)
        # With no chunks beyond the participants' first, as cheap as a static region
        while g.m > g.nparts
            # Relaxed: the counter only claims indices; the job is published with the
            # commands, results with the workers' release stores. A load first: once the
            # chunks have run out the participants share the counter's line instead of
            # taking turns writing it
            (@atomic :monotonic g.counter.next) >= g.m && break
            chunk = @atomic :monotonic g.counter.next += 1
            chunk > g.m && break
            g.f(chunk)
        end
    catch err
        # Stop the hand-out first: capturing the exception takes a while
        @atomic :monotonic g.counter.next = GREEDY_FAILED
        @inbounds g.errs[p] = ChunkFailedException(err, catch_backtrace())
        @inbounds g.errchunks[p] = chunk
        # Relaxed: published by the worker's release store `cmd = 0`, as `errs`
        pool = g.pool
        pool === nothing || (@atomic :monotonic pool.failed = true)
    end
    return nothing
end

# Counter value after a failure: no participant claims a chunk afterwards
const GREEDY_FAILED = typemax(Int) >> 1

# Rethrow the exception of the lowest failing chunk of a greedy region of `p` participants,
# if any
@noinline function throw_greedy_errors(errs::Vector{Any}, errchunks::Vector{Int}, p::Int)
    first = 0
    for k in 1:p
        if @inbounds(errs[k]) !== nothing &&
                (first == 0 || @inbounds(errchunks[k] < errchunks[first]))
            first = k
        end
    end
    first == 0 && return nothing
    throw_worker_error(@inbounds(errs[first])::ChunkFailedException)
end

# Run `f(chunk)` for `chunk in 1:n` on the pool, which the caller holds; `release`: see
# `dispatch!`
@inline run_on_pool(pool::Pool, f::F, n::Int, ::TeamScheduler, release::Bool = false) where {F} =
    dispatch!(pool, f, n, release)
# Also with `m ≤ nworkers()`, whose participants then skip the counter: a static region
# specialized on `F` as well would cost each loop body about 150 ms more to compile
# (`foreach_chunk`, compiled once per type, has both)
@inline run_on_pool(pool::Pool, f::F, m::Int, ::GreedyTeamScheduler, release::Bool = false) where {F} =
    run_greedy(pool, f, m, min(m, length(pool.workers)), release)
# The `m ≥ p` chunks on `p ≤ nworkers()` participants
@inline function run_greedy(pool::Pool, f::F, m::Int, p::Int, release::Bool) where {F}
    # Published to the workers by the (release) stores of the commands
    @atomic :monotonic pool.counter.next = p
    job = GreedyJob(f, pool.counter, m, pool.greedy_errs, pool.greedy_errchunks, pool, p)
    dispatch!(pool, job, p, release)
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n` on spawned tasks (another task holds the pool)
run_off_pool(f::F, n::Int, ::TeamScheduler) where {F} = run_tasks(f, n)
function run_off_pool(f::F, m::Int, ::GreedyTeamScheduler) where {F}
    p = min(m, nworkers())
    errs = Any[nothing for _ in 1:p]
    errchunks = zeros(Int, p)
    counter = ChunkCounter()
    @atomic :monotonic counter.next = p
    run_tasks(GreedyJob(f, counter, m, errs, errchunks, nothing, p), p)
    throw_greedy_errors(errs, errchunks, p)
    return nothing
end

# Run `f(chunk)` for `chunk in 1:n`: in parallel if the pool is free, serially if nested,
# on tasks (or serially, `busy = :serial`) if another task holds the pool. With the
# `TeamScheduler`, `n ≤ nworkers()`.
# Not inlined: its atomics must stay out of the caller (`pool_busy`). `run_chunks` and
# `run_fold` have Enzyme rules (ext/ThreadTeamsEnzymeCoreExt.jl): Enzyme can not
# differentiate the team itself (atomics, long-lived tasks), so the rules differentiate
# the chunk function and run the derivative chunks on the team.
@noinline function run_chunks(f::F, n::Int, sched::TeamSchedulers) where {F}
    pool = take_pool(sched)
    if pool === nothing
        if no_pool_mode(sched) === :tasks
            run_off_pool(f, n, sched)
        else
            for chunk in 1:n
                f(chunk)
            end
        end
        return nothing
    end
    # `dispatch!`'s exception handler releases the pool: one handler for the region
    run_on_pool(pool, f, n, sched, true)
    return nothing
end

"""
    ThreadTeams.foreach_chunk(f, m::Int, p::Int = ThreadTeams.nworkers())

Call `f(c)` for every `c in 1:m`, each once, on at most `min(p, m, nworkers())` threads at
a time, the calling task being one of them. With as many chunks as threads, chunk `c` goes
to the `c`-th participant (as with [`TeamScheduler`](@ref)); with more, participant `c`
starts with chunk `c`, then the participants take the others from a shared counter until
none are left (as with [`GreedyTeamScheduler`](@ref)). `f` decides what a chunk is, e.g.
the iterations `(c - 1) * n ÷ m + 1:c * n ÷ m` of a loop of `n`.

Runs every chunk serially on the calling task, in order, when the team is busy (held by
another task), when the call is nested in a team loop, while a `Threads.@threads` loop
runs, during precompilation, and when `m ≤ 1` or `p == 1`: never on spawned tasks (like
`busy = :serial`).

Chunks that run at the same time never share a thread; a chunk runs on one thread unless
it yields. Chunks run in the caller's dynamic scope and world age. If chunks throw, the
exception of the lowest failing chunk is rethrown once the chunks already started have
finished: every chunk before it runs, and the counter hands out no chunk after the
failure is noticed.

The region code is compiled once per type of `f`. For a concrete type `S`,
`precompile(ThreadTeams.foreach_chunk, (S, Int, Int))` compiles all of it but the body of
the workers' closure, which the first loop compiles when it creates the closure (a few
milliseconds per type, against a few hundred without the statement).
"""
function foreach_chunk(f::F, m::Int, p::Int = nworkers()) where {F}
    # (Below the greedy counter's value after a failure, see `GreedyJob`)
    0 <= m <= typemax(Int) >> 2 ||
        throw(ArgumentError("the number of chunks must be in 0:$(typemax(Int) >> 2), got $m"))
    p >= 1 || throw(ArgumentError("the number of threads must be positive, got $p"))
    p = min(p, m, nworkers())
    if p <= 1
        for c in 1:m
            f(c)
        end
        return nothing
    end
    run_foreach_chunk(f, m, p)
    return nothing
end

@noinline function run_foreach_chunk(f::F, m::Int, p::Int) where {F}
    pool = in_threaded_region() ? nothing : acquire_pool()
    if pool === nothing
        for c in 1:m
            f(c)
        end
    elseif m == p
        prepare_job_closure(F)
        # `dispatch!`'s exception handler releases the pool
        dispatch!(pool, f, m, true)
    else
        prepare_job_closure(GreedyJob{F})
        run_greedy(pool, f, m, p, true)
    end
    return nothing
end

# Create the closure of `job_closure` for chunk functions of type `F` if it is missing or
# from an older world, with a static call: `job_closure` creates it with a dynamic call
# (for Enzyme, which never sees `foreach_chunk`), which a `precompile` statement for
# `foreach_chunk` would not reach, leaving about 50 ms of compilation per type for the
# first loop. The caller holds the pool, which this releases if it throws.
@inline function prepare_job_closure(::Type{F}) where {F}
    job = job_closure_cache(F)[]
    (job isa JobClosure && job.world == world_age()) || new_job_closure!(F)
    return nothing
end
@noinline function new_job_closure!(::Type{F}) where {F}
    try
        job_closure_cache(F)[] = new_job_closure(job_slot(F), worker_slot_vector(F))
    catch
        release_pool()
        rethrow()
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
    pool = take_pool(sched)
    if pool === nothing
        if no_pool_mode(sched) === :tasks
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
    # A region can not end while we wait in one of its chunks (on the pool or, while
    # another task holds the pool, on spawned tasks: `run_tasks`)
    in_region() && error("ThreadTeams.shutdown! called from inside a parallel loop")
    while !(@atomicreplace :acquire :monotonic POOL.busy false => true).success
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
