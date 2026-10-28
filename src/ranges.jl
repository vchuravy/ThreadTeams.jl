# `foreach_range` assigns balanced contiguous blocks to fixed pool workers. Dispatch
# chunks still follow `worker_for_chunk`: `range_block` maps them to blocks using the
# caller's worker `skip`, sampled once and passed to `dispatch!`.
# A caller on worker s's thread (s ≤ p) owns block s; otherwise chunk 1 is empty and p workers
# receive chunks 2:p+1. Rescued chunks retain their block, regardless of executing thread.
#
# Range regions wake all workers directly and delay rescue of unstarted blocks
# (`rescue_due`). Immediate rescue with a wake tree ran 84% of a cold 64-thread launch
# on the caller. Static mode needs only the shared failure state; steal mode uses packed
# blocks behind `init_blocks!`, `claim_own`, `steal_from` and `block_started`.
#
# Memory ordering: command publication releases the initialized job and block state.
# Claims and republication need only relaxed atomics: they transfer indices, not callback
# results. Workers publish results and errors with release stores of `cmd = 0`.
# A stale stop flag may allow another callback after failure, as the contract permits.

# Inclusive bounds of block k; the first r blocks have q + 1 indices, the rest q.
@inline function block_bounds(q::Int, r::Int, k::Int)
    lo = (k - 1) * q + min(k - 1, r)
    return lo + 1, lo + q + (k <= r)
end

# The block that chunk `chunk` of a range region runs, `0` for none: see the header
@inline function range_block(chunk::Int, skip::Int, p::Int)
    chunk == 1 && return (1 <= skip <= p) ? skip : 0
    return worker_for_chunk(chunk, skip)
end

@inline ranges_stopped(st::RangeState) = @atomic :monotonic st.stop

# Allow 2 µs after observing an awake worker with its command pending. Measured workers
# claim within 1 µs of waking; a longer delay suggests descheduling. Sleeping workers get
# a wake-up margin growing with p: 0.5 ms at 2 workers, 0.75 ms at 4, 1.75 ms at 64.
const RANGE_AWAKE_NS = 2_000
range_deadline_ns(p::Int) = 250_000 * (1 + (8 * sizeof(Int) - leading_zeros(p - 1)))
range_deadline(p::Int) = time_ns() + UInt64(range_deadline_ns(p))
# Spin briefly, then offer the CPU with `sched_yield` on oversubscribed pools; after
# RANGE_SPIN_NS, park between scans on all pools (other processes may compete too).
# The short spin is below RANGE_AWAKE_NS, giving a competing worker time to claim.
# `yield()` alone may resume immediately; Julia timers cost ~2 ms. `nanosleep` costs
# 52-57 µs on Linux and 17 µs on macOS, including timer slack. Parking immediately made
# warm oversubscribed launches take 65-100 µs instead of 13-30; spinning for 100 µs made
# cold launches take 170 µs instead of 30-40 (9950X, 4 workers + interactive on 4 CPUs).
const RANGE_SPIN_NS = 100_000
const RANGE_SPIN_OVERSUBSCRIBED_NS = 1_000
const RANGE_PARK_S = 10.0e-6

# Rescue after the deadline, or after an awake worker has left work pending long enough.
# `seen` is the first awake observation, or zero if none.
@inline rescue_due(asleep::Bool, seen::UInt64, now::UInt64, deadline::UInt64) =
    now >= deadline || (!asleep && seen != 0 && now - seen >= RANGE_AWAKE_NS)

# The same rule for the owner of steal block `b` with its command pending, shared by
# thieves and the dispatcher: `b.seen` is the first awake observation of the region,
# never reset before `init_blocks!`, so waits do not renew; the owner must still be awake.
# Another observer's `seen` may follow our `now`, which is then not due yet.
@inline function owner_due(b::RangeBlock, asleep::Bool, now::UInt64, deadline::UInt64)
    now >= deadline && return true
    asleep && return false
    seen = @atomic :monotonic b.seen
    if seen == 0
        old, ok = @atomicreplace :monotonic :monotonic b.seen UInt64(0) => now
        seen = ok ? now : old
    end
    return seen <= now && now - seen >= RANGE_AWAKE_NS
end

# Wake all sleeping workers; their usual child wake-ups remain harmless.
@inline function wake_all(pool::Pool, n::Int, skip::Int, base::Int)
    for chunk in 2:n
        wake_chunk(pool, chunk, skip, base)
    end
    return nothing
end

# Rescue pending chunks once `rescue_due`, waking their children on their behalf.
# On failure, cancel pending commands; leave dead-worker reporting to `wait_workers`.
# Yield and park after bounded spinning. Placement was captured before any task migration.
@noinline function rescue_ranges(
        pool::Pool, job, slot, n::Int, skip::Int, base::Int, wslots, copies::Bool
    )
    st = job.state
    seen = st.seen
    # Steal jobs use their blocks' clocks
    if !(job isa StealRanges)
        for chunk in 2:n
            @inbounds seen[worker_for_chunk(chunk, skip)] = 0
        end
    end
    spin = pool.oversubscribed ? RANGE_SPIN_OVERSUBSCRIBED_NS : RANGE_SPIN_NS
    tstart = UInt64(0)
    while true
        if ranges_stopped(st)
            cancel_chunks(pool, 2, n, skip, base, wslots, copies)
            return nothing
        end
        pending = false
        # Read the clock only for pending commands; otherwise this is one scan of loads.
        now = UInt64(0)
        for chunk in n:-1:2
            k = worker_for_chunk(chunk, skip)
            ws = @inbounds pool.workers[k]
            cmd = base | chunk
            # A load first, relaxed: see `steal_chunks`
            (@atomic :monotonic ws.cmd) == cmd || continue
            asleep = @atomic :monotonic (@inbounds pool.sleepflags[k]).sleeping
            now == 0 && (now = time_ns())
            if job isa StealRanges
                # Chunk `chunk` runs block `k`
                due = owner_due(@inbounds(job.blocks[k]), asleep, now, job.deadline)
            else
                t = asleep ? UInt64(0) : (@inbounds(seen[k]) == 0 ? now : @inbounds(seen[k]))
                @inbounds seen[k] = t
                due = rescue_due(asleep, t, now, job.deadline)
            end
            if due
                (@atomicreplace :monotonic ws.cmd cmd => 0).success || continue
                wake_children(pool, cmd)
                copies && clear_slot!(@inbounds wslots[k])
                @atomic :monotonic pool.stolen = (@atomic :monotonic pool.stolen) + 1
                # Range jobs catch callback failures.
                run_chunk(slot, chunk)
                now = UInt64(0)
            else
                pending = true
            end
        end
        pending || return nothing
        t = time_ns()
        tstart == 0 && (tstart = t)
        if t - tstart < spin
            cpu_pause()
            GC.safepoint()
        elseif pool.oversubscribed && t - tstart < RANGE_SPIN_NS
            # Offer the CPU to competing OS threads.
            os_yield()
            GC.safepoint()
        else
            for chunk in 2:n
                k = worker_for_chunk(chunk, skip)
                (@atomic :monotonic (@inbounds pool.workers[k]).cmd) == base | chunk &&
                    istaskdone(@inbounds pool.tasks[k]) && return nothing
            end
            yield()
            Libc.systemsleep(RANGE_PARK_S)
        end
    end
    return
end

# Stop new calls before capturing the backtrace. The first recorded failure wins,
# including when the caller's own block fails.
@noinline function record_range_failure!(st::RangeState, pool::Pool, err)
    @atomic :monotonic st.stop = true
    e = ChunkFailedException(err, catch_backtrace())
    @atomicreplace :monotonic st.err nothing => e
    # Relaxed: published by the worker's release store `cmd = 0`, as `errs`
    @atomic :monotonic pool.failed = true
    return nothing
end

# With the pool held and participants drained, remove and rethrow the first failure.
@noinline function throw_range_error(st::RangeState)
    err = @atomic :monotonic st.err
    err === nothing && return nothing
    @atomic :monotonic st.err = nothing
    throw_worker_error(err::ChunkFailedException)
end

# Drop an error superseded by an interrupt, after all participants finish.
@inline function clear_range_error!(st::RangeState)
    (@atomic :monotonic st.err) === nothing || (@atomic :monotonic st.err = nothing)
    return nothing
end

# Reset under the pool lock. Write only when set, keeping the uncontended line shared.
@inline function reset_ranges!(st::RangeState)
    clear_range_error!(st)
    (@atomic :monotonic st.stop) && (@atomic :monotonic st.stop = false)
    return nothing
end

# One call per block; the caller records failures through the same handler as workers.
struct StaticRanges{F}
    f::F
    q::Int
    r::Int
    p::Int
    skip::Int
    deadline::UInt64   # `range_deadline`
    state::RangeState
    pool::Pool
end
function (j::StaticRanges)(chunk::Int)
    k = range_block(chunk, j.skip, j.p)
    (k == 0 || ranges_stopped(j.state)) && return nothing
    lo, hi = block_bounds(j.q, j.r, k)
    try
        j.f(lo, hi)
    catch err
        record_range_failure!(j.state, j.pool, err)
    end
    return nothing
end

# Packed static-steal: owners CAS-claim a grain from the front; thieves take a quarter
# from the back, preserving the owner's next indices and adjacent data. A thief reserves
# one grain and republishes the rest in its own empty block. Only that block's owner can
# refill it, so republication needs a store rather than CAS.
#
# Reserve before republishing: otherwise thieves can exchange the same range forever
# without executing it. This also prevents ABA for nonempty ranges: each transfer reserves
# indices permanently, so a previously observed nonempty range cannot recur.
#
# A thief scans all other blocks before retiring; a block passed earlier may be refilled.
# The participant that refills it remains responsible for draining it. Each successful
# claim or steal reserves at least one index, ensuring progress.
struct StealRanges{F}
    f::F
    blocks::Vector{RangeBlock}
    q::Int
    r::Int
    p::Int
    skip::Int
    grain::Int
    deadline::UInt64
    state::RangeState
    pool::Pool
end
const RangeJob = Union{StaticRanges, StealRanges}

# Packed bounds use 31 and 32 bits. Larger loops use static mode; on 32-bit systems
# the API's `typemax(Int) >> 2` limit is already lower.
const STEAL_MAX_N = Int(typemax(Int32))

const RANGE_STARTED = UInt64(1) << 63
# UInt64 on all platforms: started | lo | hi, with 0 ≤ lo ≤ hi ≤ STEAL_MAX_N.
# The unclaimed indices are lo + 1:hi.
@inline pack_range(started::Bool, lo::Int, hi::Int) =
    (started ? RANGE_STARTED : UInt64(0)) | (UInt64(lo) << 32) | UInt64(hi)
@inline unpack_range(w::UInt64) =
    ((w & RANGE_STARTED) != 0, Int((w >> 32) & 0x7fffffff), Int(w & 0xffffffff))

# Initialize under the pool lock, before publishing commands.
@inline function init_blocks!(blocks::Vector{RangeBlock}, q::Int, r::Int, p::Int)
    for k in 1:p
        lo, hi = block_bounds(q, r, k)
        b = @inbounds blocks[k]
        @atomic :monotonic b.word = pack_range(false, lo - 1, hi)
        @atomic :monotonic b.seen = UInt64(0)
    end
    return nothing
end

# Claim at most one grain; return inclusive bounds, or (1, 0) if empty.
# Load before CAS so empty scans share the cache line rather than taking ownership.
@inline function claim_own(j::StealRanges, me::Int)
    b = @inbounds j.blocks[me]
    while true
        w = @atomic :monotonic b.word
        _, lo, hi = unpack_range(w)
        lo >= hi && return 1, 0
        nc = min(j.grain, hi - lo)
        (@atomicreplace :monotonic :monotonic b.word w => pack_range(true, lo + nc, hi)).success &&
            return lo + 1, lo + nc
    end
    return
end

# Steal a tail quarter into our empty block; reserve one grain and republish the rest.
# The caller checks `steal_allowed`.
@inline function steal_from(j::StealRanges, me::Int, v::Int)
    b = @inbounds j.blocks[v]
    g = j.grain
    while true
        w = @atomic :monotonic b.word
        started, lo, hi = unpack_range(w)
        lo >= hi && return 1, 0
        slo = hi - cld(hi - lo, 4)
        (@atomicreplace :monotonic :monotonic b.word w => pack_range(started, lo, slo)).success ||
            continue
        rlo = slo + min(g, hi - slo)
        # Relaxed: whoever claims from it uses nothing but the word
        rlo < hi && (@atomic :monotonic (@inbounds j.blocks[me]).word = pack_range(true, rlo, hi))
        return slo + 1, rlo
    end
    return
end

# Owners set `started` on their first claim; thieves leave the victim's bit unchanged.
@inline block_started(j::StealRanges, v::Int) =
    ((@atomic :monotonic (@inbounds j.blocks[v]).word) & RANGE_STARTED) != 0

# `steal_allowed` verdicts: take; wait for an awake owner to claim or become due;
# protected (owner asleep or claimed, or the caller's block) until the deadline
const STEAL_YES = 0
const STEAL_SOON = 1
const STEAL_NO = 2

# The verdict for an unstarted, nonempty block `b` whose owner's command is `pending`
@inline function unstarted_verdict(
        b::RangeBlock, pending::Bool, asleep::Bool, now::UInt64, deadline::UInt64
    )
    pending || return now >= deadline ? STEAL_YES : STEAL_NO
    owner_due(b, asleep, now, deadline) && return STEAL_YES
    return asleep ? STEAL_NO : STEAL_SOON
end

# Started and empty blocks are freely stealable; others follow the rescue rule. No
# waiting here: a private 2 µs wait per call and block made loaded launches take ~1 ms.
@inline function steal_allowed(j::StealRanges, v::Int)
    b = @inbounds j.blocks[v]
    started, lo, hi = unpack_range(@atomic :monotonic b.word)
    (started || lo >= hi) && return STEAL_YES
    pool = j.pool
    pending = (@atomic :monotonic (@inbounds pool.workers[v]).cmd) > 0
    asleep = @atomic :monotonic (@inbounds pool.sleepflags[v]).sleeping
    return unstarted_verdict(b, pending, asleep, time_ns(), j.deadline)
end

function (j::StealRanges)(chunk::Int)
    me = range_block(chunk, j.skip, j.p)
    # An off-pool caller rescues late participants but is not an extra thief.
    me == 0 && return nothing
    try
        run_steal(j, me)
    catch err
        record_range_failure!(j.state, j.pool, err)
    end
    return nothing
end

# Retry protected (`STEAL_NO`) blocks briefly before retiring.
const PROTECTED_PASSES = 3

# Run the participant's own block, by owner rules
@inline function run_own(j::StealRanges, me::Int)
    st = j.state
    while !ranges_stopped(st)
        lo, hi = claim_own(j, me)
        lo > hi && break
        j.f(lo, hi)
    end
    return nothing
end

@inline function run_steal(j::StealRanges, me::Int)
    st = j.state
    run_own(j, me)
    p = j.p
    retries = 0
    while true
        found = protected = soon = false
        v = me
        for _ in 2:p
            v = v == p ? 1 : v + 1
            verdict = steal_allowed(j, v)
            if verdict != STEAL_YES
                protected |= verdict == STEAL_NO
                soon |= verdict == STEAL_SOON
                continue
            end
            # Again from the same block while it has work: the next steal takes a
            # quarter of what is left
            while !ranges_stopped(st)
                lo, hi = steal_from(j, me, v)
                lo > hi && break
                found = true
                j.f(lo, hi)
                run_own(j, me)   # what it republished, unless stolen meanwhile
            end
        end
        ranges_stopped(st) && break
        # Rescan after successful steals and while an owner may become due. Retry
        # protected blocks briefly, then let their owners or the dispatcher handle them
        # rather than polling until the deadline.
        if !found
            soon || (protected && (retries += 1) <= PROTECTED_PASSES) || break
            j.pool.oversubscribed ? os_yield() : cpu_pause()
        end
    end
    return nothing
end

"""
    ThreadTeams.foreach_range(f, n::Integer; threads::Integer = ThreadTeams.nworkers(),
        schedule::Symbol = :static, grain::Integer = 0)

Call `f(first, last)` on disjoint, nonempty inclusive ranges covering `1:n` exactly once.
No calls for `n == 0`. Uses at most `p = min(threads, n, nworkers())` participants, capped
at 1,048,574 on 64-bit systems and 510 on 32-bit systems.

Split `1:n` into balanced contiguous blocks: with `q, r = divrem(n, p)`, the first `r`
blocks have `q + 1` indices and the rest have `q`. Block `k` belongs to pool worker `k`,
thread `Threads.threadpooltids(:default)[k]`, independent of the caller. A caller on
worker `k ≤ p` runs block `k`; other callers wait. The caller may rescue a worker's block
if it stays awake without claiming its command for a few microseconds, or misses a
startup deadline of roughly a millisecond (longer for larger teams). Rescued participants
are counted by `ThreadTeams.stolen_chunks()`.

- `schedule = :static` (default): one call per block, for balanced work.
- `schedule = :steal`: owners claim ranges from their block's front; idle participants
  steal from other blocks' tails. Unstarted blocks get the same startup protection as
  above. This balances uneven work, but execution placement depends on timing: a stalled
  owner can lose most of its remaining block. Already running calls cannot be stolen.
  `grain` bounds each call's range length; `0` currently selects an eighth of a block.
  For `n ≥ 2^31` it runs in static mode, ignoring `grain`.

Fixed ownership matches the preferred placement of `Threads.@threads :static` for the
same `n` and `p`. NUMA first-touch locality also requires thread affinity (e.g.
`ThreadPinning.pinthreads(:cores)` or `:numa`). Rescue, stealing, serial fallback and
migration of a non-sticky caller after a yield can change execution placement.

For nonempty input, run `f(1, n)` serially on the caller when `p == 1`, the team is busy,
the call is nested in a team loop, a `Threads.@threads` loop is active, or during
precompilation. This ignores placement, schedule and grain; `threads = 1` does not
select worker 1.

Calls run in the caller's dynamic scope and world age. Concurrent calls do not share a
thread unless they yield. After a failure, participants stop once they observe it;
concurrent calls may still start. All started calls finish before the first recorded
exception is rethrown, with its backtrace in a [`ChunkFailedException`](@ref
ThreadTeams.ChunkFailedException) below it on the exception stack. There is no completed
prefix guarantee.

`n` must be in `0:(typemax(Int) >> 2)`, `threads ≥ 1`, and `grain ≥ 0`; nonzero grain
requires `schedule = :steal`. For concrete `S`,
`precompile(ThreadTeams.foreach_range, (S, Int))` compiles both modes except the workers'
closure body (see [`foreach_chunk`](@ref ThreadTeams.foreach_chunk)) and the keyword sorter.

```julia
x = Vector{Float64}(undef, 10^8)
ThreadTeams.foreach_range(length(x)) do lo, hi
    fill!(view(x, lo:hi), 0.0)
end
ThreadTeams.foreach_range(length(x); schedule = :steal) do lo, hi
    for i in lo:hi
        x[i] = work(i)
    end
end
```
"""
function foreach_range(
        f::F, n::Integer; threads::Integer = nworkers(), schedule::Symbol = :static,
        grain::Integer = 0
    ) where {F}
    # Leave headroom for block-bound arithmetic.
    0 <= n <= typemax(Int) >> 2 ||
        throw(ArgumentError("n must be in 0:$(typemax(Int) >> 2), got $n"))
    threads >= 1 || throw(ArgumentError("the number of threads must be positive, got $threads"))
    steal = schedule === :steal
    steal || schedule === :static ||
        throw(ArgumentError("schedule must be :static or :steal, not $(repr(schedule))"))
    grain >= 0 || throw(ArgumentError("grain must be non-negative, got $grain"))
    steal || grain == 0 || throw(ArgumentError("grain is only supported with schedule = :steal"))
    n = Int(n)
    # A caller without a block dispatches `p + 1` chunks, which the command must hold
    p = Int(min(threads, n, nworkers(), CMD_MASK - 1))
    if p <= 1
        n >= 1 && f(1, n)
        return nothing
    end
    # Large ranges retain static placement when they exceed the packed representation.
    if steal && n <= STEAL_MAX_N
        # At most the longest block
        g = grain == 0 ? cld(n ÷ p, 8) : Int(min(grain, cld(n, p)))
        run_steal_ranges(f, n, p, g)
    else
        run_static_ranges(f, n, p)
    end
    return nothing
end

@noinline function run_static_ranges(f::F, n::Int, p::Int) where {F}
    pool = in_threaded_region() ? nothing : acquire_pool()
    if pool === nothing
        f(1, n)
        return nothing
    end
    skip = caller_worker(pool)
    st = pool.ranges
    reset_ranges!(st)
    q, r = divrem(n, p)
    prepare_job_closure(StaticRanges{F})
    job = StaticRanges(f, q, r, p, skip, range_deadline(p), st, pool)
    # `dispatch!`'s exception handler releases the pool
    dispatch!(pool, job, 1 <= skip <= p ? p : p + 1, true, skip)
    return nothing
end

@noinline function run_steal_ranges(f::F, n::Int, p::Int, grain::Int) where {F}
    pool = in_threaded_region() ? nothing : acquire_pool()
    if pool === nothing
        f(1, n)
        return nothing
    end
    skip = caller_worker(pool)
    st = pool.ranges
    reset_ranges!(st)
    q, r = divrem(n, p)
    init_blocks!(st.blocks, q, r, p)
    prepare_job_closure(StealRanges{F})
    job = StealRanges(f, st.blocks, q, r, p, skip, grain, range_deadline(p), st, pool)
    dispatch!(pool, job, 1 <= skip <= p ? p : p + 1, true, skip)
    return nothing
end
