using ThreadTeams: foreach_range, foreach_chunk, stolen_chunks

# The ranges `foreach_range` passes to `f`, sorted. Calls that run at the same time never
# share a thread and these do not yield, so one vector per thread needs no lock.
function record_ranges(n; kw...)
    rs = [Tuple{Int, Int}[] for _ in 1:Threads.maxthreadid()]
    foreach_range(n; kw...) do lo, hi
        push!(rs[Threads.threadid()], (lo, hi))
        return nothing
    end
    return sort!(reduce(vcat, rs))
end

# Whether sorted ranges are nonempty and cover `1:n`, each index once
function tiles(rs, n)
    pos = 0
    for (lo, hi) in rs
        (lo == pos + 1 && lo <= hi) || return false
        pos = hi
    end
    return pos == n
end

# The blocks of `1:n` split into `p`, as `Threads.@threads :static` splits it
blocks(n, p) = [((k - 1) * (n ÷ p) + min(k - 1, n % p) + 1, k * (n ÷ p) + min(k, n % p)) for k in 1:p]
block_of(lo, n, p) = findfirst(b -> b[1] <= lo <= b[2], blocks(n, p))

# Whether `foreach_range` called `f` once, with `(1, n)`, on the calling task
function serial_on_caller(n; kw...)
    ct = current_task()
    calls = Tuple{Int, Int, Task}[]
    lk = ReentrantLock()
    foreach_range(n; kw...) do lo, hi
        lock(() -> push!(calls, (lo, hi, current_task())), lk)
        return nothing
    end
    return calls == [(1, n, ct)]
end

# Run `f()` in a sticky task on thread `tid`
function on_thread(f, tid)
    t = Task(f)
    t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1)
    schedule(t)
    return fetch(t)
end

# Install victim orders between regions; returns the previous ones
function set_victim_orders!(pool, victims)
    old = pool.ranges.victims
    pool.ranges.victims = victims
    return old
end

# Spin until cond() or timeout, with safepoints so allocating threads can trigger GC.
function spin_until(cond, timeout)
    t = time()
    while !cond()
        time() - t > timeout && return false
        ccall(:jl_cpu_pause, Cvoid, ())
        GC.safepoint()
    end
    return true
end

const WTIDS = Threads.threadpooltids(:default)

function add_one_static!(x)
    foreach_range(length(x)) do lo, hi
        for i in lo:hi
            @inbounds x[i] += 1.0
        end
    end
    return x
end
function add_one_steal!(x)
    foreach_range(length(x); schedule = :steal) do lo, hi
        for i in lo:hi
            @inbounds x[i] += 1.0
        end
    end
    return x
end

# Record each block's thread and rescue count. Wait up to 0.2 s for all blocks to start,
# giving workers time to claim while the caller runs its own block.
function block_threads(n, p)
    tids = zeros(Int, p)
    started = Threads.Atomic{Int}(0)
    s0 = stolen_chunks()
    foreach_range(n; threads = p) do lo, hi
        tids[block_of(lo, n, p)] = Threads.threadid()
        Threads.atomic_add!(started, 1)
        spin_until(() -> started[] == p, 0.2)
        return nothing
    end
    return tids, stolen_chunks() - s0
end

# Maximum number of misplaced blocks over cold launches (workers asleep).
function cold_misplaced(p, trials)
    worst = 0
    for _ in 1:trials
        sleep(0.005)   # longer than the spin time: the workers sleep
        tids = zeros(Int, p)
        foreach_range(100p; threads = p) do lo, hi
            tids[block_of(lo, 100p, p)] = Threads.threadid()
        end
        worst = max(worst, count(k -> tids[k] != WTIDS[k], 1:p))
    end
    return worst
end

# Record `(first, last, thread, time)` of each call
function timed_calls!(calls, n, schedule)
    foreach_range(n; schedule) do lo, hi
        push!(calls[Threads.threadid()], (lo, hi, Threads.threadid(), time_ns()))
        return nothing
    end
    return calls
end

# Occupy the thread of worker `v` with a task that does not yield (its worker sleeps),
# run `f()`, then free the thread
function with_worker_held(f, v)
    started = Threads.Atomic{Bool}(false)
    release = Threads.Atomic{Bool}(false)
    hog = Task(() -> (started[] = true; spin_until(() -> release[], 30)))
    hog.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), hog, WTIDS[v] - 1)
    schedule(hog)
    try
        spin_until(() -> started[], 10) || error("the task on worker $v did not start")
        return f()
    finally
        release[] = true
        wait(hog)
    end
end

function check_placement(n, p, trials)
    me = Threads.threadid()
    misplaced = clean = 0
    for _ in 1:trials
        tids, stolen = block_threads(n, p)
        for k in 1:p
            tids[k] == WTIDS[k] || (stolen > 0 && tids[k] == me) || (misplaced += 1)
        end
        stolen == 0 && (clean += 1)
    end
    return misplaced, clean
end

@testset "foreach_range" begin
    foreach_range((lo, hi) -> nothing, 2NW)   # starts the team
    pool = @atomic ThreadTeams.POOL.pool
    @info "foreach_range: the main task runs on" Threads.threadid() Threads.threadpool() worker = pool === nothing ? 0 : ThreadTeams.caller_worker(pool)

    @testset "every index once" begin
        for threads in 1:(NW + 2),
                n in unique((0, 1, 2, threads - 1, threads, threads + 1, 7, 13, 97, 1000, 4099))
            n >= 0 || continue
            p = min(threads, n, NW)
            # static: one call per block
            rs = record_ranges(n; threads)
            @test tiles(rs, n)
            @test rs == (n == 0 ? [] : p == 1 ? [(1, n)] : blocks(n, p))
            for grain in (0, 1, 3, 4, n + 1)
                rs = record_ranges(n; threads, schedule = :steal, grain)
                @test tiles(rs, n)
                if p > 1
                    maxlen = grain == 0 ? cld(n ÷ p, 8) : grain
                    @test all(r -> r[2] - r[1] + 1 <= maxlen, rs)
                end
            end
        end
        # (the largest `n` of the steal mode, or of all modes on 32-bit systems)
        for n in (10^6, min(ThreadTeams.STEAL_MAX_N, typemax(Int) >> 2))
            @test tiles(record_ranges(n), n)
            @test tiles(record_ranges(n; schedule = :steal), n)
            @test tiles(record_ranges(n; schedule = :steal, grain = cld(n, 3)), n)
        end
        # Beyond the packed ranges of the steal mode (64-bit only): static mode
        n = typemax(Int) >> 2
        @test record_ranges(n) == (NW == 1 ? [(1, n)] : blocks(n, NW))
        if n > ThreadTeams.STEAL_MAX_N
            @test record_ranges(n; schedule = :steal, grain = 5) == record_ranges(n)
            @test record_ranges(ThreadTeams.STEAL_MAX_N + 1; schedule = :steal) ==
                record_ranges(ThreadTeams.STEAL_MAX_N + 1)
        end
        @test tiles(record_ranges(10^5; schedule = :steal, grain = 1), 10^5)
        # other integer types
        @test tiles(record_ranges(Int32(100); threads = UInt8(3), schedule = :steal, grain = Int16(7)), 100)
    end

    @testset "steal claims" begin
        # The claims of the steal mode on hand-made blocks: `1:100` in 4 blocks of 25,
        # grain 3 (`init_blocks!` takes 0-based bounds, as the words hold them)
        pool = @atomic ThreadTeams.POOL.pool
        if pool !== nothing && NW >= 4
            st = pool.ranges
            j = ThreadTeams.StealRanges(
                (lo, hi) -> nothing, st.blocks, 25, 0, 4, 0, 3, UInt64(0), st, pool
            )
            word(k) = ThreadTeams.unpack_range(@atomic st.blocks[k].word)
            ThreadTeams.init_blocks!(st.blocks, 25, 0, 4)
            @test word(1) == (false, 0, 25) && word(4) == (false, 75, 100)
            @test !ThreadTeams.block_started(j, 1)
            # the owner from the front, `started` with the first claim
            @test ThreadTeams.claim_own(j, 1) == (1, 3)
            @test ThreadTeams.block_started(j, 1) && word(1) == (true, 3, 25)
            # block 2 runs empty
            for i in 1:9
                ThreadTeams.claim_own(j, 2)
            end
            @test ThreadTeams.claim_own(j, 2) == (1, 0) && word(2) == (true, 50, 50)
            # participant 2 steals from block 1: a quarter of its 22 indices from the
            # back (6: 20:25), runs one grain (20:22) and republishes the rest (23:25)
            @test ThreadTeams.steal_from(j, 2, 1) == (20, 22)
            @test word(1) == (true, 3, 19)
            @test word(2) == (true, 22, 25)
            @test ThreadTeams.claim_own(j, 2) == (23, 25)
            # from an unstarted block, which stays unstarted
            @test ThreadTeams.steal_from(j, 2, 3) == (69, 71)   # 7 of 25: 69:75
            @test word(3) == (false, 50, 68) && !ThreadTeams.block_started(j, 3)
            @test word(2) == (true, 71, 75)
            # a single index left: taken whole
            @atomic st.blocks[4].word = ThreadTeams.pack_range(true, 99, 100)
            @atomic st.blocks[2].word = ThreadTeams.pack_range(true, 75, 75)
            @test ThreadTeams.steal_from(j, 2, 4) == (100, 100)
            @test word(4) == (true, 99, 99) && word(2) == (true, 75, 75)
            @test ThreadTeams.steal_from(j, 2, 4) == (1, 0)
            # the largest bounds survive the packing
            m = ThreadTeams.STEAL_MAX_N
            @test ThreadTeams.unpack_range(ThreadTeams.pack_range(true, m - 1, m)) == (true, m - 1, m)
            @test ThreadTeams.unpack_range(ThreadTeams.pack_range(false, m, m)) == (false, m, m)
        end
    end

    @testset "steal clocks" begin
        TT = ThreadTeams
        YES, SOON, NO = TT.STEAL_YES, TT.STEAL_SOON, TT.STEAL_NO
        A = UInt64(TT.RANGE_AWAKE_NS)
        t, d = UInt64(10^9), UInt64(10^9 + 10^6)
        # `owner_due` and `unstarted_verdict` with explicit clocks, on a block of its own.
        # Observations in sequence: no simultaneous observers.
        b = TT.RangeBlock()
        seen() = @atomic b.seen
        verdict(pending, asleep, now) = TT.unstarted_verdict(b, pending, asleep, now, d)
        # an awake owner with its command pending: soon, then due, from the first
        # observation, which later ones do not renew
        @test verdict(true, false, t) == SOON && seen() == t
        @test verdict(true, false, t + A - 1) == SOON && seen() == t
        @test verdict(true, false, t + A) == YES && seen() == t
        @test TT.owner_due(b, false, t + A, d) && !TT.owner_due(b, false, t + A - 1, d)
        # asleep: protected until the deadline; the clock survives the observation
        @test verdict(true, true, t + 2A) == NO && seen() == t
        @test verdict(true, false, t + 2A) == YES
        @test verdict(true, true, d) == YES
        b = TT.RangeBlock()
        @test verdict(true, true, t) == NO && seen() == 0
        @test !TT.owner_due(b, true, t, d) && seen() == 0
        # claimed (or no command, as the caller's block): protected until the deadline
        @test verdict(false, false, t) == NO && verdict(false, false, d - 1) == NO
        @test verdict(false, false, d) == YES && seen() == 0
        # an observer that read the clock before another's timestamp waits, rather than
        # wrapping around to due; later ones use the first timestamp
        @test verdict(true, false, t + 5) == SOON && seen() == t + 5
        @test verdict(true, false, t) == SOON && seen() == t + 5
        @test !TT.owner_due(b, false, t, d)
        @test verdict(true, false, t + 4 + A) == SOON
        @test verdict(true, false, t + 5 + A) == YES
        # a timestamp in the future is not due before the deadline
        @atomic b.seen = d - 1
        @test verdict(true, false, t) == SOON && !TT.owner_due(b, false, t, d)

        # A pool whose workers never run: its commands stay pending, its owners awake
        function idle_pool(nw)
            return TT.Pool(
                nothing, [TT.WorkerState() for _ in 1:nw], [Base.Event(true) for _ in 1:nw],
                [Task(() -> nothing) for _ in 1:nw], Any[nothing for _ in 1:nw],
                TT.pad_before(), false, TT.pad_after(), zeros(Int, Threads.maxthreadid()),
                [TT.SleepFlag() for _ in 1:nw], TT.ChunkCounter(), Any[nothing for _ in 1:nw],
                zeros(Int, nw), false, 0, TT.RangeState(nw)
            )
        end
        # `1:100` in blocks 1:50 and 51:100; the caller (`skip = 1`) has run block 1, and
        # worker 2 has chunk 2 pending
        function fixture(deadline)
            pool = idle_pool(2)
            st = pool.ranges
            calls = Tuple{Int, Int}[]
            j = TT.StealRanges(
                (lo, hi) -> (push!(calls, (lo, hi)); nothing), st.blocks, 50, 0, 2, 1, 10,
                deadline, st, pool
            )
            TT.init_blocks!(st.blocks, 50, 0, 2)
            @atomic st.blocks[1].word = TT.pack_range(true, 50, 50)
            base = TT.cmd_base(2, 1, false)
            @atomic pool.workers[2].cmd = base | 2
            return pool, j, base, calls
        end
        # thieves rescan while an owner may become due, then take its block
        pool, j, base, calls = fixture(typemax(UInt64))
        @test TT.steal_allowed(j, 2) == SOON
        @atomic j.blocks[2].seen = UInt64(0)   # the scan's first observation: soon
        TT.run_steal(j, 1)
        @test ((_, lo, hi) = TT.unpack_range(@atomic j.blocks[2].word); lo == hi)
        @test sum(c -> c[2] - c[1] + 1, calls) == 50 && minimum(first, calls) == 51
        # the dispatcher's rescue starts the block's shared clock...
        pool, j, base, calls = fixture(typemax(UInt64))
        slot = TT.Slot{typeof(j)}()
        slot[] = j
        TT.rescue_ranges(pool, j, slot, 2, 1, base, typeof(slot)[], false)
        @test (@atomic pool.workers[2].cmd) == 0 && (@atomic j.blocks[2].seen) != 0
        @test sum(c -> c[2] - c[1] + 1, calls) == 50
        # ...and waits for one a thief started, here as if in the future
        pool, j, base, calls = fixture(typemax(UInt64))
        slot = TT.Slot{typeof(j)}()
        slot[] = j
        t0 = time_ns()
        @atomic j.blocks[2].seen = t0 + 3_000_000
        TT.rescue_ranges(pool, j, slot, 2, 1, base, typeof(slot)[], false)
        @test time_ns() - t0 >= 3_000_000 && (@atomic pool.workers[2].cmd) == 0
        @test sum(c -> c[2] - c[1] + 1, calls) == 50

        # each region starts with fresh clocks
        pool = @atomic TT.POOL.pool
        if pool !== nothing
            st = pool.ranges
            for k in 1:NW
                @atomic st.blocks[k].seen = UInt64(1)
            end
            t0 = time_ns()
            foreach_range((lo, hi) -> nothing, 100NW; schedule = :steal)
            @test all(k -> (s = @atomic st.blocks[k].seen; s == 0 || s >= t0), 1:NW)
        end
    end

    @testset "nearest victims" begin
        # Thief 1 with local victim 2 and remote victims 3 and 4, one index each. The
        # remote callback refills block 2: after a remote steal (and its republished
        # rest) the thief goes back to its node before the next remote victim.
        pool = @atomic ThreadTeams.POOL.pool
        if pool !== nothing
            st = ThreadTeams.RangeState(4)
            calls = Int[]
            pack = ThreadTeams.pack_range
            j = ThreadTeams.StealRanges(
                (lo, hi) -> begin
                    push!(calls, lo)
                    lo == 201 && (@atomic st.blocks[2].word = pack(true, 100, 101))
                end, st.blocks, 1, 0, 4, 0, 10, UInt64(0), st, pool
            )
            for (k, (lo, hi)) in enumerate([(0, 0), (100, 100), (200, 201), (300, 301)])
                @atomic st.blocks[k].word = pack(true, lo, hi)
            end
            ThreadTeams.steal_nearest(j, 1, 1, Int32[2, 3, 4], 1)
            @test calls == [201, 101, 301]
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, -1)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, (typemax(Int) >> 2) + 1)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, typemax(UInt))
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 10; threads = 0)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 10; schedule = :dynamic)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 10; schedule = :steal, grain = -1)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 10; grain = 2)
        # also when the loop would run serially
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 0; schedule = :dynamic)
        @test_throws ArgumentError foreach_range((lo, hi) -> nothing, 1; threads = 1, grain = 1)
        @test serial_on_caller(10; threads = 1)
        @test serial_on_caller(1)
        @test serial_on_caller(10; threads = 1, schedule = :steal, grain = 1)
    end

    @testset "callers" begin
        # spawned tasks, on each thread pool
        pools = Threads.nthreads(:interactive) > 0 ? (:default, :interactive) : (:default,)
        for tp in pools, schedule in (:static, :steal), n in (NW, 1000)
            @test fetch(Threads.@spawn tp tiles(record_ranges(n; schedule), n))
        end
        # a caller on each worker's thread
        for s in 1:NW, schedule in (:static, :steal)
            @test on_thread(() -> tiles(record_ranges(1000; schedule), 1000), WTIDS[s])
        end
    end

    @testset "serially" begin
        for schedule in (:static, :steal)
            # nested in a range loop (every call, also in steal mode), a chunk loop and a
            # team loop
            calls = Threads.Atomic{Int}(0)
            bad = Threads.Atomic{Int}(0)
            for outer in (:static, :steal)
                foreach_range(4NW; schedule = outer, grain = outer === :steal ? 1 : 0) do lo, hi
                    Threads.atomic_add!(calls, 1)
                    serial_on_caller(100; schedule) || Threads.atomic_add!(bad, 1)
                end
            end
            @test calls[] == (NW == 1 ? 2 : NW + 4NW)
            foreach_chunk(4NW) do c
                serial_on_caller(100; schedule) || Threads.atomic_add!(bad, 1)
            end
            @test bad[] == 0
            same = fill(false, NW)
            tforeach(1:NW; scheduler = team) do i
                same[i] = serial_on_caller(100; schedule)
            end
            @test all(same)
            # inside `Threads.@threads`
            n = max(NW, 2)
            inside = fill(false, n)
            Threads.@threads for i in 1:n
                inside[i] = serial_on_caller(100; schedule)
            end
            @test all(inside)
        end
        # while another task holds the team
        if NW > 1
            started = Base.Event()
            release = Base.Event()
            holder = Threads.@spawn tforeach(1:NW; scheduler = team) do i
                if i == 1
                    notify(started)
                    wait(release)
                end
            end
            wait(started)
            @test serial_on_caller(1000)
            @test serial_on_caller(1000; schedule = :steal)
            @test serial_on_caller(1000; schedule = :steal, grain = 1)
            notify(release)
            wait(holder)
        end
    end

    @testset "placement" begin
        if NW > 1
            ThreadTeams.parallel_phase() do
                # The caller: the main task, which runs on a worker's thread or on the
                # interactive one
                misplaced, clean = check_placement(1000, NW, 50)
                @test misplaced == 0
                @info "placement from the main task: loops without takeovers" clean
                # A caller on the thread of worker `s` runs block `s` (always, its own
                # thread), the workers the others
                for s in 1:NW
                    misplaced, clean = on_thread(() -> check_placement(1000, NW, 20), WTIDS[s])
                    @test misplaced == 0
                end
                # Fewer blocks than workers: a caller on a worker beyond them owns none
                if NW > 2
                    misplaced, _ = on_thread(() -> check_placement(1000, NW - 1, 20), WTIDS[NW])
                    @test misplaced == 0
                    misplaced, _ = on_thread(() -> check_placement(1000, NW - 1, 20), WTIDS[1])
                    @test misplaced == 0
                end
                # The placement of `Threads.@threads :static`: each block on the thread
                # `@threads` runs it on, unless the caller took it over (then on the
                # caller's thread); all of them if nothing was taken over
                n = 10NW + 3
                tt = zeros(Int, n)
                Threads.@threads :static for i in 1:n
                    tt[i] = Threads.threadid()
                end
                @test tt == [WTIDS[block_of(i, n, NW)] for i in 1:n]
                me = Threads.threadid()
                tr = zeros(Int, n)
                for _ in 1:20
                    s0 = stolen_chunks()
                    foreach_range(n) do lo, hi
                        tr[lo:hi] .= Threads.threadid()
                    end
                    stolen = stolen_chunks() - s0
                    @test all(i -> tr[i] == tt[i] || (stolen > 0 && tr[i] == me), 1:n)
                    stolen == 0 && @test tr == tt
                end
            end
        end
    end

    @testset "rescue rule" begin
        # `rescue_due(asleep, seen, now, deadline)`: a worker awake with its work pending
        # since `seen` after `RANGE_AWAKE_NS`, any worker after the deadline
        due = ThreadTeams.rescue_due
        A = UInt64(ThreadTeams.RANGE_AWAKE_NS)
        t, d = UInt64(10^9), UInt64(10^9 + 10^6)
        @test !due(false, t, t + A - 1, d)
        @test due(false, t, t + A, d)
        @test !due(false, UInt64(0), t + A, d)    # not seen awake yet
        @test !due(true, UInt64(0), d - 1, d)     # asleep: only the deadline
        @test !due(true, t, d - 1, d)
        @test due(true, UInt64(0), d, d)
        @test due(false, UInt64(0), d, d)
        D = ThreadTeams.range_deadline_ns
        @test (D(2), D(3), D(4), D(5), D(64), D(65)) ==
            (500_000, 750_000, 750_000, 1_000_000, 1_750_000, 2_000_000)
    end

    @testset "range marker" begin
        # Clear the range marker before the next region, including after failure.
        if NW > 1
            st = (@atomic ThreadTeams.POOL.pool).ranges
            active() = @atomic :monotonic st.active
            seen = fill(false, 2)
            for schedule in (:static, :steal)
                foreach_range(2NW; schedule) do lo, hi
                    lo == 1 && (seen[1] = active())
                    return nothing
                end
                foreach_chunk(2, 2) do i
                    i == 1 && (seen[2] = active())
                    return nothing
                end
                @test seen == [true, false]
                @test !active()
            end
            @test_throws ErrorException foreach_range(2NW) do lo, hi
                lo == 1 && error("boom")
                return nothing
            end
            foreach_chunk(i -> (i == 1 && (seen[2] = active()); nothing), 2, 2)
            @test !seen[2]
            @test !active()
        end
    end

    @testset "cold launches" begin
        # Cold placement diagnostics: loaded hosts may legitimately rescue late workers.
        if NW > 1
            worst = cold_misplaced(NW, 10)
            worst_off = Threads.threadpool() === :default && NW > 2 ?
                on_thread(() -> cold_misplaced(NW - 1, 10), WTIDS[NW]) : worst
            @info "cold launches: most blocks not on their worker, of 10" main = worst off_pool = worst_off
        end
    end

    @testset "oversubscribed" begin
        # Oversubscription: three threads, including the interactive one, on one CPU.
        # Use a fresh Linux process pinned with taskset.
        cpu = nothing
        if Sys.islinux() && Sys.which("taskset") !== nothing && isfile("/proc/self/status")
            for line in eachline("/proc/self/status")
                m = match(r"^Cpus_allowed_list:\s*(\d+)", line)
                m === nothing || (cpu = parse(Int, m[1]))
            end
            cpu === nothing && @info "oversubscribed: skipped, no CPU found in /proc/self/status"
        end
        if cpu !== nothing
            code = """
            using ThreadTeams
            pool_ok = true
            hits = zeros(Int, 1000)
            for _ in 1:200, schedule in (:static, :steal)
                ThreadTeams.foreach_range(1000; schedule) do lo, hi
                    for i in lo:hi
                        hits[i] += 1
                    end
                end
            end
            pool = @atomic ThreadTeams.POOL.pool
            println("oversubscribed=", pool.oversubscribed)
            println("covered=", all(==(400), hits))
            println("stolen=", ThreadTeams.stolen_chunks())
            """
            cmd = `taskset -c $cpu $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=2,1 -e $code`
            out = IOBuffer()
            proc = run(pipeline(cmd; stdout = out, stderr = out); wait = false)
            done = timedwait(() -> process_exited(proc), 300) === :ok
            done || kill(proc)
            text = String(take!(out))
            @test done && success(proc)
            (done && success(proc)) || @error "the process failed" text
            @test occursin("oversubscribed=true", text)
            @test occursin("covered=true", text)
            m = match(r"stolen=(\d+)", text)
            # at most every participant of the 400 loops (2 each)
            @test m !== nothing && parse(Int, m[1]) <= 800
            @info "oversubscribed: blocks taken over by the caller, of 800" stolen = m === nothing ? nothing : m[1]
        end
    end

    @testset "late worker" begin
        # Hold a worker's thread: rescue and stealing must wait until the deadline.
        if NW > 1
            v = Threads.threadid() == WTIDS[1] ? 2 : 1
            for schedule in (:static, :steal)
                n = 64NW
                bv = blocks(n, NW)[v]
                calls = [Tuple{Int, Int, Int, UInt64}[] for _ in 1:Threads.maxthreadid()]
                timed_calls!(calls, n, schedule)   # compiled before the time counts
                foreach(empty!, calls)
                t0 = Ref(UInt64(0))
                s0 = stolen_chunks()
                with_worker_held(v) do
                    t0[] = time_ns()
                    timed_calls!(calls, n, schedule)
                end
                sorted = sort!(reduce(vcat, calls))
                @test tiles([(c[1], c[2]) for c in sorted], n)
                inv = filter(c -> bv[1] <= c[1] <= bv[2], sorted)
                @test !isempty(inv) && all(c -> c[3] != WTIDS[v], inv)
                # static: one call, by the caller
                schedule === :static &&
                    @test map(c -> c[3], inv) == [Threads.threadid()]
                @test stolen_chunks() - s0 >= 1
                # not before the deadline (the worker sleeps)
                # (`t0` is sampled before the deadline, so the bound holds exactly)
                @test minimum(c -> c[4], inv) >= t0[] + ThreadTeams.range_deadline_ns(NW)
            end
        end
    end

    @testset "stealing" begin
        if NW > 1
            ThreadTeams.parallel_phase() do
                # Block 1 waits for a thief to run its tail; index 2 would indicate a
                # front steal. Other blocks wait for block 1 to start. Rescue can serialize
                # participants on the caller, so retry when no thief overlaps the owner.
                overlaps = clean = 0
                for _ in 1:5
                    n = 4096NW
                    b1 = blocks(n, NW)[1]
                    owner = Threads.Atomic{Int}(0)    # the thread of block 1's first piece
                    thief = Threads.Atomic{Int}(0)    # first index of the first stolen piece
                    waited = Threads.Atomic{Int}(0)   # 1: done, -1: timed out
                    ranges = [Tuple{Int, Int}[] for _ in 1:Threads.maxthreadid()]
                    caller = current_task()
                    s0 = stolen_chunks()
                    foreach_range(n; schedule = :steal, grain = 1) do lo, hi
                        tid = Threads.threadid()
                        push!(ranges[tid], (lo, hi))
                        if lo == 1
                            owner[] = tid
                            waited[] = spin_until(() -> thief[] != 0, 2) ? 1 : -1
                        elseif hi <= b1[2]
                            owner[] != 0 && owner[] != tid &&
                                Threads.atomic_cas!(thief, 0, lo)
                        elseif current_task() !== caller
                            spin_until(() -> owner[] != 0, 2)
                        end
                        return nothing
                    end
                    @test tiles(sort!(reduce(vcat, ranges)), n)
                    stolen_chunks() == s0 && (clean += 1)
                    if waited[] == 1
                        overlaps += 1
                        @test thief[] > 2
                    end
                end
                @info "stealing: launches in which a thief overlapped the owner, without takeovers, of 5" overlaps clean
                # (fails only if no launch out of 5 ran a thief next to the blocked owner)
                @test overlaps > 0
                # With threads busy with other tasks (the workers there do not run): the
                # loops still cover their ranges
                stop = Threads.Atomic{Bool}(false)
                hogs = [Threads.@spawn :default spin_until(() -> stop[], 10) for _ in 1:(NW - 1)]
                try
                    for _ in 1:20, schedule in (:static, :steal)
                        @test tiles(record_ranges(10^4; schedule, grain = schedule === :steal ? 7 : 0), 10^4)
                    end
                finally
                    stop[] = true
                    foreach(wait, hogs)
                end
            end
        end
    end

    @testset "victim order" begin
        # Blocks 1:NW-1 hold their first call until the last block's owner, done with its
        # own block, steals: its first victim. Only the order matters, so rescues of other
        # blocks are fine (with more threads than CPUs every region has one); an attempt
        # counts if worker NW ran its own block and stole.
        function first_victim()
            n = 64NW
            bs = blocks(n, NW)
            started = Threads.Atomic{Int}(0)
            victim = Threads.Atomic{Int}(0)
            foreach_range(n; schedule = :steal, grain = 1) do lo, hi
                k = block_of(lo, n, NW)
                if Threads.threadid() == WTIDS[NW]
                    if k != NW
                        Threads.atomic_cas!(victim, 0, k)
                    elseif lo == bs[NW][1]
                        spin_until(() -> started[] == NW - 1, 2)
                    end
                elseif lo == bs[k][1]
                    Threads.atomic_add!(started, 1)
                    spin_until(() -> victim[] != 0, 2)
                end
                return nothing
            end
            return started[] == NW - 1 ? victim[] : 0
        end
        function first_victims()
            vs = Int[]
            for _ in 1:10
                v = first_victim()
                v != 0 && push!(vs, v)
                length(vs) == 3 && break
            end
            isempty(vs) && @info "victim order: the last worker never ran its block and stole"
            return vs
        end
        if NW >= 3
            ThreadTeams.parallel_phase() do
                # The last two workers on one node: the thief tries its neighbour first
                node = [k >= NW - 1 ? 2 : 1 for k in 1:NW]
                vo = ThreadTeams.victim_orders(node, node, [10 20; 20 10])
                pool = @atomic ThreadTeams.POOL.pool
                old = set_victim_orders!(pool, ThreadTeams.CYCLIC_VICTIMS)
                try
                    # Cyclic order takes block 1 first
                    vs = first_victims()
                    isempty(vs) ? (@test_skip false) : (@test all(==(1), vs))
                    set_victim_orders!(pool, vo)
                    vs = first_victims()
                    isempty(vs) ? (@test_skip false) : (@test all(==(NW - 1), vs))
                    for p in 2:NW
                        @test tiles(record_ranges(10^4; threads = p, schedule = :steal, grain = 7), 10^4)
                    end
                finally
                    set_victim_orders!(pool, old)
                end
            end
        end
    end

    @testset "exceptions" begin
        # A failing block: its exception, with the backtrace in a `ChunkFailedException`
        # below it, once every call that started has finished
        p = NW
        for schedule in (:static, :steal), bad in unique((1, 2, p, 1000))
            started = zeros(Int, 1000)
            done = zeros(Int, 1000)
            excs = try
                foreach_range(1000; schedule) do lo, hi
                    started[lo] = 1
                    lo <= bad <= hi && throw(DomainError(bad))
                    done[lo] = 1
                end
                nothing
            catch
                current_exceptions()
            end
            @test excs !== nothing && excs[end].exception == DomainError(bad)
            # (with one thread, serially: the exception as thrown)
            if NW > 1 && excs !== nothing && length(excs) >= 2
                cause = excs[end - 1].exception
                @test cause isa ThreadTeams.ChunkFailedException
                @test cause.exception == DomainError(bad)
            else
                @test NW == 1
            end
            @test count(started .!= done) == 1   # all but the failing call finished
            @test tiles(record_ranges(1000; schedule), 1000)   # the team still works
        end
        # Several failures: the one recorded first, here the one that throws while the other
        # waits
        if NW > 1
            st = (@atomic ThreadTeams.POOL.pool).ranges
            for schedule in (:static, :steal), (first, second) in ((1, NW), (NW, 1))
                n = 1000
                bs = blocks(n, NW)
                # Both calls run at the same time (unless one thread runs both, e.g. the
                # caller taking over: then try again), and the second throws once the
                # failure of the first is recorded
                overlaps = 0
                for _ in 1:20
                    waiting = Threads.Atomic{Bool}(false)
                    overlapped = Threads.Atomic{Bool}(false)
                    err = try
                        foreach_range(n; schedule) do lo, hi
                            if lo == bs[first][1]
                                spin_until(() -> waiting[], 2)
                                error("first")
                            elseif lo == bs[second][1]
                                waiting[] = true
                                overlapped[] = spin_until(() -> (@atomic st.err) !== nothing, 2)
                                error("second")
                            end
                        end
                        nothing
                    catch e
                        e
                    end
                    @test err isa ErrorException
                    if overlapped[]
                        overlaps += 1
                        @test err == ErrorException("first")
                        overlaps >= 3 && break
                    end
                end
                # (fails only if no attempt out of 20 ran the two calls at the same time)
                @test overlaps > 0
            end
        end
        # the state of a failed region does not leak into the next ones
        @test tiles(record_ranges(1000), 1000)
        @test tiles(record_ranges(1000; schedule = :steal), 1000)
        if NW > 1
            st = (@atomic ThreadTeams.POOL.pool).ranges
            @test (@atomic st.err) === nothing
            @test !(@atomic st.stop)
        end
    end

    @testset "interrupted while waiting for the workers" begin
        if NW > 1
            main = current_task()
            running = Threads.Atomic{Bool}(false)
            gate = Threads.Atomic{Bool}(false)
            returned = Threads.Atomic{Bool}(false)
            finished = Threads.Atomic{Bool}(false)
            # (`throws`: the running call fails after the interrupt; its failure is not
            # rethrown, and not kept)
            for mode in (:static, :steal), throws in (false, true)
                running[] = false
                gate[] = false
                returned[] = false
                finished[] = false
                # As in exceptions.jl: interrupt the caller while a worker runs `f`, again
                # while the caller waits for it, and only then let `f` finish
                observer = @async begin
                    t = time()
                    while !running[] && time() - t < 10
                        yield()
                    end
                    schedule(main, InterruptException(); error = true)
                    yield()
                    t = time()
                    while time() - t < 0.2
                        yield()
                    end
                    returned[] || schedule(main, InterruptException(); error = true)
                    yield()
                    t = time()
                    while time() - t < 0.2
                        yield()
                    end
                    gate[] = true
                end
                err = try
                    foreach_range(4NW; schedule = mode) do lo, hi
                        t = time()
                        if current_task() === main
                            while !running[] && time() - t < 10
                                GC.safepoint()
                            end
                        elseif !Threads.atomic_xchg!(running, true)
                            while !gate[] && time() - t < 10
                                GC.safepoint()
                            end
                            finished[] = true
                            throws && error("after the interrupt")
                        end
                    end
                    nothing
                catch e
                    e
                end
                returned[] = true
                # the loop returned only after the running call
                @test finished[]
                wait(observer)
                @test err isa InterruptException
                st = (@atomic ThreadTeams.POOL.pool).ranges
                @test (@atomic st.err) === nothing
                @test !(@atomic ThreadTeams.POOL.pool).failed
                @test tiles(record_ranges(1000; schedule = mode), 1000)
            end
        end
    end

    @testset "allocations" begin
        x = zeros(4356)
        for _ in 1:10
            add_one_static!(x)
            add_one_steal!(x)
        end
        @test all(==(20.0), x)
        @test minimum(_ -> (@allocated add_one_static!(x)), 1:20) == 0
        @test minimum(_ -> (@allocated add_one_steal!(x)), 1:20) == 0
    end

    @testset "precompile statement" begin
        # In a fresh process, precompiling the positional entry point leaves only the
        # keyword sorter to compile on the first keyword call.
        code = """
        using ThreadTeams
        struct S
            x::Vector{Float64}
        end
        (s::S)(lo::Int, hi::Int) = (s.x[lo:hi] .+= 1.0; nothing)
        precompile(ThreadTeams.foreach_range, (S, Int))
        x = zeros(64)
        println(stderr, "first loops")
        ThreadTeams.foreach_range(S(x), 64)
        ThreadTeams.foreach_range(S(x), 64; schedule = :steal)
        ThreadTeams.foreach_range(S(x), 64; threads = 2, schedule = :steal, grain = 3)
        println(stderr, "done ", x == fill(3.0, 64))
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads(:default)) --trace-compile=stderr -e $code`
        out = IOBuffer()
        ok = success(pipeline(cmd; stdout = out, stderr = out))
        text = String(take!(out))
        ok || @error "the process failed" text
        @test ok
        i = findfirst("first loops", text)
        compiled = i === nothing ? ["(missing)"] :
            filter(split(text[last(i):end], '\n')) do l
                occursin("ThreadTeams", l) &&
                !any(p -> startswith(l, p), ("precompile(Tuple{Tuple{", "precompile(Tuple{typeof(Core.kwcall)"))
        end
        coverage = Base.JLOptions().code_coverage != 0
        isempty(compiled) || coverage || @error "compiled in the first loops" compiled
        @test isempty(compiled) skip = coverage
        @test occursin(r"done.*true"s, text)
    end
end
