# Chunk 1 (on the caller) takes much longer than the spin time; the workers must not sleep
# while they wait for it. The `sleeping` flags are read from inside chunk 1, while the pool
# is busy: a worker whose spin time counted from the end of its own chunk would be asleep
# by then, while one that yields to a sticky task (`local_work`) or is still
# being woken is not a bug. (Read after the loop, the flags would also catch a worker that
# the OS held across the end of the loop.) In a function, so that the second call compiles
# nothing inside the loop.
function sleeping_during_long_chunk1()
    sleeping = fill(false, NW - 1)
    tforeach(1:NW; scheduler = team) do c
        if c == 1
            t = time_ns()
            while time_ns() - t < 20 * ThreadTeams.SPIN_NS[]
            end
            pool = @atomic ThreadTeams.POOL.pool
            skip = get(pool.tid2worker, Threads.threadid(), 0)   # chunk 1 runs on the caller
            for c2 in 2:NW
                k = ThreadTeams.worker_for_chunk(c2, skip)
                sleeping[c2 - 1] = @atomic pool.sleepflags[k].sleeping
            end
        end
    end
    return sleeping
end

@testset "spin time counts from the end of the loop" begin
    if NW > 1
        sleeping_during_long_chunk1()   # compiles the loop body and the read
        # A worker asleep for a legitimate reason (another task on its thread: the test
        # harness, GC, libuv) is rare and does not repeat; one whose spin time counts from
        # the end of its own chunk is asleep in every run
        @test any(_ -> !any(sleeping_during_long_chunk1()), 1:5)
    end
end

# Queue a task while chunk 1 runs. One worker runs it and resumes spinning
# (`run_queued_task`); the others keep spinning. Chunk 1 outlasts the spin time, which
# must still count from the region's end, even for the worker that ran the task. Return
# whether the task ran, how many workers were awake when it was queued, and how many
# of those are asleep at the end of chunk 1.
function sleep_after_queued_task()
    res = Ref((false, 0, 0))
    tforeach(1:NW; scheduler = team) do c
        if c == 1
            pool = @atomic ThreadTeams.POOL.pool
            skip = get(pool.tid2worker, Threads.threadid(), 0)
            ks = [ThreadTeams.worker_for_chunk(c2, skip) for c2 in 2:NW]
            asleep() = [@atomic(pool.sleepflags[k].sleeping) for k in ks]
            t0 = time_ns()
            while time_ns() - t0 < 50_000   # let the workers finish waking up
            end
            before = asleep()
            t = Threads.@spawn :default nothing
            t0 = time_ns()
            while !istaskdone(t) && time_ns() - t0 < 10^9
            end
            t0 = time_ns()
            while time_ns() - t0 < 3 * ThreadTeams.SPIN_NS[]
            end
            res[] = (istaskdone(t), count(.!before), count(asleep() .& .!before))
        end
    end
    return res[]
end

@testset "a queued task does not put the workers to sleep" begin
    if NW > 2 && ThreadTeams.HAS_MULTIQ
        sleep_after_queued_task()   # compiles
        runs = [sleep_after_queued_task() for _ in 1:20]
        @test all(r -> r[1], runs)
        # Require enough awake workers to make the sleep count meaningful.
        @test sum(r -> r[2], runs) >= 10 * (NW - 1)
        # In 1.2.0, every worker that saw the task slept: 4 of 7 on average, 52 of 63.
        @test sum(r -> r[3], runs) <= 5
    end
end

# Wake every worker, then run loops of `n < NW` chunks of `chunk_ns` each back to back for
# `dur_ns`, so that the pool is busy nearly all the time, and return whether the workers
# that no chunk went to sleep, read from inside chunk 1 of the last loop. (A worker that no
# loop uses is not woken by one; it spins after the pool starts or `parallel_phase`.)
function sleeping_unused_workers(n, chunk_ns, dur_ns)
    ThreadTeams.parallel_phase(() -> nothing)
    t0 = time_ns()
    while time_ns() - t0 < dur_ns
        tforeach(1:n; scheduler = team) do c
            t = time_ns()
            while time_ns() - t < chunk_ns
            end
        end
    end
    sleeping = Bool[]
    tforeach(1:n; scheduler = team) do c
        if c == 1
            pool = @atomic ThreadTeams.POOL.pool
            skip = get(pool.tid2worker, Threads.threadid(), 0)
            used = [ThreadTeams.worker_for_chunk(c2, skip) for c2 in 2:n]
            for k in 1:NW
                (k == skip || k in used) && continue
                push!(sleeping, @atomic pool.sleepflags[k].sleeping)
            end
        end
    end
    return sleeping
end

@testset "a worker the loops do not use sleeps" begin
    # Busy regions do not keep a worker spinning that has no chunk in them: with loops back
    # to back, it would never sleep (and with more threads than cores, `-t N` on N cores
    # with an interactive thread, it takes CPU time from the others)
    if NW > 2
        n = NW - 1   # the workers of chunks `n + 1:NW` (and of the caller's thread) get none
        sleeping_unused_workers(n, 1000, 1000)   # compiles
        @test any(1:5) do _
            s = sleeping_unused_workers(n, 10_000, 50 * ThreadTeams.SPIN_NS[])
            !isempty(s) && all(s)
        end
        x = zeros(100)
        add_one!(x)
        @test all(==(1.0), x)
    end
end

@testset "spin time" begin
    old = ThreadTeams.spin_time()
    @test old == 100.0e-6
    try
        ThreadTeams.set_spin_time!(0)
        @test ThreadTeams.spin_time() == 0
        x = zeros(100)
        for _ in 1:200
            add_one!(x)
            sleep(0.0005)   # workers sleep in between
        end
        @test all(==(200.0), x)
        # every shape of the wake-up tree
        for len in 2:(2NW + 1), _ in 1:5
            y = zeros(len)
            add_one!(y)
            @test all(==(1.0), y)
            @test psum(y) == len
            sleep(0.0002)
        end
        # greedy regions while the workers sleep
        for _ in 1:50
            counts = zeros(Int, 1000)
            tforeach(i -> (counts[i] += 1), 1:1000; scheduler = GPS(chunksize = 1))
            @test all(==(1), counts)
            sleep(0.0005)
        end
        if NW > 1
            # a greedy region that ends before a sleeping worker can wake up (one trivial
            # chunk per participant) runs the participants of those workers itself
            stolen = ThreadTeams.stolen_chunks()
            for _ in 1:50
                counts = zeros(Int, NW)
                tforeach(i -> (counts[i] += 1), 1:NW; scheduler = GPS(chunksize = 1))
                @test all(==(1), counts)
                sleep(0.0005)
            end
            @test ThreadTeams.stolen_chunks() > stolen
        end
        @test_throws ArgumentError ThreadTeams.set_spin_time!(-1)
    finally
        ThreadTeams.set_spin_time!(old)
    end
end

@testset "timers fire while workers spin" begin
    # Spinning workers on every thread run the event loop now and then. Without that, the
    # timers below would wait for the spin time (`set_spin_time!(2.0)` bounds a failure).
    if NW > 1
        old = ThreadTeams.spin_time()
        try
            ThreadTeams.set_spin_time!(2.0)
            # with a long spin time (#35)
            for _ in 1:5
                t = Threads.@spawn begin
                    add_one!(zeros(64))
                    t0 = time()
                    sleep(0.01)
                    time() - t0
                end
                ok = timedwait(() -> istaskdone(t), 10.0) === :ok
                @test ok
                ok && @test fetch(t) < 1.0
            end
            # while another task holds the team in a region whose finished workers keep
            # spinning until it ends (#73); in a process of its own, as without the event
            # loop it never ends
            code = """
            using ThreadTeams
            const NW = ThreadTeams.nworkers()
            for _ in 1:200
                gate = Base.Event()
                holder = Threads.@spawn tforeach(i -> i == 1 && wait(gate), 1:NW;
                    scheduler = TeamScheduler())
                while !(@atomic ThreadTeams.POOL.busy)
                    yield()
                end
                sleep(0.002)
                notify(gate)
                wait(holder)
            end
            """
            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads(:default)) -e $code`
            p = run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = false)
            exited = timedwait(() -> process_exited(p), 120.0) === :ok
            exited || (kill(p); wait(p))
            @test exited && success(p)
        finally
            ThreadTeams.set_spin_time!(old)
        end
    end
end
