@testset "concurrent chunks run on distinct threads" begin
    # A chunk can run on the calling task when its worker did not pick it up in time,
    # but chunks that run at the same time never share a thread.
    tids = zeros(Int, NW)
    t0 = zeros(UInt64, NW)
    t1 = zeros(UInt64, NW)
    tforeach(1:NW; scheduler = team) do c
        tids[c] = Threads.threadid()
        t0[c] = time_ns()
        while time_ns() - t0[c] < 1_000_000  # 1 ms
        end
        t1[c] = time_ns()
    end
    overlap(a, b) = t0[a] < t1[b] && t0[b] < t1[a]
    @test all(!(tids[a] == tids[b] && overlap(a, b)) for a in 1:NW, b in 1:NW if a != b)
end

@testset "workers run chunks" begin
    # Which task runs a chunk isn't guaranteed: the caller takes over the chunks of workers
    # that have not woken up yet, also on a loaded machine where that is all of them. So the
    # caller's first chunk waits until a worker has run a chunk; the caller can take over
    # unclaimed chunks only after that. (Nothing else holds the team here, otherwise the
    # loop would run on spawned tasks.)
    if NW > 1
        for sched in (team, GPS(chunksize = 1))
            caller = current_task()
            on_worker = Threads.Atomic{Bool}(false)
            waited = Threads.Atomic{Bool}(false)
            tforeach(1:(4NW); scheduler = sched) do i
                if current_task() === caller
                    # only once: the caller also runs the chunks it takes over
                    Threads.atomic_xchg!(waited, true) && return
                    t = time()
                    while !on_worker[] && time() - t < 60
                        yield()
                    end
                elseif current_task().code isa ThreadTeams.WorkerCode
                    on_worker[] = true
                end
            end
            @test on_worker[]
        end
    end
end

@testset "concurrent tasks" begin
    ntasks = 8
    results = zeros(Int, ntasks)
    tasks = map(1:ntasks) do t
        Threads.@spawn begin
            ok = 0
            for _ in 1:2000
                x = zeros(Int, 37 + t)
                tforeach(i -> (x[i] = i), eachindex(x); scheduler = team)
                s = tmapreduce(i -> x[i], +, eachindex(x); scheduler = team)
                ok += (x == 1:(37 + t)) && s == sum(1:(37 + t))
            end
            ok
        end
    end
    results = fetch.(tasks)
    @test all(==(2000), results)
    # from the interactive thread pool, if there is one
    if Threads.nthreads(:interactive) > 0
        r = fetch(Threads.@spawn :interactive psum(collect(1.0:100.0)))
        @test r == 5050.0
    end
end

@testset "loops while another task runs one" begin
    # A loop started while another task holds the pool runs on spawned tasks
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
        ct = current_task()
        tasks = Vector{Task}(undef, NW)
        tforeach(i -> (tasks[i] = current_task()), 1:NW; scheduler = team)
        @test tasks[1] === ct
        @test all(t -> t !== ct && !(t.code isa ThreadTeams.WorkerCode), tasks[2:end])
        # loops nested in the chunks run serially, as on the pool
        inner = zeros(Int, NW)
        same_task = fill(true, NW)
        tforeach(1:NW; scheduler = team) do i
            inner[i] = team_nchunks(1:100)
            outer = current_task()
            tforeach(1:100; scheduler = team) do j
                same_task[i] &= current_task() === outer
            end
        end
        @test all(==(1), inner)
        @test all(same_task)
        # exceptions: the one of the lowest failing chunk
        err = try
            tforeach(i -> i >= 2 && error("chunk $i"), 1:NW; scheduler = team)
            nothing
        catch e
            e
        end
        @test err == ErrorException("chunk 2")
        @test tmapreduce(identity, +, 1:100; scheduler = team) == 5050
        @test treduce(+, 1:100; scheduler = team) == 5050
        notify(release)
        wait(holder)
    end
end

@testset "blocked worker thread" begin
    # A task that does not yield occupies a worker's thread; the caller runs that
    # worker's chunk instead of waiting for the task to finish.
    if NW > 2
        add_one!(zeros(64))
        stop = Threads.Atomic{Bool}(false)
        blocker = Threads.@spawn begin
            t = time_ns()
            while !stop[] && time_ns() - t < 2_000_000_000
            end
        end
        while !istaskstarted(blocker)
            yield()
        end
        x = zeros(1000)
        elapsed = @elapsed for _ in 1:100
            add_one!(x)
        end
        stop[] = true
        wait(blocker)
        @test all(==(100.0), x)
        @test elapsed < 1.0
    end
end
