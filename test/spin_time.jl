# Chunk 1 (on the caller) takes much longer than the spin time; the workers must not sleep
# while they wait for it. The `sleeping` flags are read from inside chunk 1, while the pool
# is busy: a worker whose spin time counted from the end of its own chunk would be asleep
# by then, while one that gives its thread to another task (`other_work`) or is still
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
            # chunk per participant) takes the participants of those workers back
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
