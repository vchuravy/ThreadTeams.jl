# Chunk 1 (on the caller) takes much longer than the spin time; the workers must not go to
# sleep while they wait for it. Reads the `sleeping` flags of the workers that took part
# from inside chunk 1, while the pool is busy: a worker that spun out after its own chunk
# would be asleep by then, while one that gives its thread to another task
# (`other_work`) or is still being woken up is not a bug (reading after the loop would
# also catch a worker that the OS held across the end of the loop). In a function, so that
# the second call does not compile anything inside the loop.
function sleeping_during_long_chunk1()
    sleeping = fill(false, NW - 1)
    tforeach(1:NW; scheduler = team) do c
        if c == 1
            t = time_ns()
            while time_ns() - t < 20 * ThreadTeams.SPIN_NS[]
            end
            pool = @atomic ThreadTeams.POOL.pool
            skip = @atomic pool.skip
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
        # A worker sleeping for a legitimate reason (another task on its thread: the test
        # harness, GC, libuv) is rare and does not repeat; a worker whose spin time counts
        # from the end of its own chunk is asleep in every run
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
        @test_throws ArgumentError ThreadTeams.set_spin_time!(-1)
    finally
        ThreadTeams.set_spin_time!(old)
    end
end
