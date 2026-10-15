# Chunk 1 (on the caller) takes much longer than the spin time; the workers must not go to
# sleep while they wait for it. Reads the `sleeping` flags of the workers that took part
# right after the loop (in a function, so that the second call does not compile anything
# between the loop and the read).
function sleeping_after_long_chunk1()
    tforeach(1:NW; scheduler = team) do c
        if c == 1
            t = time_ns()
            while time_ns() - t < 20 * ThreadTeams.SPIN_NS[]
            end
        end
    end
    pool = @atomic ThreadTeams.POOL.pool
    skip = get(pool.tid2worker, Threads.threadid(), 0)
    return [(@atomic pool.sleepflags[ThreadTeams.worker_for_chunk(c, skip)].sleeping) for c in 2:NW]
end

@testset "spin time counts from the end of the loop" begin
    if NW > 1
        sleeping_after_long_chunk1()   # compiles the loop body and the read
        @test !any(sleeping_after_long_chunk1())
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
