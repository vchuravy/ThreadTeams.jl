@testset "parallel phase" begin
    x = ThreadTeams.parallel_phase() do
        x = zeros(64)
        for _ in 1:5
            sleep(0.002)   # longer than the spin time
            add_one!(x)
        end
        if NW > 1
            # Idle workers keep spinning, except the one on our thread (it sleeps when we
            # become runnable again) and, rarely, one whose thread has another task to run
            # (`other_work`): pass if every other worker is awake in one of five tries
            pool = @atomic ThreadTeams.POOL.pool
            awake() = all(eachindex(pool.workers)) do k
                k == get(pool.tid2worker, Threads.threadid(), 0) ||
                    !(@atomic pool.sleepflags[k].sleeping)
            end
            @test any(1:5) do _
                sleep(0.002)
                awake()
            end
        end
        ThreadTeams.parallel_phase(() -> add_one!(x))
    end
    @test all(==(6.0), x)
    @test ThreadTeams.PHASE[] == 0
    @test_throws ErrorException ThreadTeams.parallel_phase(() -> error("in phase"))
    @test ThreadTeams.PHASE[] == 0
end
