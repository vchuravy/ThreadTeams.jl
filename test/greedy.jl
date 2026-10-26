@testset "greedy scheduler" begin
    @testset "every element once" begin
        for sched in (GPS(), GPS(chunksize = 1), GPS(chunksize = 7), GPS(nchunks = 3)),
                itr in (
                    1:0, 1:1, 1:2, 1:(NW + 1), 0:1000, CartesianIndices((7, 9)),
                    collect(1:3:300),
                )

            counts = Dict{Any, Int}()
            lk = ReentrantLock()
            tforeach(itr; scheduler = sched) do x
                lock(() -> counts[x] = get(counts, x, 0) + 1, lk)
            end
            @test counts == Dict(x => 1 for x in itr)
        end
    end

    @testset "reductions keep chunk order" begin
        # (the chunk order of `tmapreduce(string, *)` and `tmap` is checked in schedulers.jl)
        for sched in (GPS(), GPS(chunksize = 1), GPS(chunksize = 5))
            s, m = tmapreduce(
                i -> (i, i), (a, b) -> (a[1] + b[1], max(a[2], b[2])), 1:1000;
                scheduler = sched
            )
            @test (s, m) == (500500, 1000)
        end
        f = rand(10_000)
        @test all(_ -> psum_greedy(f) == psum_greedy(f), 1:100)
        @test psum_greedy(f) ≈ sum(f)
    end

    @testset "exceptions" begin
        for bad in (1, 50, 100), sched in (GPS(chunksize = 1), GPS(chunksize = 7))
            done = zeros(Int, 100)
            err = try
                tforeach(1:100; scheduler = sched) do i
                    i == bad && throw(DomainError(i))
                    i > bad && i % 10 == 0 && throw(DomainError(i))   # later failures
                    done[i] = 1
                end
                nothing
            catch e
                e
            end
            @test err == DomainError(bad)
            # everything a serial loop would have done before the error was done
            @test all(==(1), done[1:(bad - 1)])
        end
        # the pool still works afterwards
        @test psum_greedy(collect(1.0:100.0)) == 5050.0
        # and reports the failures of later loops, not those of earlier ones
        for bad in (1, 2, 60, 2)
            err = try
                tforeach(i -> i >= bad && throw(DomainError(i)), 1:100; scheduler = GPS(chunksize = 1))
                nothing
            catch e
                e
            end
            @test err == DomainError(bad)
        end
        if NW > 1
            pool = @atomic ThreadTeams.POOL.pool
            @test all(isnothing, pool.greedy_errs)
            # A region cut short after its workers finished but before it reported its
            # failure (an interrupt) leaves nothing for the next region to report
            pool.greedy_errs[1] = ThreadTeams.ChunkFailedException(DomainError(-1), nothing)
            pool.greedy_errchunks[1] = 1
            @atomic pool.failed = true
            wslots = ThreadTeams.worker_slot_vector(Nothing)
            ThreadTeams.clean_up_region(pool, 1, 0, wslots, false, false, true)
            @test all(isnothing, pool.greedy_errs)
            @test !(@atomic pool.failed)
            err = try
                tforeach(i -> i >= 4 && throw(DomainError(i)), 1:100; scheduler = GPS(chunksize = 1))
                nothing
            catch e
                e
            end
            @test err == DomainError(4)
        end
    end

    @testset "nested" begin
        m = zeros(Int, 20, 30)
        tforeach(1:20; scheduler = GPS()) do j
            tforeach(i -> (m[j, i] = i + 100j), 1:30; scheduler = GPS())
        end
        @test m == [i + 100j for j in 1:20, i in 1:30]
    end

    @testset "while another task runs a loop" begin
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
            @test tmapreduce(string, *, 1:100; scheduler = GPS(chunksize = 3)) == join(1:100)
            err = try
                tforeach(i -> i >= 50 && error("bad $i"), 1:100; scheduler = GPS(chunksize = 1))
            catch e
                e
            end
            @test err == ErrorException("bad 50")
            notify(release)
            wait(holder)
        end
    end
end
