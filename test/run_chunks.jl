# The building blocks of other packages' loops, called directly
using ThreadTeams: run_chunks, run_fold, StaticSchedule, GreedySchedule, runs_serially

@testset "run_chunks and run_fold" begin
    for sched in (StaticSchedule(), GreedySchedule(), StaticSchedule(busy = :serial), GreedySchedule(busy = :serial))
        ns = sched isa StaticSchedule ? (0:NW) : (0, 1, 2, NW, 4NW, 1000)
        for n in ns
            counts = zeros(Int, n)
            run_chunks(c -> (counts[c] += 1; nothing), n, sched)
            @test all(==(1), counts)
            n >= 1 && @test run_fold(c -> [c], vcat, n, sched) == 1:n
        end
        @test runs_serially(sched) == (NW == 1)
        @test_throws ArgumentError run_fold(identity, +, 0, sched)
    end
    # the default schedule is static: at most one chunk per worker
    counts = zeros(Int, NW)
    run_chunks(c -> (counts[c] += 1; nothing), NW)
    @test all(==(1), counts)
    @test run_fold(identity, +, NW) == sum(1:NW)
    @test_throws ArgumentError run_chunks(identity, NW + 1)
    @test_throws ArgumentError run_fold(identity, +, NW + 1, StaticSchedule())
    @test_throws ArgumentError StaticSchedule(busy = :spawn)
    # nested operations run serially, on the task of their chunk
    if NW > 1
        nested = fill(false, NW)
        same_task = fill(false, NW)
        run_chunks(NW) do c
            nested[c] = runs_serially(StaticSchedule()) && runs_serially(GreedySchedule())
            t = current_task()
            same_task[c] = run_fold(_ -> current_task() === t, &, 8, GreedySchedule())
            nothing
        end
        @test all(nested) && all(same_task)
    end
    # the exception of the lowest failing chunk
    @test_throws "chunk 2" run_chunks(c -> c >= 2 && error("chunk $c"), 3NW, GreedySchedule())
end
