# `busy = :serial`: a loop that can not have the team (another task holds it, or a
# `Threads.@threads` loop runs) runs serially on the calling task instead of on spawned
# tasks.

const SERIAL_SCHEDS = (PS(busy = :serial), GPS(busy = :serial), GPS(chunksize = 1, busy = :serial))

# Whether every chunk of loops with `sched` ran on the calling task, for `tforeach`,
# `tmapreduce` and the functions that run a region (`run_chunks`, `run_fold`) with more
# than one chunk; also checks the results
function all_on_caller(sched)
    ct = current_task()
    on_caller = Ref(true)
    x = zeros(Int, 100)
    tforeach(eachindex(x); scheduler = sched) do i
        x[i] = i
        on_caller[] &= current_task() === ct
    end
    s = tmapreduce(+, 1:100; scheduler = sched) do i
        on_caller[] &= current_task() === ct
        i
    end
    n = max(NW, 2)
    ThreadTeams.run_chunks(c -> (on_caller[] &= current_task() === ct; nothing), n, sched)
    f = ThreadTeams.run_fold(c -> (on_caller[] &= current_task() === ct; c), +, n, sched)
    return on_caller[] && x == 1:100 && s == 5050 && f == sum(1:n)
end

@testset "busy" begin
    @testset "construction" begin
        @test PS() === PS(busy = :tasks)
        @test PS(busy = :serial) !== PS()
        @test GPS(chunksize = 3, busy = :serial).chunksize == 3
        @test_throws ArgumentError PS(busy = :wait)
        @test_throws ArgumentError GPS(busy = :spawn)
        @test occursin("When busy: tasks", sprint(show, MIME"text/plain"(), PS()))
        @test occursin("When busy: serial", sprint(show, MIME"text/plain"(), PS(busy = :serial)))
        @test occursin("When busy: serial", sprint(show, MIME"text/plain"(), GPS(busy = :serial)))
        @test ThreadTeams.nchunks(1:1000, PS(busy = :serial)) == (NW == 1 ? 1 : NW)
    end

    @testset "while another task holds the team" begin
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
            for sched in SERIAL_SCHEDS
                @test ThreadTeams.nchunks(1:1000, sched) == 1
                @test all_on_caller(sched)
            end
            # the default still runs such loops on spawned tasks
            @test ThreadTeams.nchunks(1:1000, team) == NW
            @test !all_on_caller(team)
            notify(release)
            wait(holder)
        end
    end

    @testset "inside Threads.@threads" begin
        n = max(NW, 2)
        for sched in SERIAL_SCHEDS
            ok = fill(false, n)
            busy = fill(true, n)
            Threads.@threads for i in 1:n
                ok[i] = all_on_caller(sched)
                # no iteration takes the team (read from inside a loop of this one)
                b = Ref(false)
                tforeach(j -> (b[] |= ThreadTeams.pool_busy()), 1:2; scheduler = sched)
                busy[i] = b[]
            end
            @test all(ok)
            @test !any(busy)
        end
        # the default scheduler is not affected
        x = zeros(Int, n, 100)
        Threads.@threads for i in 1:n
            tforeach(j -> (x[i, j] = j), 1:100; scheduler = team)
        end
        @test all(x[i, :] == 1:100 for i in 1:n)
    end

    @testset "outside: the team as usual" begin
        x = zeros(100)
        tforeach(i -> (x[i] += 1.0), eachindex(x); scheduler = PS(busy = :serial))
        @test all(==(1.0), x)
        @test tmapreduce(i -> x[i], +, eachindex(x); scheduler = PS(busy = :serial)) == 100.0
    end
end
