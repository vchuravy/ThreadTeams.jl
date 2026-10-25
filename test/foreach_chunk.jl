using ThreadTeams: foreach_chunk

# A chunk function of a concrete, non-parametric type, as a caller that runs many loops
# with one callable type would pass
struct CountChunks
    counts::Vector{Threads.Atomic{Int}}
end
(c::CountChunks)(i::Int) = (Threads.atomic_add!(c.counts[i], 1); nothing)

function chunk_counts(m, p)
    counts = [Threads.Atomic{Int}(0) for _ in 1:m]
    foreach_chunk(CountChunks(counts), m, p)
    return [c[] for c in counts]
end

# The tasks that ran the chunks
function chunk_tasks(m, p)
    tasks = Vector{Task}(undef, m)
    foreach_chunk(c -> (tasks[c] = current_task(); nothing), m, p)
    return tasks
end

struct AddOne
    x::Vector{Float64}
    m::Int
end
function (a::AddOne)(c::Int)
    n = length(a.x)
    for i in ((c - 1) * n ÷ a.m + 1):(c * n ÷ a.m)
        @inbounds a.x[i] += 1.0
    end
    return nothing
end
add_one_chunks!(x, m, p) = foreach_chunk(AddOne(x, m), m, p)

@testset "foreach_chunk" begin
    @testset "every chunk once" begin
        for p in 1:(NW + 1), m in unique((0, 1, 2, p, p + 1, 4p, 4p + 3, 1000))
            @test chunk_counts(m, p) == ones(Int, m)
        end
        @test chunk_counts(10, typemax(Int)) == ones(Int, 10)
        x = zeros(4356)
        for m in (NW, 4NW, 1000)
            add_one_chunks!(x, m, NW)
        end
        @test all(==(3.0), x)
        @test_throws ArgumentError foreach_chunk(identity, -1, 2)
        @test_throws ArgumentError foreach_chunk(identity, 4, 0)
        @test_throws ArgumentError foreach_chunk(identity, typemax(Int), 2)
    end

    @testset "participants" begin
        ct = current_task()
        # one chunk, or one thread: on the caller
        @test chunk_tasks(1, NW) == [ct]
        @test all(t -> t === ct, chunk_tasks(10, 1))
        # fewer chunks than threads: as many participants as chunks
        if NW > 2
            @test length(unique(chunk_tasks(2, NW))) <= 2
        end
    end

    @testset "serially when the team is busy" begin
        ct = current_task()
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
            @test all(t -> t === ct, chunk_tasks(NW, NW))
            @test all(t -> t === ct, chunk_tasks(4NW, NW))
            notify(release)
            wait(holder)
        end
        # nested in a team loop
        same = fill(true, NW)
        tforeach(1:NW; scheduler = team) do i
            outer = current_task()
            same[i] = all(t -> t === outer, chunk_tasks(4NW, NW)) && all(t -> t === outer, chunk_tasks(NW, NW))
        end
        @test all(same)
        # inside `Threads.@threads`
        n = max(NW, 2)
        ok = fill(false, n)
        Threads.@threads for i in 1:n
            outer = current_task()
            ok[i] = all(t -> t === outer, chunk_tasks(4NW, NW)) && all(t -> t === outer, chunk_tasks(NW, NW))
        end
        @test all(ok)
    end

    @testset "exceptions" begin
        for m in (NW, 4NW + 3)
            started = fill(false, m)
            done = fill(false, m)
            err = try
                foreach_chunk(m, NW) do c
                    started[c] = true
                    (c == 2 || c == m) && error("chunk $c")
                    c == 1 && sleep(0.02)
                    done[c] = true
                end
                nothing
            catch e
                e
            end
            # the lowest failing chunk, after every chunk that started has finished
            @test err == ErrorException("chunk $(min(2, m))")
            @test all(c -> !started[c] || done[c] || c == 2 || c == m, 1:m)
            @test chunk_counts(m, NW) == ones(Int, m)   # the team still works
        end
    end

    @testset "allocations" begin
        x = zeros(4356)
        for _ in 1:10, m in (NW, 4NW)
            add_one_chunks!(x, m, NW)
        end
        @test minimum(_ -> (@allocated add_one_chunks!(x, NW, NW)), 1:20) == 0
        @test minimum(_ -> (@allocated add_one_chunks!(x, 4NW, NW)), 1:20) == 0
    end

    @testset "precompile statement" begin
        # After `precompile(foreach_chunk, (S, Int, Int))`, the first loops with an `S`
        # compile none of the team's code (in a fresh process: this one has run loops)
        code = """
        using ThreadTeams
        struct S
            x::Vector{Float64}
        end
        (s::S)(c::Int) = (s.x[c] += 1.0; nothing)
        precompile(ThreadTeams.foreach_chunk, (S, Int, Int))
        x = zeros(64)
        println(stderr, "first loops")
        ThreadTeams.foreach_chunk(S(x), $NW, $NW)
        ThreadTeams.foreach_chunk(S(x), 64, $NW)
        println(stderr, "done ", sum(x) == 64 + $NW)
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads(:default)) --trace-compile=stderr -e $code`
        out = IOBuffer()
        ok = success(pipeline(cmd; stdout = out, stderr = out))
        text = String(take!(out))
        ok || @error "the process failed" text
        @test ok
        i = findfirst("first loops", text)
        # The body of the workers' opaque closure is compiled when the closure is created
        # (`precompile(Tuple{Tuple{...}, Int64, Int64})`, traced on Julia 1.10 only)
        compiled = i === nothing ? ["(missing)"] :
            filter(split(text[last(i):end], '\n')) do l
                occursin("ThreadTeams", l) && !startswith(l, "precompile(Tuple{Tuple{")
        end
        # Under coverage (`Pkg.test(coverage = true)`, as in CI) the package's code is not
        # loaded from its package image, so the first loops compile it
        coverage = Base.JLOptions().code_coverage != 0
        isempty(compiled) || coverage || @error "compiled in the first loops" compiled
        @test isempty(compiled) skip = coverage
        @test occursin(r"done.*true"s, text)
    end
end
