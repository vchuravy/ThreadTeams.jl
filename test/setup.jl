# Shared definitions, evaluated in the sandbox module of every test file
using ThreadTeams: nworkers
using OhMyThreads: OhMyThreads, WithTaskIndex
using ScopedValues: ScopedValues

const NW = nworkers()
const PS = TeamScheduler
const GPS = GreedyTeamScheduler
const team = PS()

# Count how often every element is visited
function visits(itr)
    counts = Dict{Any, Int}()
    lk = ReentrantLock()
    tforeach(itr; scheduler = team) do x
        lock(lk) do
            counts[x] = get(counts, x, 0) + 1
        end
    end
    return counts
end

function check_visits(itr)
    counts = visits(itr)
    expected = Dict{Any, Int}()
    for x in itr
        expected[x] = get(expected, x, 0) + 1
    end
    return counts == expected
end

# Allocation test helpers (functions, so that `@allocated` measures only the loop)
function add_one!(x)
    tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = team)
    return x
end
psum(x) = tmapreduce(i -> (@inbounds x[i]), +, eachindex(x); scheduler = team)
function add_one_tasks!(x)
    @tasks for i in eachindex(x)
        @set scheduler = team
        @inbounds x[i] += 1.0
    end
    return x
end
function add_one_greedy!(x)
    tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = GPS())
    return x
end
psum_greedy(x) = tmapreduce(i -> (@inbounds x[i]), +, eachindex(x); scheduler = GPS())

# World age: chunks run in the world of the caller
pw_world_g(::Integer) = 1
function pw_world_h()
    r = zeros(Int, 64)
    tforeach(i -> (r[i] = pw_world_g(i)), eachindex(r); scheduler = team)
    return r
end
