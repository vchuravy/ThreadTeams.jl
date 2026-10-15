# How many chunks a parallel operation gets, and the index range of a chunk.

# Number of elements of an iteration space
@inline plength(itr::AbstractUnitRange) = length(itr)
@inline plength(itr) = lastindex(itr) - firstindex(itr) + 1

"""
    ThreadTeams.nchunks(itr, scheduler = TeamScheduler()) -> Int

Number of chunks a parallel operation over `itr` would be split into right now: with the
[`TeamScheduler`](@ref) at most one per thread of the `:default` thread pool, with the
[`GreedyTeamScheduler`](@ref) as many as it asks for, and at most one per element. `1`
means the operation runs serially: with a single thread, and inside another parallel
operation on the pool. (Loops also run serially during precompilation, but that is
decided later.)
"""
@inline nchunks(itr, sched::TeamSchedulers = TeamScheduler()) =
    nchunks_for_length(plength(itr), sched)

# Only plain and relaxed loads here (see `pool_busy`). Precompilation is detected later, in
# `acquire_pool`.
@inline function nchunks_for_length(len::Int, ::TeamScheduler)
    len <= 1 && return 1
    nw = nworkers()
    nw == 1 && return 1
    pool_busy() && in_region() && return 1
    return min(nw, len)
end
@inline function nchunks_for_length(len::Int, g::GreedyTeamScheduler)
    len <= 1 && return 1
    nw = nworkers()
    nw == 1 && return 1
    pool_busy() && in_region() && return 1
    m = g.chunksize > 0 ? cld(len, g.chunksize) : g.nchunks > 0 ? g.nchunks : 4nw
    return min(m, len)
end

# The linear indices of chunk `c` of `n` contiguous chunks of `A`, split with
# ChunkSplitters.jl like OhMyThreads' schedulers (`index_chunks` is imported in
# schedulers.jl). Chunking the index range rather than `A` itself also covers `A`s that
# ChunkSplitters does not chunk, e.g. `enumerate(chunks(...))`. `n == 1` is the whole
# collection (also when empty).
@inline function chunk_range(A, c::Int, n::Int)
    r = firstindex(A):lastindex(A)
    n == 1 && return r
    # One element per chunk (e.g. `chunks(...; n = nthreads())` on a `TeamScheduler`):
    # skip ChunkSplitters' integer divisions on every chunk's critical path
    if n == length(r)
        i = first(r) + (c - 1)
        return i:i
    end
    return r[index_chunks(r; n)[c]]
end
