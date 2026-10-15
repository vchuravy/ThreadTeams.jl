# Running OhMyThreads.jl's parallel functions on the pool.
#
# OhMyThreads' parallel functions all go through `Implementation._tmapreduce` (`tforeach`,
# `tmap!`, `tcollect` and `@tasks` too) and `Implementation._tmap`, dispatching on the
# scheduler. These are internal functions of OhMyThreads, hence the tight compat bound.
# The schedulers report no chunking arguments to OhMyThreads (the default), so OhMyThreads
# leaves the chunking to us (`nchunks`, then ChunkSplitters.jl for the index ranges): it
# only runs an input of at most one element serially itself.

using OhMyThreads.Implementation: Implementation, promise_task_local,
    check_all_have_same_indices, BoxedVariableError, append!!
using ChunkSplitters: ChunkSplitters, index_chunks

OhMyThreads.Schedulers.from_symbol(::Val{:team}) = TeamScheduler
OhMyThreads.Schedulers.from_symbol(::Val{:greedyteam}) = GreedyTeamScheduler

function Base.show(io::IO, ::MIME"text/plain", ::TeamScheduler)
    return print(io, "TeamScheduler\n└ Chunking: one chunk per thread")
end
function Base.show(io::IO, ::MIME"text/plain", g::GreedyTeamScheduler)
    return print(
        io, "GreedyTeamScheduler\n└ Chunking: ",
        g.chunksize > 0 ? "fixed size ($(g.chunksize))" :
            g.nchunks > 0 ? "fixed count ($(g.nchunks))" : "fixed count (4 per thread)"
    )
end

const ChunksInput = Union{
    ChunkSplitters.Internals.AbstractChunks,
    ChunkSplitters.Internals.Enumerate,
}

# OhMyThreads' check for variables captured in a `Core.Box`, but computed from the type of
# the closure at compile time, so that it costs nothing (OhMyThreads' own
# `throw_if_boxed_captures` allocates about 100 bytes per call). Same error and same
# opt-out (`@allow_boxed_captures`). Like OhMyThreads, it looks into closures that the
# closure captures.
function boxed_fields!(out::Vector{Symbol}, @nospecialize(T), seen::Vector{Any})
    (isconcretetype(T) && !any(S -> S === T, seen)) || return out
    push!(seen, T)
    for i in 1:fieldcount(T)
        FT = fieldtype(T, i)
        if FT === Core.Box
            push!(out, fieldname(T, i))
        elseif FT !== Union{} && FT <: Function
            boxed_fields!(out, FT, seen)
        end
    end
    return out
end
@generated function boxed_captures(::Type{F}) where {F}
    return Expr(:tuple, map(QuoteNode, boxed_fields!(Symbol[], F, Any[]))...)
end
@noinline function boxed_capture_error(vars::Tuple)
    OhMyThreads.allowing_boxed_captures[] && return nothing
    throw(BoxedVariableError(Symbol[vars...]))
end
@inline throw_if_boxed_captures() = nothing
@inline function throw_if_boxed_captures(f, fs...)
    vars = boxed_captures(typeof(f))
    isempty(vars) || boxed_capture_error(vars)
    return throw_if_boxed_captures(fs...)
end

# Chunk `c` of `n` of arrays with equal indices: `mapreduce` over views, task index `c`
struct MapReduceArrays{F, Op, A, K}
    f::F
    op::Op
    arrs::A
    n::Int
    kwargs::K
end
function (j::MapReduceArrays)(c::Int)
    r = chunk_range(first(j.arrs), c, j.n)
    args = map(B -> view(B, r), j.arrs)
    return mapreduce(
        promise_task_local(j.f, c), promise_task_local(j.op), args...;
        j.kwargs...
    )
end

# Chunk `c` of `n` of `chunks`/`index_chunks` input: every element is one of the user's
# chunks, and its position is the task index (as with OhMyThreads' own schedulers)
struct MapReduceItems{F, Op, I, K}
    f::F
    op::Op
    items::I
    n::Int
    kwargs::K
end
function (j::MapReduceItems)(c::Int)
    if j.n == length(j.items) && j.kwargs isa NamedTuple{(:init,)}
        # One item per chunk (as for `chunks(...; n = nthreads())` inputs), with `init`
        # (as for `tforeach`): `mapreduce` over the item is `op(init, f(item))`. Written
        # out, this keeps the index arithmetic and `mapreduce` off the critical path of
        # every chunk.
        i = firstindex(j.items) + (c - 1)
        x = promise_task_local(j.f, i)(j.items[i])
        return promise_task_local(j.op)(j.kwargs.init, x)
    end
    r = chunk_range(j.items, c, j.n)
    return mapreduce(
        i -> promise_task_local(j.f, i)(j.items[i]),
        promise_task_local(j.op), r; j.kwargs...
    )
end

# `f::F` and `op::Op`: without the static parameters, Julia would not specialize on the
# closures (they are only passed on), and would box them on every call.
function Implementation._tmapreduce(
        f::F, op::Op, Arrs, ::Type{OutputType},
        sched::TeamSchedulers,
        mapreduce_kwargs
    )::OutputType where {F, Op, OutputType}
    throw_if_boxed_captures(f, op)
    A = first(Arrs)
    if A isa ChunksInput
        length(Arrs) == 1 ||
            throw(ArgumentError("chunks input can not be combined with other inputs"))
        job = MapReduceItems(f, op, A, nchunks(A, sched), mapreduce_kwargs)
    elseif A isa AbstractArray
        check_all_have_same_indices(Arrs)
        job = MapReduceArrays(f, op, Arrs, nchunks(A, sched), mapreduce_kwargs)
    else
        throw(
            ArgumentError(
                "$(nameof(typeof(sched))) supports arrays and " *
                    "`chunks`/`index_chunks` inputs, got $(typeof(A))"
            )
        )
    end
    job.n == 1 && return job(1)
    if OutputType === Nothing
        # `tforeach` (and `@tasks` without a reduction) has no results to combine: run the
        # chunks without the partials buffer of `run_fold` and its exception handler
        run_chunks(job, job.n, sched)
        return nothing
    end
    return run_fold(job, promise_task_local(op), job.n, sched)
end

# Chunk `c` of `n` of `tmap`: the vector of results of the chunk
struct MapArrays{F, A}
    f::F
    arrs::A
    n::Int
end
function (j::MapArrays)(c::Int)
    r = chunk_range(first(j.arrs), c, j.n)
    return map(promise_task_local(j.f, c), map(B -> view(B, r), j.arrs)...)
end

struct MapItems{F, I}
    f::F
    items::I
    n::Int
end
function (j::MapItems)(c::Int)
    r = chunk_range(j.items, c, j.n)
    return map(i -> promise_task_local(j.f, i)(j.items[i]), r)
end

function map_chunks(sched::TeamSchedulers, job::J) where {J}
    job.n == 1 && return job(1)
    return run_fold(job, append!!, job.n, sched)
end

function Implementation._tmap(
        sched::TeamSchedulers, f::F, A::AbstractArray,
        _Arrs::AbstractArray...
    ) where {F}
    throw_if_boxed_captures(f)
    Arrs = (A, _Arrs...)
    v = map_chunks(sched, MapArrays(f, Arrs, nchunks(A, sched)))
    return reshape(v, size(A)...)
end

function Implementation._tmap(
        sched::TeamSchedulers, f::F, A::ChunksInput,
        _Arrs::AbstractArray...
    ) where {F}
    isempty(_Arrs) ||
        throw(ArgumentError("chunks input can not be combined with other inputs"))
    throw_if_boxed_captures(f)
    return map_chunks(sched, MapItems(f, A, nchunks(A, sched)))
end
