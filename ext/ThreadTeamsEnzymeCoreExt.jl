# Enzyme rules for the parallel operations of ThreadTeams (an extension on EnzymeCore).
#
# Enzyme can not differentiate the team itself (atomics, long-lived tasks), so
# `run_chunks(f, n, sched)` and `run_fold(f, op, n, sched)` (`tforeach`, `tmapreduce`,
# `treduce`, `tmap!`, `@tasks` with a team scheduler) have rules that differentiate the
# chunk function `f(c)` and run the derivative chunks on the team, with the same scheduler.
# `tmap` and `tcollect` throw: their chunk results are vectors, and the reduction rule
# needs a number (or tuple of numbers) as the result.
#
# - forward mode: chunk `c` runs `autodiff(Forward, ...)` of `f(c)`;
# - reverse mode: the augmented primal runs every chunk's augmented forward pass in
#   parallel and keeps one tape per chunk; the reverse pass runs the chunks' reverse
#   passes in parallel.
#
# This is how Enzyme.jl differentiates `Threads.@threads`. Chunks that read the same memory
# add their adjoints to the same shadow concurrently; Enzyme makes these updates atomic
# when Julia has more than one thread. The derivative chunks call the chunk function
# through a reference (`call_chunk(r, c)`), so the adjoints of values it captures by value
# (a scalar coefficient: the closure arrives as `MixedDuplicated`, or as `Active` if it
# captures nothing else) accumulate in its shadow reference.
module ThreadTeamsEnzymeCoreExt

using ThreadTeams: ThreadTeams, TeamSchedulers, run_chunks, run_fold
# EnzymeCore defines the functions and types; Enzyme adds the methods when loaded
using EnzymeCore: EnzymeCore, EnzymeRules, Annotation, Const, Active, Duplicated,
    MixedDuplicated, BatchDuplicated, BatchMixedDuplicated, Forward,
    ForwardWithPrimal, ReverseSplitWithPrimal, ReverseSplitModified,
    ReverseSplitWidth, autodiff, autodiff_thunk, make_zero,
    set_runtime_activity, set_strong_zero
using EnzymeCore.EnzymeRules: FwdConfig, RevConfig, AugmentedReturn

# One chunk of the loop
@inline function call_chunk(r::Base.RefValue{F}, c::Int) where {F}
    r[](c)
    return nothing
end

# The chunk function as a reference, for `call_chunk`. A `MixedDuplicated` closure (active
# references and active or constant numbers) comes with its shadow in a `Ref`; an `Active`
# closure (only numbers) gets a zero shadow, returned as its adjoint afterwards.
as_ref(f::Const) = Const(Ref(f.val))
as_ref(f::Duplicated) = Duplicated(Ref(f.val), Ref(f.dval))
as_ref(f::MixedDuplicated) = Duplicated(Ref(f.val), f.dval)
as_ref(f::Active) = Duplicated(Ref(f.val), Ref(make_zero(f.val)))
as_ref(f::BatchDuplicated) = BatchDuplicated(Ref(f.val), map(Ref, f.dval))
as_ref(f::BatchMixedDuplicated) = BatchDuplicated(Ref(f.val), f.dval)

# The adjoint to return for the chunk function: only an `Active` closure has one
adjoint(f::Active, r::Duplicated) = r.dval[]
adjoint(f, r) = nothing

# Enzyme passes the reverse rule a different shadow `Ref` for a `MixedDuplicated` argument
# than the augmented primal got, and reads that one afterwards. The chunks' reverse passes
# must accumulate into it, so the reference from the tape is re-bound to it.
rebind(f::MixedDuplicated, r::Duplicated) = Duplicated(r.val, f.dval)
rebind(f::BatchMixedDuplicated, r::BatchDuplicated) = BatchDuplicated(r.val, f.dval)
rebind(f, r) = r

function forward_mode(config::FwdConfig)
    mode = set_runtime_activity(Forward, config)
    return set_strong_zero(mode, config)
end

# Reverse split mode for `call_chunk(r, c)`. Whatever the loop function captures may be
# overwritten between the forward and the reverse pass if the rule's argument may be.
function reverse_mode(config::RevConfig, overwritten::Bool, width::Int = EnzymeRules.width(config))
    mode = ReverseSplitModified(ReverseSplitWithPrimal, Val((false, overwritten, false)))
    mode = ReverseSplitWidth(mode, Val(width))
    mode = set_runtime_activity(mode, config)
    return set_strong_zero(mode, config)
end

#
# The chunks of a loop without results (`run_chunks`, and `run_fold` with `nothing` as the
# chunk result, e.g. `tforeach`): `r` is the chunk function (or the partials store) as a
# reference, in any activity
#

function forward_chunks(config::FwdConfig, r::Annotation, n::Int, sched::TeamSchedulers)
    mode = forward_mode(config)
    run_chunks(n, sched) do c
        autodiff(mode, Const(call_chunk), Const, r, Const(c))
        nothing
    end
    return nothing
end

# Augmented forward pass of the chunks; the tape is the reverse thunk and one tape per chunk
function augmented_chunks(
        config::RevConfig, r::Annotation, n::Int, sched::TeamSchedulers,
        overwritten::Bool, width::Int = EnzymeRules.width(config)
    )
    mode = reverse_mode(config, overwritten, width)
    fwd, rev = autodiff_thunk(mode, Const{typeof(call_chunk)}, Const, Core.Typeof(r), Const{Int})
    tapes = Vector{EnzymeRules.tape_type(typeof(fwd))}(undef, n)
    run_chunks(n, sched) do c
        @inbounds tapes[c] = fwd(Const(call_chunk), r, Const(c))[1]
        nothing
    end
    return rev, tapes
end

function reverse_chunks(r::Annotation, n::Int, sched::TeamSchedulers, rev, tapes)
    r isa Const && return nothing
    run_chunks(n, sched) do c
        rev(Const(call_chunk), r, Const(c), @inbounds tapes[c])
        nothing
    end
    return nothing
end

#
# run_chunks(f, n, sched)
#

function EnzymeRules.forward(
        config::FwdConfig, ::Const{typeof(run_chunks)},
        ::Type{RT}, f::Annotation, n::Const{Int},
        sched::Const{<:TeamSchedulers}
    ) where {RT}
    forward_chunks(config, as_ref(f), n.val, sched.val)
    return nothing
end

function EnzymeRules.augmented_primal(
        config::RevConfig, ::Const{typeof(run_chunks)},
        ::Type{RT}, f::Annotation, n::Const{Int},
        sched::Const{<:TeamSchedulers}
    ) where {RT}
    r = as_ref(f)
    rev, tapes = augmented_chunks(config, r, n.val, sched.val, EnzymeRules.overwritten(config)[2])
    return AugmentedReturn(nothing, nothing, (r, rev, tapes))
end

function EnzymeRules.reverse(
        config::RevConfig, ::Const{typeof(run_chunks)},
        ::Type{RT}, tape, f::Annotation, n::Const{Int},
        sched::Const{<:TeamSchedulers}
    ) where {RT}
    r = rebind(f, tape[1])
    rev, tapes = tape[2], tape[3]
    reverse_chunks(r, n.val, sched.val, rev, tapes)
    return (adjoint(f, r), nothing, nothing)
end

#
# run_fold(f, op, n, sched): chunk `c` stores `f(c)` into a buffer of partial results,
# combined in chunk order. Chunk results of type `Nothing` (`tforeach`, `tmap!`) are
# handled like `run_chunks`, in every mode. Other results need a constant `op`, width 1,
# and a constant or `Active` result (numbers, tuples of numbers).
#

# One chunk of a reduction: store the partial result
struct StorePartial{F, T}
    r::Base.RefValue{F}
    buf::Vector{T}
end
@inline function (s::StorePartial)(c::Int)
    @inbounds s.buf[c] = s.r[](c)
    return nothing
end

# Combine the partial results of a reduction in chunk order (as `run_fold` does)
function fold_all(op::Op, buf::Vector{T}) where {Op, T}
    acc = @inbounds buf[1]
    for c in 2:length(buf)
        acc = op(acc, @inbounds buf[c])
    end
    return acc
end

partials_type(f) = (T = Base.promote_op(f, Int); T === Union{} ? Any : T)

function check_fold(op::Annotation, width)
    op isa Const ||
        error("ThreadTeams: differentiating a reduction requires a constant operator")
    width == 1 ||
        error("ThreadTeams: batched differentiation of reductions is not supported")
    return nothing
end

# The partial results and their shadow, as the argument of `call_chunk`
function store_partials(r::Annotation, ::Type{T}, n::Int) where {T}
    buf = Vector{T}(undef, n)
    if r isa Const
        return Const(Ref(StorePartial(r.val, buf))), buf, nothing
    end
    dbuf = Vector{T}(undef, n)
    store = Duplicated(Ref(StorePartial(r.val, buf)), Ref(StorePartial(r.dval, dbuf)))
    return store, buf, dbuf
end

function EnzymeRules.forward(
        config::FwdConfig, ::Const{typeof(run_fold)},
        ::Type{RT}, f::Annotation, op::Annotation,
        n::Const{Int}, sched::Const{<:TeamSchedulers}
    ) where {RT}
    T = partials_type(f.val)
    if T === Nothing
        forward_chunks(config, as_ref(f), n.val, sched.val)
        return EnzymeRules.needs_shadow(config) ?
            (EnzymeRules.needs_primal(config) ? Duplicated(nothing, nothing) : nothing) :
            nothing
    end
    check_fold(op, EnzymeRules.width(config))
    store, buf, dbuf = store_partials(as_ref(f), T, n.val)
    forward_chunks(config, store, n.val, sched.val)
    if EnzymeRules.needs_shadow(config)
        if dbuf === nothing
            acc = fold_all(op.val, buf)
            dacc = make_zero(acc)
        else
            dacc, acc = autodiff(
                set_runtime_activity(ForwardWithPrimal, config),
                Const(fold_all), op, Duplicated(buf, dbuf)
            )
        end
        return EnzymeRules.needs_primal(config) ? Duplicated(acc, dacc) : dacc
    else
        acc = fold_all(op.val, buf)
        return EnzymeRules.needs_primal(config) ? acc : nothing
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig, ::Const{typeof(run_fold)},
        ::Type{RT}, f::Annotation, op::Annotation,
        n::Const{Int}, sched::Const{<:TeamSchedulers}
    ) where {RT}
    T = partials_type(f.val)
    overwritten = EnzymeRules.overwritten(config)[2]
    if T === Nothing
        r = as_ref(f)
        rev, tapes = augmented_chunks(config, r, n.val, sched.val, overwritten)
        return AugmentedReturn(nothing, nothing, (r, nothing, nothing, rev, tapes))
    end
    check_fold(op, EnzymeRules.width(config))
    RT <: Union{Const, Active} ||
        error("ThreadTeams: differentiating a reduction requires a constant or active result, got $RT")
    store, buf, dbuf = store_partials(as_ref(f), T, n.val)
    rev, tapes = augmented_chunks(config, store, n.val, sched.val, overwritten, 1)
    if dbuf !== nothing
        for c in 1:n.val
            @inbounds dbuf[c] = make_zero(buf[c])
        end
    end
    acc = EnzymeRules.needs_primal(config) ? fold_all(op.val, buf) : nothing
    return AugmentedReturn(acc, nothing, (store, buf, dbuf, rev, tapes))
end

function EnzymeRules.reverse(
        config::RevConfig, ::Const{typeof(run_fold)},
        dret, tape, f::Annotation, op::Annotation, n::Const{Int},
        sched::Const{<:TeamSchedulers}
    )
    store, buf, dbuf, rev, tapes = tape
    if buf === nothing
        # chunks without results
        r = rebind(f, store)
        reverse_chunks(r, n.val, sched.val, rev, tapes)
        return (adjoint(f, r), nothing, nothing, nothing)
    end
    if f isa MixedDuplicated
        # the partials store's shadow holds the chunk function's shadow reference
        store.dval[] = StorePartial(f.dval, dbuf)
    end
    if dret isa Active && dbuf !== nothing
        # adjoints of the partial results, seeded with the adjoint of the result
        mode = set_runtime_activity(ReverseSplitWithPrimal, config)
        fwd_c, rev_c = autodiff_thunk(
            mode, Const{typeof(fold_all)},
            Active{typeof(dret.val)}, Core.Typeof(op),
            Duplicated{typeof(buf)}
        )
        bufs = Duplicated(buf, dbuf)
        tape_c = fwd_c(Const(fold_all), op, bufs)[1]
        rev_c(Const(fold_all), op, bufs, dret.val, tape_c)
    end
    # adjoints of the chunks (also of what they wrote into shadow memory, with a constant
    # result)
    reverse_chunks(store, n.val, sched.val, rev, tapes)
    df = f isa Active ? store.dval[].r[] : nothing
    return (df, nothing, nothing, nothing)
end

end # module
