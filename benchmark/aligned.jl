# Copies of benchmark arrays at a cache-line-aligned address. Julia places small arrays
# (under about 2 KB) in its pools at 16-byte alignment, so whether the 64 `Float64` of a
# benchmark share cache lines between neighbouring chunks differs from process to process,
# and with it the time per loop (20–70 ns on a Xeon). Linux and macOS only
# (`posix_memalign`); Julia frees the memory with `free` when the array is collected.
using FixedSizeArrays: FixedSizeArrays, FixedSizeArray

const ALIGNMENT = 64

function aligned_pointer(::Type{T}, n::Integer) where {T}
    isbitstype(T) || throw(ArgumentError("aligned arrays need an isbits element type, not $T"))
    p = Ref{Ptr{Cvoid}}(C_NULL)
    err = ccall(
        :posix_memalign, Cint, (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t),
        p, ALIGNMENT, max(n * sizeof(T), 1)
    )
    err == 0 || throw(OutOfMemoryError())
    return Ptr{T}(p[])
end

"""
    aligned(a::Array)
    aligned(FixedSizeArray, a::AbstractArray)

A copy of `a`, as an `Array` or a `FixedSizeArray` (with the default memory backend), whose
first element is at an address that is a multiple of `ALIGNMENT` bytes.
"""
function aligned(a::Array{T}) where {T}
    b = unsafe_wrap(Array, aligned_pointer(T, length(a)), size(a); own = true)
    return copyto!(b, a)
end
function aligned(::Type{FixedSizeArray}, a::AbstractArray{T}) where {T}
    p = aligned_pointer(T, length(a))
    mem = @static if isdefined(Base, :Memory)
        unsafe_wrap(Memory{T}, p, length(a); own = true)
    else
        unsafe_wrap(Vector{T}, p, length(a); own = true)
    end
    return copyto!(FixedSizeArrays.wrap(FixedSizeArray, mem, size(a)), a)
end
