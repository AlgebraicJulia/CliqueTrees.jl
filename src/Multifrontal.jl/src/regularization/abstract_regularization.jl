"""
    AbstractRegularization

A dynamic regularization strategy. It can be passed to [`cholesky!`](@ref)
or [`ldlt!`](@ref) in order to compute a *modified* Cholesky (or LDLt) factorization.

### Modified Factorizations

If a matrix ``A`` is ill-conditioned (or even indefinite!), it is sometimes helpful
to perturb it by a small matrix ``E`` before computing a Cholesky factorization.

```math
    P (A + E) P^\\mathsf{T} = L L^\\mathsf{T}.
```

The diagonal matrix ``E`` can be chosen statically (before the factorization) dynamically
(during the factorization). Each `AbstractRegularization` object is an algorithm for
finding ``E``.

"""
abstract type AbstractRegularization end
