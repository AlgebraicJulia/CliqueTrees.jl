# This file is part of the Scotch distribution.
# It does not include the standard Scotch header because it is an
# adaptation of the MUMPS_HAMF4 routine which was released under
# the BSD3 license.
# Consequently, this file is distributed according to the terms of
# the BSD3 licence, see copyright notice below.
#
# Copyright 2004,2007,2010,2012,2018-2020,2023,2024 IPB, Universite de Bordeaux, INRIA & CNRS & others
#
#============================================================#
#
#     NAME       : hall_order_hf.c
#
#     AUTHOR     : Patrick AMESTOY & al.
#                  Francois PELLEGRINI
#
#     FUNCTION   : This module orders a halo graph or mesh
#                  structure using the block-oriented Halo
#                  Approximate (Multiple) Minimum Fill
#                  algorithm, with super-variable
#                  accounting HAMF4 (v20190828).
#
#     DATES      : # Version 3.4  : from : 15 may 2001
#                                   to   : 23 nov 2001
#                  # Version 4.0  : from : 10 jan 2003
#                                   to   : 29 aug 2007
#                  # Version 5.1  : from : 08 dec 2010
#                                   to   : 08 dec 2010
#                  # Version 6.0  : from : 08 mar 2012
#                                   to   : 30 apr 2018
#                  # Version 6.1  : from : 29 oct 2019
#                                   to   : 10 feb 2020
#                  # Version 7.0  : from : 19 jan 2023
#                                   to   : 06 aug 2024
#
#     NOTES      : # This module contains pieces of code
#                    that belong to other people; see
#                    below.
#
#============================================================#

module AMFLib

using Base: oneto
using FillArrays
using ..Utilities
using ..CliqueTrees

export amf, camf

const FLOAT = Float64
const AMF_SCOREMAX = 2.0^52   # largest rounded score; integers up to here are exact in a `FLOAT`

function amf(n::V, xadj::AbstractVector{E}, adjncy::AbstractVector{V}; kwargs...) where {V, E}
    vwght = Ones{V}(n)
    return amf(n, vwght, xadj, adjncy; kwargs...)
end

# The vertex weights may be any finite positive numbers. A vertex of weight `w`
# stands for a class of true twins of total weight `w`, and the ordering does
# not depend on the unit in which the weights are measured.
#
# If `exact` is true, then the initial score of each variable is computed from
# its exact fill-in, as for every later score, rather than from its degree.
function amf(n::V, vwght::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector{V}; exact::Bool = false) where {V, E}
    @assert n <= length(vwght)
    @inbounds nn = n + one(V); mm = xadj[nn]; m = mm - one(E)

    wt = FVector{FLOAT}(undef, n)
    wsum = amf_weights!(wt, vwght, n)
    wunit = one(FLOAT)
    nfine = amf_nfine(wsum, n)
    nbbuck = twice(nfine)
    iwlen = twice(m) + convert(E, 4n) + one(E)

    len = FVector{V}(undef, n)
    pe = FVector{E}(undef, nn)
    iw = FVector{V}(undef, iwlen)
    nv = FVector{V}(undef, n)
    elen = FVector{V}(undef, n)
    last = FVector{V}(undef, n)
    degree = FVector{FLOAT}(undef, n)
    wf = FVector{FLOAT}(undef, n)
    next = FVector{V}(undef, n)
    w = FVector{Int}(undef, n)
    head = FVector{V}(undef, nbbuck + two(V))

    perm, invp = amf_impl!(n, nfine, nbbuck, iwlen, pe,
        len, iw, nv, elen, last, degree, wf, next, w, head,
        wt, wunit, vwght, xadj, adjncy, exact)

    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

# Check that the weights `vwght[1:n]` are finite and positive, and store them in
# `wt[1:n]` in units of the smallest one. The ordering does not depend on the
# unit of the weights, and in these units the squared areas neither underflow
# nor overflow unless the weights span more than about 150 decades. Scores are
# then rounded to whole units. Returns the total weight.
function amf_weights!(wt::AbstractVector{F}, vwght::AbstractVector, n::Integer) where {F}
    wmin = convert(F, Inf); wsum = zero(F)

    @inbounds for i in oneto(n)
        wi = convert(F, vwght[i])

        if !(isfinite(wi) && wi > zero(F))
            throw(ArgumentError("vertex weights must be finite and positive, but weight $i is $(vwght[i])"))
        end

        wmin = min(wmin, wi)
    end

    @inbounds for i in oneto(n)
        wt[i] = convert(F, vwght[i]) / wmin
        wsum += wt[i]
    end

    return wsum
end

# Each of the first `nfine` rounded scores gets its own bucket, and larger
# scores share buckets; `nfine` only affects speed.
function amf_nfine(wsum::F, n::V) where {F, V}
    if iszero(n)
        return zero(V)
    else
        return convert(V, round(Int, clamp(wsum, convert(F, n), convert(F, max(16 * Int(n), 1 << 16)))))
    end
end

# The weighted fill-in of each variable: the sum of `weights[a] * weights[b]`
# over the missing edges {`a`, `b`} in its neighborhood. Computed with
# Wing-Huang updates in O(m√m) time.
function amf_fillin(n::V, weights::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V, E}
    @inbounds m = xadj[n + one(V)] - one(E)
    vfill = FVector{FLOAT}(undef, n)
    wdegree = FVector{FLOAT}(undef, n)
    marker = FVector{V}(undef, n)
    order = FVector{V}(undef, n)
    tail = FVector{E}(undef, n)
    adjtgt = FVector{V}(undef, m)
    number = FVector{V}(undef, n)

    @inbounds for v in oneto(n)
        number[v] = convert(V, xadj[v + one(V)] - xadj[v])
    end

    begptr = view(xadj, oneto(n))
    endptr = view(xadj, (one(V) + one(V)):(n + one(V)))
    CliqueTrees.fillin!(vfill, weights, wdegree, marker, order, tail, adjtgt, number, adjncy, begptr, endptr, n)
    return vfill
end

# unit weights stay a `Ones`, which `fillin!` handles without the weight lookups
function amf_fillweights(vwght::Ones, wt::AbstractVector, n::V) where {V}
    return Ones{FLOAT}(n)
end

function amf_fillweights(vwght::AbstractVector, wt::AbstractVector, n::V) where {V}
    return view(wt, oneto(n))
end

function amf_impl!(
        n::V,
        nfine::V,
        nbbuck::V,
        iwlen::E,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        nv::AbstractVector{V},
        elen::AbstractVector{V},
        last::AbstractVector{V},
        degree::AbstractVector{F},
        wf::AbstractVector{F},
        next::AbstractVector{V},
        w::AbstractVector,
        head::AbstractVector{V},
        wt::AbstractVector{F},
        wunit::F,
        vwght::AbstractVector,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        exact::Bool,
    ) where {V, E, F}
    @inbounds nn = n + one(V); mm = xadj[nn]; m = mm - one(E)

    nbelts = zero(V)
    pfree = mm

    @inbounds for p in oneto(m)
        iw[p] = adjncy[p]
    end

    @inbounds p = pe[begin] = xadj[begin]

    @inbounds for i in oneto(n)
        pp = pe[i + one(E)] = xadj[i + one(E)]
        nv[i] = one(V)
        len[i] = convert(V, pp - p)
        p = pp
    end

    if exact
        vfill = amf_fillin(n, amf_fillweights(vwght, wt, n), xadj, adjncy)
    else
        vfill = nothing
    end

    ncmpa = hamf_impl!(n, n, nbelts, nfine, nbbuck, iwlen, pe,
        pfree, len, iw, nv, elen, last, degree, wf, next, w, head, wt, wunit, vfill)

    @inbounds for j in oneto(n)
        head[j] = zero(V)
    end

    @inbounds for i in oneto(n)
        j = abs(elen[i])
        head[j] = i
    end

    k = zero(V)

    @inbounds for j in oneto(n)
        i = head[j]

        if ispositive(i)
            last[i] = k += one(V)
            elen[k] = i
        end
    end

    return elen, last
end

function camf(n::V, m::V, cptr::AbstractVector{E}, ctgt::AbstractVector{V}) where {V, E}
    vwght = Ones{V}(n)
    return camf(n, vwght, m, cptr, ctgt)
end

# Approximate minimum fill ordering of the graph G on the vertices 1:n whose
# edges are covered by the cliques 1:m, where the members of the clique c are
# ctgt[cptr[c]:cptr[c + 1] - 1]. G is never formed (see `camf_impl!`). The
# vertex weights are as for `amf`.
function camf(n::V, vwght::AbstractVector, m::V, cptr::AbstractVector{E}, ctgt::AbstractVector{V}) where {V, E}
    @assert n <= length(vwght)
    @assert m < length(cptr)
    @inbounds nnode = n + m; nnz = cptr[m + one(V)] - one(E)

    wt = FVector{FLOAT}(undef, nnode)
    wsum = amf_weights!(wt, vwght, n)
    nfine = amf_nfine(wsum, n)
    nbbuck = twice(nfine)
    norig = nnode                             # positions: the cliques, numbered first, and the vertices
    iwlen = twice(nnz) + convert(E, 4nnode) + one(E)

    len = FVector{V}(undef, nnode)
    pe = FVector{E}(undef, nnode + one(V))
    iw = FVector{V}(undef, iwlen)
    nv = FVector{V}(undef, nnode)
    elen = FVector{V}(undef, nnode)
    last = FVector{V}(undef, nnode)
    degree = FVector{FLOAT}(undef, nnode)
    wf = FVector{FLOAT}(undef, nnode)
    next = FVector{V}(undef, nnode)
    w = FVector{Int}(undef, nnode)
    head = FVector{V}(undef, max(nbbuck + two(V), norig))
    marker = FVector{V}(undef, n)

    perm, invp = camf_impl!(norig, n, m, nfine, nbbuck, iwlen, pe, len, iw, nv,
        elen, last, degree, wf, next, w, head, marker, wt, cptr, ctgt)

    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

# A clique cover is a quotient graph partway through elimination: join one
# node to the members of each clique, and eliminate the clique nodes. They
# are pairwise nonadjacent, so they create no fill among themselves, and what
# remains is G. So the cliques are passed to `hamf_impl!` as elements, which
# it treats as already eliminated, and the storage is Σ |C| rather than
# Σ |C|². The nodes are the vertices 1:n and the cliques n + 1:n + m, and the
# weighted external degree of each vertex in G is computed by `coverdegree!`.
#
# Returns the permutation and its inverse, as views of `elen` and `last`.
function camf_impl!(
        norig::V,
        n::V,
        m::V,
        nfine::V,
        nbbuck::V,
        iwlen::E,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        nv::AbstractVector{V},
        elen::AbstractVector{V},
        last::AbstractVector{V},
        degree::AbstractVector{F},
        wf::AbstractVector{F},
        next::AbstractVector{V},
        w::AbstractVector,
        head::AbstractVector{V},
        marker::AbstractVector{V},
        wt::AbstractVector{F},
        cptr::AbstractVector{E},
        ctgt::AbstractVector{V},
    ) where {V, E, F}
    nnode = n + m

    # the incidence graph of the cover
    pfree = coverincidence!(pe, len, iw, marker, n, m, cptr, ctgt)

    # every node is one position: a vertex, or a clique
    @inbounds for i in oneto(nnode)
        nv[i] = one(V)
    end

    # the weighted external degrees of the vertices
    coverdegree!(degree, marker, wt, pe, len, iw, n)

    @inbounds for v in oneto(n)
        elen[v] = zero(V)
    end

    # the element markers; an empty list has no head for `hamf_impl!` to
    # mark when it compresses `iw`, so empty cliques are given a null
    # pointer, as eliminated vertices are
    @inbounds for e in (n + one(V)):nnode
        elen[e] = -nnode - one(V)

        if iszero(len[e])
            pe[e] = zero(E)
        end
    end

    hamf_impl!(norig, nnode, m, nfine, nbbuck, iwlen, pe, pfree, len, iw, nv, elen,
        last, degree, wf, next, w, head, wt, one(F), nothing)

    # `abs(elen[i])` is the position of the node `i`: the elements
    # come first, and are left out of the permutation
    @inbounds for j in oneto(norig)
        head[j] = zero(V)
    end

    @inbounds for v in oneto(n)
        head[abs(elen[v])] = v
    end

    k = zero(V)

    @inbounds for j in oneto(norig)
        v = head[j]

        if ispositive(v)
            last[v] = k += one(V)
            elen[k] = v
        end
    end

    return view(elen, oneto(n)), view(last, oneto(n))
end

# round a score to a whole number of units, as an integer-valued float
function amf_round(rmf::F, n::Integer) where {F}
    if rmf < AMF_SCOREMAX
        q = floor(rmf + convert(F, 0.5))
    elseif rmf / convert(F, n) < AMF_SCOREMAX
        q = floor(rmf / convert(F, n) + convert(F, 0.5))
    else
        q = convert(F, AMF_SCOREMAX)
    end

    return max(q, one(F))
end

# the bucket of a rounded score: one bucket for each of the first `nfine`
# scores, then one for each `pas` scores, up to `nbbuck`. Only the buckets
# above `nfine` hold more than one score, and they are searched for the best
# one, so which variable is selected does not depend on `nfine`. (In HAMF4,
# the scores just above `nfine` shared bucket `nfine`, which was not searched.)
function amf_bucket(q::F, nfine::V, pas::V, nbbuck::V) where {F, V}
    if q > convert(F, nfine)
        return convert(V, min(div(convert(Int, q) - Int(nfine) - 1, Int(pas)) + Int(nfine) + 1, Int(nbbuck)))
    else
        return convert(V, q)
    end
end

#  -- translated by f2c (version 19970219).
#  -- hand-made adaptation (as of HAMF 20191101).
#  -- hand-made adaptation (as of HAMF 20200111).

#------------------------------------------------------------------------#
#   Version generated on October 29th 2019
#
#   This file includes various modifications of an original
#   LGPL/ CeCILL-C compatible
#   code implementing the AMD (Approximate Minimum Degree) ordering
#     Patrick Amestoy, Timothy A. Davis, and Iain S. Duff,
#      "An approximate minimum degree ordering algorithm,"
#      SIAM J. Matrix Analysis  vol 17, pages=886--905 (1996)
#      MUMPS_ANA_H is based on the original AMD code:
#
#      AMD, Copyright (c), 1996-2016, Timothy A. Davis,
#      Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.
#      Used in MUMPS under the BSD 3-clause license.
#      THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#      CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#      INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#      MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#      DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE
#      FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#      CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#      OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#      THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
#      TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#      OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#      OF SUCH DAMAGE.
#
#     MUMPS_HAMF4 is a major modification of AMD
#     since the metric used to select pivots in not anymore the
#     degree but an approximation of the fill-in.
#     In this approximation
#     all cliques of elements adjacent to the variable are deducted.
#     Written by Patrick Amestoy between 1999 and 2019.
#     and used by F. Pellegrini in SCOTCH since 2000.
#
#     Unique version to order both graph of variables and
#     graphs with both elements and variables.
#
#     Notations used:
#     Let us refer to as
#       Gv a graph with only variables
#       Ge a graph with both variables and elements
#
#     Let V be the set of nodes in the graph
#         V = Ve + V0
#             V0 = Set of variable nodes
#             Ve = Set of element nodes |Ve|=nbelts
#
#     The two sets are disjoint, and Ve can be empty.
#     If nbelts>0 then a bipartite graph between
#        V0 and Ve is provided on entry.
#
#     A graph of elements and variables is a bipartite
#     graph between the set of variables and the set of elements.
#     Thus variables are only adjacent to elements and
#     in the element list we only have variables.
#     Elements are "considered" as already eliminated and
#     are provided here only to describe the adjacency between
#     variables. Only variables in V0 need to be eliminated.
#
#     (CliqueTrees) The HAMF4 halo (the set V1 of variables that are
#     never ordered) has been removed.
#
#     (CliqueTrees) Weights. Each variable carries a weight wt(i), any
#     finite positive number; a variable of weight w stands for a class
#     of true twins of total weight w. The structural bookkeeping (nv,
#     the counts in w, positions in the ordering) counts vertices, and
#     a separate weight layer (wt, degree, wf) computes the scores. All
#     areas are squares of weights, so that the score does not depend
#     on a unit. After pivot me is eliminated, each variable i in Lme
#     gets the score
#
#       ( d (d + 2 W(Lme)) - sum_e x_e (2 W(Le) - x_e)
#                          - 2 wt(i) a - sum_j wt(j)^2 ) / wt(i)
#
#     where x_e = W(Le \ Lme) for each element e adjacent to i,
#     j runs over the variables adjacent to i outside Lme, a = sum_j wt(j),
#     and d = sum_e x_e + a bounds W(N(i) \ Lme). The numerator estimates
#     twice the weighted fill of eliminating i, counted as squares of
#     weights: all of N[i], minus the cliques Lme, Le and {j} it is
#     covered by. If d + W(Lme) exceeds the remaining weight r, the
#     numerator is the minimum of the above and
#     (r - wt(i))^2 - (W(Lme) - wt(i))^2. Scores are rounded to multiples
#     of wunit, the smallest weight. The initial score is the weighted
#     degree, or 2 F(i) / wt(i) for the exact fill-in F(i).
#
#     Comments relative to 64/32 bits integer representation:
#     This is [based on] a restrictive integer 64 bit variant:
#     It is assumed that IW array size can exceed 32-bit integer and
#     thus IWLEN is INTEGER(8) and PE is an INTEGER(8) array.
#     Graph size n must be smaller than 2x10^9 but number of
#     edges is a 64-bit integer.
#
#   Parameters
#      Input only, not modified on output :
#        N : number of nodes, N = size of V0 U Ve
#        NFINE : each of the first NFINE rounded scores has its own
#                bucket; larger scores share buckets
#        NBBUCK : number of buckets, at least NFINE
#                 advised value is 2*NFINE
#        IWLEN : Should be uint64_t # Patch FP: used Gnum as well #
#        WUNIT : scores are rounded to multiples of WUNIT
#      Input, undefined on output:
#        LEN(1..N)
#        IW(1..IWLEN)
#        WT(N) : vertex weights
#      Output only :
#        NCMPA
#        not ELEN(N)  # Patch FP: always initialized before call #
#        LAST(N)
#      Input/output
#        ELEN(N)  # Patch PA+FP #
#        PFREE
#        PE(N) : Should be uint64_t # Patch FP: used Gnum as well #
#        NV(N)
#        DEGREE(N) : input only in element mode (nbelts>0)
#      Internal workspace only
#        Min fill approximation one extra array of size NBBUCK+2
#         is also needed
#        NEXT(N)
#        W(N)
#        HEAD(0:NBBUCK+1)
#        WF(N)
#
#    Comments on the OUTPUT :
#    ------------------------
#    Let V0 be the variables of the initial graph (|V0|=n).
#    The assembly tree corresponds to the tree
#      of the supernodes (or supervariables). Each node of the
#      assembly tree is then composed of one principal variable
#      and a list of secondary variables. The list of
#      variable of a node (principal + secondary variables) then
#      describes the structure of the diagonal bloc of the supernode.
#    The elimination tree denotes the tree of all the variables(=node)
#      and is therefore of order n.
#
#    The arrays NV(N) and PE(N) give a description of the assembly tree.
#
#     1/ Description of array nv(N) (on OUTPUT)
#      nv(i)=0 i is a secondary variable
#      nv(i) >0 i is a principal variable, nv(i) holds the number of
#        vertices in its element, not counting its own block
#
#     2/ Description of array PE(N) (on OUTPUT)
#         Note that on
#         pe(i) = -(father of variable/node i) in the elimination tree:
#         If nv (i) .gt. 0, then i represents a node in the assembly
#         tree, and the parent of i is -pe (i), or zero if i is a root.
#         If nv (i) = 0, then (i,-pe (i)) represents an edge in a
#         subtree, the root of which is a node in the assembly tree.
#
#     3/ Example:
#        Let If be a root node father of Is in the assembly tree.
#        If is the principal variable of the node If and let If1, If2,
#        If3 be the secondary variables of node If.
#        Is is the principal variable of the node Is and let Is1, Is2 be
#        the secondary variables of node Is.
#
#        THEN:
#          NV(If1) = NV(If2) = NV(If3) = 0  (secondary variables)
#          NV(Is1) = NV(Is2) = 0  (secondary variables)
#          NV(If) > 0  (principal variable)
#          NV(Is) > 0  (principal variable)
#          PE(If) = 0 (root node)
#          PE(Is) = -If (If is the father of Is in the assembly tree)
#          PE(If1)=PE(If2)=PE(If3)= -If  ( If is the principal variable)
#          PE(Is1)=PE(Is2)= -Is  ( Is is the principal variable)
#
#
#     Given a representation of the nonzero pattern of a symmetric
#         matrix A, (excluding the diagonal) perform an approximate
#         minimum fill-in heuristic. Aggresive absorption is
#         used to tighten the bound on the degree.  This can result an
#         significant improvement in the quality of the ordering for
#         some matrices.
#  ---------------------------------------------------------------------
#   INPUT ARGUMENTS (unaltered):
#  ---------------------------------------------------------------------
#   n:    number of nodes in the graph
#         n = size of V0 U Ve
#
#   norig   number of positions in the ordering: the variables,
#           plus the elements of Ve, which are numbered first
#
#   nbelts number of elements (size of Ve)
#              =0 if Gv (graph of variables)
#              >0 if Ge
#           nbelts > 0 extends/changes the meaning of
#                     len/elen/degree on entry (see below)
#
#   wt:   the weight of each variable (finite and positive); the
#         entries of elements are unused on input. During execution,
#         wt (i) is the weight of the principal supervariable i, and
#         for an element e, wt (e) is the weight of Le \ Lme.
#
#   wunit: scores are rounded to multiples of wunit.
#
#   nfine, nbbuck: the rounded score q of a variable selects bucket q
#         if q .le. nfine, and nfine + (q - nfine) / pas, at most
#         nbbuck, otherwise, where pas = max(nfine / 8, 1). The
#         buckets above nfine are searched for their best score. These
#         parameters only affect the running time.
#
#   iwlen:        The length of iw (1..iwlen).  On input, the matrix is
#         stored in iw (1..pfree-1).  However, iw (1..iwlen) should be
#         slightly larger than what is required to hold the matrix, at
#         least iwlen .ge. pfree + n is recommended.  Otherwise,
#         excessive compressions will take place.
#         *** We do not recommend running this algorithm with ***
#         ***      iwlen .lt. pfree + n.                      ***
#         *** Better performance will be obtained if          ***
#         ***      iwlen .ge. pfree + n                       ***
#         *** or better yet                                   ***
#         ***      iwlen .gt. 1.2 * pfree                     ***
#         *** (where pfree is its value on input).            ***
#         The algorithm will not run at all if iwlen .lt. pfree-1.
#
#         Restriction: iwlen .ge. pfree-1
#  ---------------------------------------------------------------------
#   INPUT/OUPUT ARGUMENTS:
#  ---------------------------------------------------------------------
#   pe:   On input, pe (i) is the index in iw of the start of row i, or
#         zero if row i has no off-diagonal non-zeros.
#
#         During execution, it is used for both supervariables and
#         elements:
#
#         * Principal supervariable i:  index into iw of the
#                 description of supervariable i.  A supervariable
#                 represents one or more rows of the matrix
#                 with identical nonzero pattern.
#         * Non-principal supervariable i:  if i has been absorbed
#                 into another supervariable j, then pe (i) = -j.
#                 That is, j has the same pattern as i.
#                 Note that j might later be absorbed into another
#                 supervariable j2, in which case pe (i) is still -j,
#                 and pe (j) = -j2.
#         * Unabsorbed element e:  the index into iw of the description
#                 of element e, if e has not yet been absorbed by a
#                 subsequent element.  Element e is created when
#                 the supervariable of the same name is selected as
#                 the pivot.
#         * Absorbed element e:  if element e is absorbed into element
#                 e2, then pe (e) = -e2.  This occurs when the pattern
#                 of e (that is, Le) is found to be a subset of the
#                 pattern of e2 (that is, Le2).  If element e is "null"
#                 (it has no nonzeros outside its pivot block), then
#                 pe (e) = 0.
#
#         On output, pe holds the assembly tree/forest, which implicitly
#         represents a pivot order with identical fill-in as the actual
#         order (via a depth-first search of the tree).
#
#         On output:
#         If nv (i) .gt. 0, then i represents a node in the assembly
#         tree, and the parent of i is -pe (i), or zero if i is a root.
#         If nv (i) = 0, then (i,-pe (i)) represents an edge in a
#         subtree, the root of which is a node in the assembly tree.
#
#   pfree:        On input, the matrix is stored in iw (1..pfree-1) and
#         the rest of the array iw is free.
#         During execution, additional data is placed in iw, and pfree
#         is modified so that components  of iw from pfree are free.
#         On output, pfree is set equal to the size of iw that
#         would have been needed for no compressions to occur.  If
#         ncmpa is zero, then pfree (on output) is less than or equal
#         to iwlen, and the space iw (pfree+1 ... iwlen) was not used.
#         Otherwise, pfree (on output) is greater than iwlen, and all
#         the memory in iw was used.
#
#   nv:   On input, nv (i) = 1 for every variable i; for an element e,
#         nv (e) is its number of positions in the ordering.
#         During execution, abs (nv (i)) is equal to the number of
#         vertices that are represented by the principal supervariable i.
#         If i is a nonprincipal variable, then nv (i) = 0.  nv (i) .lt. 0
#         signifies that i is a principal variable in the pattern Lme of
#         the current pivot element me.  For an element e, nv (e) = |Le|,
#         the number of vertices in Le when e was created.
#  ---------------------------------------------------------------------
#   INPUT/MODIFIED (undefined on output):
#  ---------------------------------------------------------------------
#   len:  On input, len (i)
#                     if (nbelts==0) then (graph of variables)
#                       len(i) holds the number of entries in row i of
#                       the matrix, excluding the diagonal.
#                     else (graph of elements+variables)
#                       if i ∈ V0 then len(i) = nb of elements adjacent
#                           to i
#                       if i ∈ Ve then len(i) = nb of variables
#                           adjacent to i
#                     endif
#         The content of len (1..n) is undefined on output.
#
#   elen: defined on input only if nbelts>0
#         if e ∈ Ve then elen (e) = -N-1
#         if v ∈ V0 then elen (v) = 0
#
#   degree: defined on input only if nbelts>0
#         if v ∈ V0 then degree (v) = weighted external degree of v,
#                                    the sum of wt (j) over its neighbours
#
#   iw:   On input, iw (1..pfree-1) holds the description of each row i
#         in the matrix.  The matrix must be symmetric, and both upper
#         and lower triangular parts must be present.  The diagonal must
#         not be present.  Row i is held as follows:
#
#                 len (i):  the length of the row i data structure
#                 iw (pe (i) ... pe (i) + len (i) - 1):
#                         the list of column indices for nonzeros
#                         in row i (simple supervariables), excluding
#                         the diagonal.  All supervariables start with
#                         one row/column each (supervariable i is just
#                         row i).
#                 if len (i) is zero on input, then pe (i) is ignored
#                 on input.
#
#                 Note that the rows need not be in any particular
#                 order, and there may be empty space between the rows.
#
#         During execution, the supervariable i experiences fill-in.
#         This is represented by placing in i a list of the elements
#         that cause fill-in in supervariable i:
#
#                 len (i):  the length of supervariable i
#                 iw (pe (i) ... pe (i) + elen (i) - 1):
#                         the list of elements that contain i.  This
#                         list is kept short by removing absorbed
#                         elements.
#                 iw (pe (i) + elen (i) ... pe (i) + len (i) - 1):
#                         the list of supervariables in i.  This list
#                         is kept short by removing nonprincipal
#                         variables, and any entry j that is also
#                         contained in at least one of the elements
#                         (j in Le) in the list for i (e in row i).
#
#         When supervariable i is selected as pivot, we create an
#         element e of the same name (e=i):
#
#                 len (e):  the length of element e
#                 iw (pe (e) ... pe (e) + len (e) - 1):
#                         the list of supervariables in element e.
#
#         An element represents the fill-in that occurs when
#         supervariable i is selected as pivot (which represents the
#         selection of row i and all non-principal variables whose
#         principal variable is i). We use the term Le to denote the
#         set of all supervariables in element e. Absorbed
#         supervariables and elements are pruned from these lists when
#         computationally convenient.
#
#         CAUTION:  THE INPUT MATRIX IS OVERWRITTEN DURING COMPUTATION.
#         The contents of iw are undefined on output.
#
#  ---------------------------------------------------------------------
#   OUTPUT (need not be set on input):
#  ---------------------------------------------------------------------
#   elen: See the description of iw above.  At the start of execution,
#         elen (i) is set to zero.  During execution, elen (i) is the
#         number of elements in the list for supervariable i.  When e
#         becomes an element, elen (e) = -nel is set, where nel is the
#         current step of factorization.  elen (i) = 0 is done when i
#         becomes nonprincipal.
#
#         For variables, elen (i) .ge. 0 holds
#         For elements, elen (e) .lt. 0 holds.
#
#   last: In a degree list, last (i) is the supervariable preceding i,
#         or zero if i is the head of the list.  In a hash bucket,
#         last (i) is the hash key for i.  last (head (hash)) is also
#         used as the head of a hash bucket if head (hash) contains a
#         degree list (see head, below).
#
#   ncmpa:        The number of times iw was compressed.  If this is
#         excessive, then the execution took longer than what could
#         have been. To reduce ncmpa, try increasing iwlen to be 10%
#         or 20% larger than the value of pfree on input (or at least
#         iwlen .ge. pfree + n).  The fastest performance will be
#         obtained when ncmpa is returned as zero.  If iwlen is set to
#         the value returned by pfree on *output*, then no compressions
#         will occur.
#
#  ---------------------------------------------------------------------
#   LOCAL (not input or output - used only during execution):
#  ---------------------------------------------------------------------
#   degree:       If i is a supervariable, then degree (i) holds the
#         current approximation of the weighted external degree of row i
#         (an upper bound): the weight of the neighbours of i, not
#         counting i itself.  The bound is equal to the external degree
#         if elen (i) is less than or equal to two.
#         We also use the term "external degree" for elements e to refer
#         to |Le \ Lme|.  If e is an element, then degree (e) holds
#         the weight of Le (not including the diagonal part).
#
#   head: head is used for degree lists.  head (b) is the first
#         supervariable in the list of bucket b (see nfine above).  If
#         the list b is empty then head (b) = 0.
#
#         During supervariable detection head (hash) also serves as a
#         pointer to a hash bucket.
#         If head (hash) .gt. 0, there is a degree list of degree hash.
#                 The hash bucket head pointer is last (head (hash)).
#         If head (hash) = 0, then the degree list and hash bucket are
#                 both empty.
#         If head (hash) .lt. 0, then the degree list is empty, and
#                 -head (hash) is the head of the hash bucket.
#         After supervariable detection is complete, all hash buckets
#         are empty, and the (last (head (hash)) = 0) condition is
#         restored for the non-empty degree lists.
#   next: next (i) is the supervariable following i in a link list, or
#         zero if i is the last in the list.  Used for two kinds of
#         lists:  degree lists and hash buckets (a supervariable can be
#         in only one kind of list at a time).
#   w:    The flag array w determines the status of elements and
#         variables, and the external degree of elements.
#
#         for elements:
#            if w (e) = 0, then the element e is absorbed
#            if w (e) .ge. wflg, then w (e) - wflg is the size of
#                 the set |Le \ Lme|, in terms of vertices (the
#                 sum of abs (nv (i)) for each principal variable i that
#                 is both in the pattern of element e and NOT in the
#                 pattern of the current pivot element, me), and
#                 wt (e) is its weight.
#            if wflg .gt. w (e) .gt. 0, then e is not absorbed and has
#                 not yet been seen in the scan of the element lists in
#                 the computation of |Le\Lme| in loop 150 below.
#
#         for variables:
#            during supervariable detection, if w (j) .ne. wflg then j
#            is not in the pattern of variable i
#
#         The w array is initialized by setting w (i) = 1 for all i,
#         and by setting wflg = 2.  It is reinitialized if wflg becomes
#         too large (to ensure that wflg+n does not cause integer
#         overflow).
#
#   wf :  for a variable i, its rounded score; while i is in Lme, the
#         area to subtract in its next score. For an element e, the area
#         x_e (2 W(Le) - x_e) of its part outside Lme.
function hamf_impl!(
        norig::V,                  # number of positions: the variables, plus the elements of Ve
        n::V,                      # number of nodes (variables and elements)
        nbelts::V,                 # number of elements
        nfine::V,                  # each of the first `nfine` rounded scores has its own bucket
        nbbuck::V,                 # number of buckets
        iwlen::E,                  # length of array `iw`
        pe::AbstractVector{E},     # array of indices in `iw` of start of row `i`
        pfree::E,                  # useful size in `iw`
        len::AbstractVector{V},    # array of lengths of adjacency lists
        iw::AbstractVector{V},     # adjacency list array
        nv::AbstractVector{V},     # supervariable sizes (vertex counts); |Le| for an element
        elen::AbstractVector{V},   # array that holds the inverse permutation
        last::AbstractVector{V},   # array that holds the permutation
        degree::AbstractVector{F}, # weighted degree bound of a variable; weight of Le for an element
        wf::AbstractVector{F},     # score of a variable; area of an element
        next::AbstractVector{V},   # linked list structure
        w::AbstractVector{I},      # flag array
        head::AbstractVector{V},   # linked list structure
        wt::AbstractVector{F},     # weight of a supervariable; weight of Le \ Lme for an element
        wunit::F,                  # scores are rounded to multiples of `wunit`
        vfill::Union{Nothing, AbstractVector}, # exact initial fill-in (optional)
    ) where {I, V, E, F}
    @assert nbelts <= n
    @assert pfree <= iwlen <= length(iw)
    @assert nfine <= nbbuck
    @assert n < length(pe)
    @assert n <= length(len)
    @assert n <= length(nv)
    @assert n <= length(elen)
    @assert n <= length(last)
    @assert n <= length(degree)
    @assert n <= length(wf)
    @assert n <= length(next)
    @assert n <= length(w)
    @assert n <= length(wt)
    @assert nbbuck + two(V) <= length(head)

    @inbounds for i in oneto(n)
        if iszero(nbelts)                     # `elen` is input if the graph has elements
            elen[i] = zero(V)
        end

        last[i] = zero(V)
        wf[i] = zero(F)
        next[i] = zero(V)
        w[i] = zero(I)

        if iszero(nbelts)                     # in element mode, `degree` holds the external degrees of the variables on input
            degree[i] = zero(F)
        end
    end

    @inbounds for i in oneto(nbbuck + two(V))
        head[i] = zero(V)
    end

    #=====================================================================#
    #
    # deg :       the degree of a variable or element
    # degme :     size, |Lme|, of the current element, me
    # wdegme :    weight of Lme
    # dext :      external degree, |Le \ Lme|, of some element e
    # dmax :      largest |Le| seen so far
    # e :         an element
    # elenme :    the length, elen (me), of element list of pivotal var.
    # eln :       the length, elen (...), of an element list
    # hash :      the computed value of the hash function
    # hmod :      the hash function is computed modulo hmod = max (1,n-1)
    # i :         a supervariable
    # ilast :     the entry in a link list preceding i
    # inext :     the entry in a link list following i
    # j :         a supervariable
    # jlast :     the entry in a link list preceding j
    # jnext :     the entry in a link list, or path, following j
    # k :         the pivot order of an element or variable
    # knt1 :      loop counter used during element construction
    # knt2 :      loop counter used during element construction
    # knt3 :      loop counter used during compression
    # lenj :      len (j)
    # ln :        length of a supervariable list
    # maxint_n :  large integer to test risk of overflow on wflg
    # maxmem :    amount of memory needed for no compressions
    # me :        current supervariable being eliminated, and the
    #                     current element created by eliminating that
    #                     supervariable
    # mem :       memory in use assuming no compressions have occurred
    # mindeg :    current minimum bucket
    # nel :       number of pivots selected so far
    # newmem :    amount of new memory needed for current pivot element
    # wleft :     weight of the nonpivotal variables
    # nvi :       the number of variables in a supervariable i (= nv (i))
    # nvj :       the number of variables in a supervariable j (= nv (j))
    # nvpiv :     number of pivots in current element
    # slenme :    number of variables in variable list of pivotal variable
    # we :        w (e)
    # wflg :      used for flagging the w array.  See description of iw.
    # wnvi :      wflg - nv (i)
    # x :         either a supervariable or an element
    # wf3 :       weight of the variable neighbours outside Lme
    # wf4 :       areas of the elements outside Lme
    # wf5 :       areas of the variable neighbours outside Lme
    #
    #=====================================================================#

    #=====================================================================#
    #
    #             Any parameter (pe (...) or pfree) or local variable
    #             starting with "p" (for Pointer) is an index into iw,
    #             and all indices into iw use variables starting with
    #             "p."  The only exception to this rule is the iwlen
    #             input argument.
    # p :         pointer into lots of things
    # p1 :        pe (i) for some variable i (start of element list)
    # p2 :        pe (i) + elen (i) -  1 for some var. i (end of el. list)
    # p3 :        index of first supervariable in clean list
    # pdst :      destination pointer, for compression
    # pend :      end of memory to compress
    # pj :        pointer into an element or variable
    # pme :       pointer into the current element (pme1...pme2)
    # pme1 :      the current element, me, is stored in iw (pme1...pme2)
    # pme2 :      the end of the current element
    # pn :        pointer into a "clean" variable, also used to compress
    # psrc :      source pointer, for compression
    #
    #=====================================================================#

    pas = max(div(nfine, eight(V)), one(V)) # width of the buckets above `nfine`
    wflg = two(I)
    maxint_n = typemax(I) - convert(I, norig)
    ncmpa = zero(I)
    nel = zero(V)
    totel = zero(V)
    wnel = zero(F)
    wtotel = zero(F)
    hmod = max(one(E), convert(E, nbbuck) - one(E))
    dmax = zero(V)
    mem = pfree - one(E)
    maxmem = mem
    mindeg = zero(V)

    if iszero(nbelts)                         # if graph has no elements and only variables
        @inbounds for i in oneto(n)
            w[i] = one(I)
            totel += nv[i]
            wtotel += wt[i]
            deg = zero(F)

            for p in pe[i]:(pe[i] + convert(E, len[i]) - one(E))
                deg += wt[iw[p]]
            end

            degree[i] = deg
        end
    else                                      # graph has elements
        @inbounds for i in oneto(n)
            w[i] = one(I)

            if isnegative(elen[i])            # if `i` ∈ Ve
                nel += nv[i]
                totel += nv[i]                # elements are numbered first, so they count toward `totel`
                elen[i] = -nel
                deg = zero(F)

                for p in pe[i]:(pe[i] + convert(E, len[i]) - one(E))
                    deg += wt[iw[p]]
                end

                degree[i] = deg               # the weight of Le
                nv[i] = len[i]                # from now on, `nv[i]` is |Le|

                if len[i] > dmax
                    dmax = len[i]
                end
            else                              # `i` ∈ V0; `degree[i]` holds its weighted external degree
                totel += nv[i]
                wtotel += wt[i]
                elen[i] = len[i]
            end
        end
    end

    @inbounds for i in oneto(n)               # initialize degree lists and eliminate rows with no off-diag. nz.
        if isnegative(elen[i])                # skip element vertices (Ve)
            continue
        end

        deg = degree[i]

        if ispositive(deg)
            if isnothing(vfill)               # the initial score is the weighted degree
                q = amf_round(deg / wunit, n)
            else                              # score from the exact fill-in, as in the score update below
                q = amf_round(twice(convert(F, vfill[i])) / wt[i] / wunit, n)
            end

            wf[i] = q
            b = amf_bucket(q, nfine, pas, nbbuck)
            inext = head[b + one(V)]

            if !iszero(inext)
                last[inext] = i
            end

            next[i] = inext
            head[b + one(V)] = i
        else                                  # variable can be eliminated at once because no off-diagonal non-zero in its row
            nel += nv[i]
            wnel += wt[i]
            elen[i] = -nel
            pe[i] = zero(E)
            w[i] = zero(I)
        end
    end

    @inbounds while nel < totel
        deg = mindeg - one(V); me = zero(V)

        while deg < nbbuck && !ispositive(me)
            deg += one(V); me = head[deg + one(V)]
        end

        mindeg = deg

        if !ispositive(me)
            error()
        end

        if deg > nfine                        # linear search to find variable with best score in the list
            j = next[me]
            k = wf[me]

            while ispositive(j)               # L55: CONTINUE
                if wf[j] < k
                    me = j
                    k = wf[me]
                end

                j = next[j]
            end

            ilast = last[me]
            inext = next[me]

            if !iszero(inext)
                last[inext] = ilast
            end

            if !iszero(ilast)
                next[ilast] = inext
            else
                head[deg + one(V)] = inext    # `me` is at the head of the degree list
            end
        else                                  # remove chosen variable from link list
            inext = next[me]

            if !iszero(inext)
                last[inext] = zero(V)
            end

            head[deg + one(V)] = inext
        end

        elenme = elen[me]                     # `me` represents the elimination of pivots `nel + 1` to `nel + nv[me]`
        elen[me] = -one(V) - nel              # place `me` itself as the first in this set. it will be moved
        nvpiv = nv[me]                        # to the `nel + nv[me]` position when the permutation vectors are
        nel += nvpiv                          # computed
        wnel += wt[me]
        nv[me] = -nvpiv                       # at this point, `me` is the pivotal supervariable
        degme = zero(V)                       # it will be converted into the current element
        wdegme = zero(F)

        if iszero(elenme)                     # construct the new element in place
            pme1 = pe[me]
            pme2 = pme1 - one(E)

            for p in pme1:(pme1 + convert(E, len[me]) - one(E)) # scan list of the pivotal supervariable `me`
                i = iw[p]
                nvi = nv[i]

                if ispositive(nvi)            # `i` is a principal variable not yet placed in Lme
                    degme += nvi              # store `i` in new list
                    wdegme += wt[i]
                    nv[i] = -nvi              # flag `i` as being in Lme by negating `nv[i]`
                    pme2 += one(E)
                    iw[pme2] = i
                    ilast = last[i]           # remove variable `i` from degree list
                    inext = next[i]

                    if !iszero(inext)
                        last[inext] = ilast
                    end

                    if !iszero(ilast)
                        next[ilast] = inext
                    else
                        head[amf_bucket(wf[i], nfine, pas, nbbuck) + one(V)] = inext
                    end
                end
            end                               # L60:

            newmem = zero(E)
        else                                  # construct the new element in empty space, `iw[pfree...]`
            p = pe[me]
            pme1 = pfree
            slenme = len[me] - elenme
            knt1_updated = zero(V)

            for knt1 in oneto(elenme + one(V))
                knt1_updated += one(V)

                if knt1 > elenme              # search the supervariables in `me`
                    e = me
                    pj = p
                    ln = slenme
                else                          # search the elements in `me`
                    e = iw[p]; p += one(E)
                    pj = pe[e]
                    ln = len[e]
                end

                knt2_updated = zero(V)

                for knt2 in oneto(ln)              # search for different supervariables and add to new list
                    knt2_updated += one(V)
                    i = iw[pj]; pj += one(E)
                    nvi = nv[i]

                    if ispositive(nvi)
                        if pfree > iwlen      # compress `iw` if necessary
                            pe[me] = p        # prepare for compressing iw by adjusting pointers and lengths so that the lists
                            len[me] -= knt1_updated  # being searched in the inner and outer loops contain only the remaining entries
                            knt1_updated = zero(V)   # reset `knt1_updated` in case of recompress at same iteration of the loop 120

                            if iszero(len[me])       # check if anything left in supervariable `me`
                                pe[me] = zero(E)
                            end

                            pe[e] = pj
                            len[e] = ln - knt2_updated
                            knt2_updated = zero(V)   # reset `knt2_updated` in case of recompress at same iteration of the loop 110

                            if iszero(len[e])
                                pe[e] = zero(E)
                            end

                            ncmpa += one(I)

                            for j in oneto(n) # store first item in `pe`; set first entry to `-item`
                                pn = pe[j]

                                if ispositive(pn)
                                    pe[j] = convert(E, iw[pn])
                                    iw[pn] = -j
                                end
                            end               # L70

                            pdst = one(E)
                            psrc = one(E)
                            pend = pme1 - one(E)

                            while psrc <= pend # L80
                                j = -iw[psrc]
                                psrc += one(E)

                                if ispositive(j)
                                    iw[pdst] = convert(V, pe[j])
                                    pe[j] = pdst; pdst += one(E)
                                    lenj = len[j]

                                    for knt3 in zero(V):(lenj - two(V)) # L90
                                        iw[pdst + convert(E, knt3)] = iw[psrc + convert(E, knt3)]
                                    end

                                    pdst += convert(E, lenj - one(V))
                                    psrc += convert(E, lenj - one(V))
                                end
                            end

                            p1 = pdst         # move the new partially-constructed element

                            for psrc in pme1:pfree - one(E) # L100
                                iw[pdst] = iw[psrc]
                                pdst += one(E)
                            end

                            pme1 = p1
                            pfree = pdst
                            pj = pe[e]
                            p = pe[me]
                        end

                        degme += nvi          # `i` is a principal variable not yet placed in Lme; store `i` in new list
                        wdegme += wt[i]
                        nv[i] = -nvi
                        iw[pfree] = i; pfree += one(E)
                        ilast = last[i]       # remove variable `i` from degree link list
                        inext = next[i]

                        if !iszero(inext)
                            last[inext] = ilast
                        end

                        if !iszero(ilast)
                            next[ilast] = inext
                        else
                            head[amf_bucket(wf[i], nfine, pas, nbbuck) + one(V)] = inext
                        end
                    end
                end                           # L110:

                if e != me                    # set tree pointer and flag to indicate element `e` is absorbed into new element `me` (the parent of `e` is `me`)
                    pe[e] = convert(E, -me)
                    w[e] = zero(I)
                end
            end                               # L120:

            pme2 = pfree - one(E)
            newmem = pfree - pme1             # this element takes `newmem` new memory in `iw` (possibly zero)
            mem += newmem
            maxmem = max(maxmem, mem)
        end

        degree[me] = wdegme                   # `me` has now been converted into an element in `iw[pme1...pme2]`
        pe[me] = pme1
        len[me] = convert(V, pme2 - pme1 + one(E))

        if wflg > maxint_n                    # make sure that `wflg` is not too large; `wflg + n` must not cause integer overflow
            for x in oneto(n)
                if !iszero(w[x])
                    w[x] = one(I)
                end
            end

            wflg = two(I)                     # L130:
        end

        for pme in pme1:pme2                  # compute `w[e] - wflg` = |Le\Lme| and `wt[e]` = its weight for all elements
            i = iw[pme]
            eln = elen[i]

            if ispositive(eln)                # note that `nv[i]` has been negated to denote `i` in Lme
                nvi = -nv[i]
                wnvi = wflg - convert(I, nvi)
                wi = wt[i]

                # L140:
                for p in pe[i]:(pe[i] + convert(E, eln) - one(E))
                    e = iw[p]
                    we = w[e]

                    if we >= wflg             # unabsorbed element `e` has been seen in this loop
                        we -= convert(I, nvi)
                        wt[e] -= wi
                    elseif !iszero(we)        # `e` is an unabsorbed element; this is the first time we have seen `e` in all of scan 1
                        we = convert(I, nv[e]) + wnvi
                        wt[e] = degree[e] - wi
                        wf[e] = zero(F)
                    end

                    w[e] = we
                end                           # L140:
            end
        end                                   # L150:

        for pme in pme1:pme2                  # degree update and element absorption
            i = iw[pme]
            p1 = pe[i]
            p2 = p1 + convert(E, elen[i]) - one(E)
            pn = p1
            hash = zero(E)
            deg = zero(V)
            wdeg = zero(F)
            wf3 = zero(F)
            wf4 = zero(F)
            wf5 = zero(F)

            for p in p1:p2                    # scan the element list associated with supervariable `i`
                e = iw[p]
                dext = convert(V, w[e] - wflg) # `dext` = |Le\Lme|

                if ispositive(dext)
                    if iszero(wf[e])          # first time we meet `e`: compute `wf[e]`, the area of `e` outside Lme
                        xw = wt[e]
                        wf[e] = xw * (twice(degree[e]) - xw)
                    end

                    wf4 += wf[e]
                    deg += dext
                    wdeg += wt[e]
                    iw[pn] = e; pn += one(E)
                    hash += convert(E, e)
                elseif iszero(dext)           # aggressive absorption: `e` is not adjacent to `me`, but |Le\Lme| is 0, so absorb it into `me`
                    pe[e] = convert(E, -me)
                    w[e] = zero(I)
                end
            end                               # L160:

            elen[i] = convert(V, pn - p1) + one(V) # count the number of elements in `i` (including `me`)
            p3 = pn

            for p in (p2 + one(E)):(p1 + convert(E, len[i]) - one(E)) # scan the supervariables in the list associated with `i`
                j = iw[p]
                nvj = nv[j]

                if ispositive(nvj)            # `j` is unabsorbed, and not in Lme; add to degree and add to new list
                    wj = wt[j]
                    deg += nvj
                    wdeg += wj
                    wf3 += wj
                    wf5 += wj * wj
                    iw[pn] = j; pn += one(E)
                    hash += convert(E, j)
                end
            end

            if iszero(deg)                    # mass elimination
                pe[i] = convert(E, -me)
                nvi = -nv[i]
                degme -= nvi
                wdegme -= wt[i]
                nvpiv += nvi
                nel += nvi
                wnel += wt[i]
                nv[i] = zero(V)
                elen[i] = zero(V)
            else                              # update the upper-bound degree of `i`
                if degree[i] < wdeg           # our approx degree is loose; we cannot subtract the areas
                    wf3 = zero(F)
                    wf4 = zero(F)
                    wf5 = zero(F)
                else
                    degree[i] = wdeg
                end

                wf[i] = wf4 + twice(wt[i]) * wf3 + wf5
                iw[pn] = iw[p3]               # add `me` to the list for `i`
                iw[p3] = iw[p1]
                iw[p1] = me                   # add new element to front of list
                len[i] = convert(V, pn - p1) + one(V) # store the new length of the list in `len[i]`
                hash = (hash % hmod) + one(E) # place in hash bucket; save hash key of `i` in `last[i]`
                j = head[hash + one(E)]

                if !ispositive(j)             # the degree list is empty; hash head is `-j`
                    next[i] = -j
                    head[hash + one(E)] = -i
                else                          # degree list is not empty; use `last[head[hash + 1]]` as hash head
                    next[i] = last[j]
                    last[j] = i
                end

                last[i] = hash
            end
        end                                  # L180:

        degree[me] = wdegme

        if degme > dmax                      # clear the counter array `w` by incrementing `wflg`
            dmax = degme
        end

        wflg += convert(I, dmax)

        if wflg > maxint_n                   # make sure that `wflg + n` does not cause integer overflow
            for x in oneto(n)
                if !iszero(w[x])
                    w[x] = one(I)
                end
            end

            wflg = two(I)
        end

        for pme in pme1:pme2                 # supervariable detection
            i = iw[pme]

            if isnegative(nv[i])             # `i` is a principal variable in Lme
                hash = last[i]
                j = head[hash + one(E)]

                if iszero(j)
                    continue                 # goto L250:
                end

                if isnegative(j)             # degree list is empty
                    i = -j
                    head[hash + one(E)] = zero(V)
                else                         # degree list is not empty: restore `last` of head
                    i = last[j]
                    last[j] = zero(V)
                end

                while !iszero(i) && !iszero(next[i]) # while loop L200:
                    ln = len[i]
                    eln = elen[i]

                    for p in (pe[i] + one(E)):(pe[i] + ln - one(E))
                        w[iw[p]] = wflg
                    end

                    jlast = i
                    j = next[i]

                    while !iszero(j)         # while loop L220:
                        L240 = false
                        L240 = L240 || len[j] != ln   # jump if `i` and `j` do not have same size data structure
                        L240 = L240 || elen[j] != eln # jump if `i` and `j` do not have same number adj elts

                        if !L240
                            for p in (pe[j] + one(E)):(pe[j] + convert(E, ln) - one(E))
                                L240 && break
                                L240 = w[iw[p]] != wflg # jump if an entry `w[iw[p]]` is in `j` but not in `i`
                            end
                        end

                        if L240
                            jlast = j        # `j` cannot be absorbed into `i`
                            j = next[j]
                        else
                            pe[j] = convert(E, -i) # found it! `j` can be absorbed into `i`

                            if wf[j] > wf[i]
                                wf[i] = wf[j]
                            end

                            nv[i] += nv[j]   # both `nv[i]` and `nv[j]` are negated since they are in Lme
                            wt[i] += wt[j]
                            nv[j] = zero(V)
                            elen[j] = zero(V)
                            j = next[j]      # delete `j` from hash bucket
                            next[jlast] = j
                        end
                    end

                    wflg += one(I)
                    i = next[i]
                end
            end
        end

        p = pme1                             # restore degree lists and remove nonprincipal supervariable from element
        wleft = wtotel - wnel

        for pme in pme1:pme2
            i = iw[pme]
            nvi = -nv[i]

            if ispositive(nvi)               # `i` is a principal variable in Lme; restore `nv[i]` to signify that `i` is principal
                nv[i] = nvi
                wi = wt[i]                   # compute the external degree (add size of current elem)
                dg = degree[i]

                if dg + wdegme > wleft
                    rmf1 = dg * (dg + twice(wdegme)) - wf[i]
                    degree[i] = wleft - wi
                    dg = degree[i]
                    rmf = dg * dg - (wdegme - wi) * (wdegme - wi)
                    rmf = min(rmf, rmf1)
                else
                    degree[i] += wdegme - wi
                    rmf = dg * (dg + twice(wdegme)) - wf[i]
                end

                q = amf_round(rmf / wi / wunit, n)
                wf[i] = q
                b = amf_bucket(q, nfine, pas, nbbuck)
                inext = head[b + one(V)]

                if !iszero(inext)
                    last[inext] = i
                end

                next[i] = inext
                last[i] = zero(V)
                head[b + one(V)] = i

                if mindeg > b                # save the new score, and find the minimum bucket
                    mindeg = b
                end

                iw[p] = i; p += one(E)       # place the supervariable in the element pattern
            end
        end # L260:

        nv[me] = degme                       # finalize the new element: from now on, `nv[me]` is |Le|
        len[me] = convert(V, p - pme1)

        if iszero(len[me])                   # there is nothing left of the current pivot element
            pe[me] = zero(E)
            w[me] = zero(I)
        end

        if !iszero(newmem)                   # element was not constructed in place: deallocate part of it
            pfree = p
            mem = mem - newmem + len[me]
        end
    end                                      # end while (selecting pivots)

    @inbounds for i in oneto(n)
        if iszero(elen[i])
            j = convert(V, -pe[i])

            while !isnegative(elen[j])      # L270:
                j = convert(V, -pe[j])
            end

            e = j
            k = -elen[e]                    # get the current pivot ordering of `e`
            j = i

            while !isnegative(elen[j])      # L280:
                jnext = convert(V, -pe[j])
                pe[j] = convert(E, -e)

                if iszero(elen[j])
                    elen[j] = k; k += one(V)
                end

                j = jnext
            end

            elen[e] = -k
        end
    end                                     # L290:

    return ncmpa
end

end
