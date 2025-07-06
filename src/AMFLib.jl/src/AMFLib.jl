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

using ArgCheck
using Base: oneto
using FillArrays
using ..Utilities

export amf

const FLOAT = Float64

function anint(::Type{I}, x::F) where {I, F}
    return floor(I, x + convert(F, 0.5))
end

function amf(n::V, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V, E}
    vwght = Ones{V}(n)
    return amf(n, vwght, xadj, adjncy)
end

function amf(n::V, vwght::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V, E}
    @argcheck n <= length(vwght)
    @inbounds nn = n + one(V); mm = xadj[nn]; m = mm - one(E)

    norig = zero(V)
    iwlen = m + convert(E, 4n + 10000)

    @inbounds for i in oneto(n)
        norig += trunc(V, vwght[i])
    end

    nbbuck = twice(norig)

    len = FVector{V}(undef, n)
    pe = FVector{E}(undef, nn)
    iw = FVector{V}(undef, iwlen)
    nv = FVector{V}(undef, n)
    elen = FVector{V}(undef, n)
    last = FVector{V}(undef, n)
    degree = FVector{V}(undef, n)
    wf = FVector{E}(undef, n)
    next = FVector{V}(undef, n)
    w = FVector{Int}(undef, n)
    head = FVector{V}(undef, nbbuck + two(V))

    perm, invp = amf_impl!(norig, n, nbbuck, iwlen, pe,
        len, iw, nv, elen, last, degree, wf, next, w, head,
        vwght, xadj, adjncy)

    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

function amf_impl!(
        norig::V,
        n::V,
        nbbuck::V,
        iwlen::E,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        nv::AbstractVector{V},
        elen::AbstractVector{V},
        last::AbstractVector{V},
        degree::AbstractVector{V},
        wf::AbstractVector{E},
        next::AbstractVector{V},
        w::AbstractVector,
        head::AbstractVector{V},
        vwght::AbstractVector,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
    ) where {V, E}
    @inbounds nn = n + one(V); mm = xadj[nn]; m = mm - one(E)

    nbelts = zero(V)
    pfree = mm

    @inbounds for p in oneto(m)
        iw[p] = adjncy[p]
    end

    @inbounds p = pe[begin] = xadj[begin]

    @inbounds for i in oneto(n)
        ii = i + one(E)
        pp = xadj[ii]

        nv[i] = trunc(V, vwght[i])
        len[i] = convert(V, pp - p)
        pe[ii] = p = pp
    end

    ncmpa = hamf_impl!(norig, n, nbelts, nbbuck, iwlen, pe,
        pfree, len, iw, nv, elen, last, degree, wf, next, w, head)

    @inbounds for j in oneto(norig)
        head[j] = zero(V)
    end

    @inbounds for i in oneto(n)
        j = abs(elen[i])
        head[j] = i
    end

    k = zero(V)

    @inbounds for j in oneto(norig)
        i = head[j]

        if ispositive(i)
            last[i] = k += one(V)
            elen[k] = i
        end
    end

    return elen, last
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
#     MUMPS_HAMF4 also enables to take into account a halo in the graph.
#     The graph is composed is partitioned in two types of nodes
#     the so called internal nodes and the so called halo nodes.
#     Halo nodes cannot be selected the both the initial degrees
#     and updated degrees of internal node should be taken
#     into account.
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
#         V = Ve + V0 + V1
#             V0 = Set of variable nodes (not in halo)
#             V1 = Set of variable nodes (in halo)
#             Ve = Set of element nodes |Ve|=nbelts
#
#     All 3 sets are disjoint, Ve and V1 can be empty
#     If nbelts>0 then a bipartite graph bewteen
#        (V0 U V1) and Ve is provided on entry.
#
#     A graph of elements and variables is a bipartite
#     graph between the set of variables (that may
#     include halo variables) and the set of elements.
#     Thus variables are only adjacent to elements and
#     in the element list we only have variables.
#     Elements are "considered" as already eliminated and
#     are provided here only to describe the adjacency between
#     variables. Only variables in V0 need to be eliminated.
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
#        N : number of nodes in the complete graph including halo
#            N = size of V0 U V1
#        NBBUCK : should be greater than norig
#                 advised value is 2*norig
#        IWLEN : Should be uint64_t # Patch FP: used Gnum as well #
#      Input, undefined on output:
#        LEN(1..N)
#        IW(1..IWLEN)
#        NV(N) : also meaningful as input to encode compressed graphs
#      Output only :
#        NCMPA
#        not ELEN(N)  # Patch FP: always initialized before call #
#        LAST(N)
#      Input/output
#        ELEN(N)  # Patch PA+FP #
#        PFREE
#        PE(N) : Should be uint64_t # Patch FP: used Gnum as well #
#      Internal workspace only
#        Min fill approximation one extra array of size NBBUCK+2
#         is also needed
#        NEXT(N)
#        DEGREE(N)
#        W(N)
#        HEAD(0:NBBUCK+1)
#        WF(N)
#
#    Comments on the OUTPUT :
#    ------------------------
#    Let V= V0 U V1 the nodes of the initial graph (|V|=n).
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
#        elements in column i of L (true degree of i)
#      nv(i) can be greater than N since degree can be as large as NORIG.
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
#    Comments on the OUTPUT:
#    ----------------------
#    Let V= V0 U V1 the nodes of the initial graph (|V|=n).
#    The assembly tree corresponds to the tree
#      of the supernodes (or supervariables). Each node of the
#      assembly tree is then composed of one principal variable
#      and a list of secondary variables. The list of
#      variable of a node (principal + secondary variables) then
#      describes the structure of the diagonal block of the
#      supernode.
#    The elimination tree denotes the tree of all the variables(=node)
#      and is therefore of order n.
#
#    The arrays NV(N) and PE(N) give a description of the
#    assembly tree.
#
#     1/ Description of array nv(N) (on OUTPUT)
#      nv(i)=0 i is a secondary variable
#      nv(i) >0 i is a principal variable, nv(i) holds the number of
#        elements in column i of L (true degree of i)
#      with compressed graph (nv(1).ne.-1 on input), nv(i) can be
#      greater than N since degree can be as large as NORIG.
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
#        If is the principal
#        variable of the node If and let If1, If2, If3 be the secondary
#        variables of node If.
#        Is is the principal
#        variable of the node Is and let Is1, Is2 be the secondary
#        variables of node Is.
#
#        THEN:
#          NV(If1)=NV(If2)=NV(If3) = 0  (secondary variables)
#          NV(Is1)=NV(Is2) = 0  (secondary variables)
#          NV(If) > 0  ( principal variable)
#          NV(Is) > 0  ( principal variable)
#          PE(If)  = 0 (root node)
#          PE(Is)  = -If (If is the father of Is in the assembly tree)
#          PE(If1)=PE(If2)=PE(If3)= -If  ( If is the principal variable)
#          PE(Is1)=PE(Is2)= -Is  ( Is is the principal variable)
#
#
#
#   HALOAMD_V1: (September 1997)
#   **********
#   Initial version designed to experiment the numerical (fill-in)
#   impact of taking into account the halo. This code should be able
#   to experiment no-halo, partial halo, complete halo.
#   DATE: September 17th 1997
#
#   HALOAMD is designed to process a gragh composed of two types
#              of nodes, V0 and V1, extracted from a larger gragh.
#              V0^V1 = {},
#
#              We used Min. degree heuristic to order only
#              nodes in V0, but the adjacency to nodes
#              in V1 is taken into account during ordering.
#              Nodes in V1 are odered at last.
#              Adjacency between nodes of V1 need not be provided,
#              however |len(i)| must always corresponds to the number
#              of edges effectively provided in the adjacency list of i.
#            On input :
#            ********
#              Nodes INODE in V1 are flagged with len(INODE) = -degree
#                             if len(i) =0 and i ∈ V1 then
#                             len(i) must be set on input to -NORIG-1
#            ERROR return (negative values in ncmpa)
#            ************
#              negative value in ncmpa indicates an error detected
#                 by HALOAMD.
#
#              The graph provided MUST follow the rule:
#               if (i,j) is an edge in the gragh then
#               j must be in the adjacency list of i AND
#               i must be in the adjacency list of j.
#      REMARKS
#      -------
#
#          1/  Providing edges between nodes of V1 should not
#              affect the final ordering, only the amount of edges
#              of the halo should effectively affect the solution.
#              This code should work in the following cases:
#                1/ halo not provided
#                2/ halo partially provided
#                3/ complete halo
#                4/ complete halo+interconnection between nodes of V1.
#
#                1/ should run and provide identical results (w.r.t to
#                   current implementation of AMD in SCOTCH).
#                3/ and 4/ should provide identical results.
#
#          2/ All modifications of the AMD initial code are indicated
#             with begin HALO .. end HALO
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
#   n:    number of nodes in the complete graph including halo
#         n = size of V0 U V1 U Ve
#         Restriction:  n .ge. 1
#
#   norig   if compressed graph (nv(1).ne.-1) then
#              norig is the sum(nv(i)) for i \in [1:n]
#              and could be larger than n
#           else norig = n
#
#   nbelts number of elements (size of Ve)
#              =0 if Gv (graph of variables)
#              >0 if Ge
#           nbelts > 0 extends/changes the meaning of
#                     len/elen on entry (see below)
#
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
#   nv:   On input, encoding of compressed graph:
#         During execution, abs (nv (i)) is equal to the number of rows
#         that are represented by the principal supervariable i.  If i
#         is a nonprincipal variable, then nv (i) = 0.  Initially,
#         nv (i) = 1 for all i.  nv (i) .lt. 0 signifies that i is a
#         principal variable in the pattern Lme of the current pivot
#         element me.  On output, nv (e) holds the true degree of
#         element e at the time it was created (including the diagonal
#         part).
#   begin HALO
#         On output, nv(I) can be used to find node in set V1.
#         Not true anymore : ( nv(I) = N+1 characterizes nodes in V1.
#                   instead nodes in V1 are considered as a dense root
#                   node)
#   end HALO
#  ---------------------------------------------------------------------
#   INPUT/MODIFIED (undefined on output):
#  ---------------------------------------------------------------------
#   len:  On input, len (i)
#             positive or null (>=0) : i ∈ V0 U Ve and
#                     if (nbelts==0) then (graph of variables)
#                       len(i) holds the number of entries in row i of
#                       the matrix, excluding the diagonal.
#                     else (graph of elements+variables)
#                       if i ∈ V0 then len(i) = nb of elements adjacent
#                           to i
#                       if i ∈ Ve then len(i) = nb of variables
#                           adjacent to i
#                     endif
#             negative (<0) : i ∈ V1, and
#                     if (nbelts==0) then (graph of variables)
#                       -len(i) hold the number of entries in row i of
#                        the matrix, excluding the diagonal.
#                        len(i) = - | Adj(i) | if i ∈ V1
#                                or -N -1 if | Adj(i) | = 0 and i ∈ V1
#                     else  (graph of elements+variables)
#                       -len(i) nb of elements adjacent to i
#                     endif
#         The content of len (1..n) is undefined on output.
#
#   elen: defined on input only if nbelts>0
#         if e ∈ Ve then elen (e) = -N-1
#         if v ∈ V0 then elen (v) = External degree of v
#                                   that should thus be provided
#                                   on entry to initialize degree
#
#         if v ∈ V1 then elen (v) = 0
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
#   begin HALO
#          on output ncmpa <0 --> error detected during HALO_AMD:
#             error 1: ncmpa = -NORIG , ordering was stopped.
#   end HALO
#
#  ---------------------------------------------------------------------
#   LOCAL (not input or output - used only during execution):
#  ---------------------------------------------------------------------
#   degree:       If i is a supervariable, then degree (i) holds the
#         current approximation of the external degree of row i (an
#         upper bound).  The external degree is the number of nonzeros
#         minus abs (nv (i)) (the diagonal part).  The bound is equal to
#         in row i, the external degree if elen (i) is less than or
#         equal to two.
#         We also use the term "external degree" for elements e to refer
#         to |Le \ Lme|.  If e is an element, then degree (e) holds
#         |Le|, which is the degree of the off-diagonal part of the
#         element e (not including the diagonal part).
#   begin HALO
#         degree(I) = n+1 indicates that i belongs to V1
#   end HALO
#
#   head: head is used for degree lists.  head (deg) is the first
#         supervariable in a degree list (all supervariables i in a
#         degree list deg have the same approximate degree, namely,
#         deg = degree (i)).  If the list deg is empty then
#         head (deg) = 0.
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
#                 the set |Le \ Lme|, in terms of nonzeros (the
#                 sum of abs (nv (i)) for each principal variable i that
#                 is both in the pattern of element e and NOT in the
#                 pattern of the current pivot element, me).
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
#   wf : integer array  used to store the already filled area of
#        the variables adajcent to current pivot.
#        wf is then used to update the score of variable i.
function hamf_impl!(
        norig::V,                  # uncompressed matrix order
        n::V,                      # current matrix order
        nbelts::V,                 # number of elements
        nbbuck::V,                 # number of buckets
        iwlen::E,                  # length of array `iw`
        pe::AbstractVector{E},     # array of indices in `iw` of start of row `i`
        pfree::E,                  # useful size in `iw`
        len::AbstractVector{V},    # array of lengths of adjacency lists
        iw::AbstractVector{V},     # adjacency list array
        nv::AbstractVector{V},     # array of element degrees (weights)
        elen::AbstractVector{V},   # array that holds the inverse permutation
        last::AbstractVector{V},   # array that holds the permutation
        degree::AbstractVector{V}, # array that holds degree data
        wf::AbstractVector{E},     # flag array
        next::AbstractVector{V},   # linked list structure
        w::AbstractVector{I},      # flag array
        head::AbstractVector{V},   # linked list structure
    ) where {I, V, E}
    @argcheck nbelts <= n
    @argcheck pfree <= iwlen <= length(iw)
    @argcheck norig <= nbbuck
    @argcheck n < length(pe)
    @argcheck n <= length(len)
    @argcheck n <= length(nv)
    @argcheck n <= length(elen)
    @argcheck n <= length(last)
    @argcheck n <= length(degree)
    @argcheck n <= length(wf)
    @argcheck n <= length(next)
    @argcheck n <= length(w)
    @argcheck nbbuck + two(V) <= length(head)

    @inbounds for i in oneto(n)
        elen[i] = zero(V)
        last[i] = zero(V)
        degree[i] = zero(V)
        wf[i] = zero(V)
        next[i] = zero(V)
        w[i] = zero(I)
    end

    @inbounds for i in oneto(nbbuck + two(V))
        head[i] = zero(V)
    end

    #=====================================================================#
    #
    # deg :       the degree of a variable or element
    # degme :     size, |Lme|, of the current element, me (= degree (me))
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
    # mindeg :    current minimum degree
    # nel :       number of pivots selected so far
    # newmem :    amount of new memory needed for current pivot element
    # nleft :     n - nel, the number of nonpivotal rows/columns remaining
    # nvi :       the number of variables in a supervariable i (= nv (i))
    # nvj :       the number of variables in a supervariable j (= nv (j))
    # nvpiv :     number of pivots in current element
    # slenme :    number of variables in variable list of pivotal variable
    # we :        w (e)
    # wflg :      used for flagging the w array.  See description of iw.
    # wnvi :      wflg - nv (i)
    # x :         either a supervariable or an element
    # wf3 :       off diagoanl block area
    # wf4 :       diagonal block area
    # mf :        minimum fill
    # nbflag :    number of flagged entries in the initial gragh.
    # nreal :     number of entries on which ordering must be perfomed
    #             (nreal = n - nbflag)
    # nelme :     number of pivots selected when reaching the root
    # lastd :     index of the last row in the list of dense rows
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

    idummy = typemax(E) - one(E)
    dummy = convert(FLOAT, idummy)
    n2 = -one(V) - nbbuck                   # variable with degree `n2` are in halo; ; bucket `nbbuck + 1` used for halo variables
    pas = max(div(norig, eight(V)), one(V)) # distance between elements of the `n`, ..., `nbbuck` entries of `head`
    wflg = two(I)
    maxint_n = typemax(I) - convert(I, norig)
    ncmpa = zero(I)
    nel = zero(V)
    totel = zero(V)
    hmod = max(one(E), nbbuck - one(E))
    dmax = zero(V)
    mem = pfree - one(E)
    maxmem = mem
    mindeg = zero(V)
    nbflag = zero(V)
    lastd = zero(V)

    if iszero(nbelts)                         # if graph has no elements and only variables
        @inbounds for i in oneto(n)
            elen[i] = zero(V)                 # already done before calling
            w[i] = one(I)

            if isnegative(len[i])
                degree[i] = n2
                nbflag += one(V)

                if len[i] == -one(V) - norig  # variable in V1 with empty adj. list
                    len[i] = zero(V)          # because of compress, we force skipping this entry (which is anyway empty)
                    pe[i] = zero(E)
                else
                    len[i] = -len[i]
                end
            else
                totel += nv[i]

                if n == norig                 # if graph not compressed
                    degree[i] = len[i]
                else
                    degree[i] = zero(V)

                    for p in pe[i]:(pe[i] + convert(E, len[i]) - one(E))
                        degree[i] += nv[iw[p]]
                    end
                end
            end
        end
    else                                      # graph has elements
        @inbounds for i in oneto(n)
            w[i] = one(I)

            if isnegative(len[i])
                degree[i] = n2
                nbflag += one(V)

                if len[i] == -one(V) - norig
                    len[i] = zero(V)          # because of compress, we force skipping this entry (which is anyway empty)
                    pe[i] = zero(E)
                    elen[i] = zero(V)         # already done before calling
                else
                    len[i] = -len[i]
                    elen[i] = len[i]
                end
            else                              # non-halo vertex or element
                if isnegative(elen[i])        # if `i` ∈ Ve
                    nel += nv[i]
                    elen[i] = -nel

                    if n == norig
                        degree[i] = len[i]
                    else
                        degree[i] = zero(V)

                        for p in pe[i]:(pe[i] + convert(E, len[i]) - one(E))
                            degree[i] += nv[iw[p]]
                        end
                    end

                    if degree[i] > dmax
                        dmax = degree[i]
                    end
                else                          # `i` ∈ V0
                    totel += nv[i]
                    degree[i] = elen[i]
                    elen[i] = len[i]
                end
            end
        end
    end

    nreal = n - nbflag                        # number of entries to be ordered

    @inbounds for i in oneto(n)               # initialize degree lists and eliminate rows with no off-diag. nz.
        if isnegative(elen[i])                # skip element vertices (Ve)
            continue
        end

        deg = degree[i]
        if deg == n2                          # V1 variables (`deg == n2`): flagged variables stored in degree list of `nbbuck + 1`
            deg = nbbuck + one(V)

            if iszero(lastd)                  # degree list is empty
                lastd = i
                head[deg + one(V)] = i
                next[i] = zero(V)
                last[i] = zero(V)
            else
                next[lastd] = i
                last[i] = lastd
                lastd = i
                next[i] = zero(V)
            end
        elseif ispositive(deg)
            wf[i] = convert(I, deg)           # version 1

            if deg > norig
                deg = min(div(deg - norig, pas) + norig, nbbuck)  # note the if `deg == 0`, no fill-in will occur but one variable adjacent to i
            end

            inext = head[deg + one(V)]

            if !iszero(inext)
                last[inext] = i
            end

            next[i] = inext
            head[deg + one(V)] = i
        else                                  # variable can be eliminated at once because no off-diagonal non-zero in its row
            nel += nv[i]
            elen[i] = -nel
            pe[i] = zero(E)
            w[i] = zero(I)
        end
    end

    nleft = totel - nel                       # if elements provided (`nbelts > 0`), they are eliminated

    @inbounds while nel < totel
        deg = mindeg - one(V); me = zero(V)

        while deg < nbbuck && !ispositive(me)
            deg += one(V); me = head[deg + one(V)]
        end

        mindeg = deg

        if !ispositive(me)
            error()
        end

        if deg > norig                        # linear search to find variable with best score in the list
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
        nv[me] = -nvpiv                       # at this point, `me` is the pivotal supervariable
        degme = zero(V)                       # it will be converted into the current element

        if iszero(elenme)                     # construct the new element in place
            pme1 = pe[me]
            pme2 = pme1 - one(E)

            for p in pme1:(pme1 + convert(E, len[me]) - one(E)) # scan list of the pivotal supervariable `me`
                i = iw[p]
                nvi = nv[i]

                if ispositive(nvi)            # `i` is a principal variable not yet placed in Lme
                    degme += nvi              # store `i` in new list
                    nv[i] = -nvi              # flag `i` as being in Lme by negating `nv[i]`
                    pme2 += one(E)
                    iw[pme2] = i

                    if degree[i] != n2        # remove variable `i` from degree list. (only if `i` ∈ V0)
                        ilast = last[i]
                        inext = next[i]

                        if !iszero(inext)
                            last[inext] = ilast
                        end

                        if !iszero(ilast)
                            next[ilast] = inext
                        else
                            if wf[i] > norig
                                deg = min(div(wf[i] - norig, pas) + norig, nbbuck)
                            else
                                deg = wf[i]
                            end

                            head[deg + one(V)] = inext
                        end
                    end
                end
            end                               # L60:

            newmem = zero(E)
        else                                  # construct the new element in empty space, `iw[pfree...]`
            p = pe[me]
            pme1 = pfree
            slenme = len[me] - elenme
            knt1_updated = zero(V)
            knt1 = one(V)

            while knt1 <= elenme + one(V)
                knt1_updated += one(V)

                if knt1 > elenme              # search the supervariables in `me`
                    e = me
                    pj = p
                    ln = slenme
                else                          # search the elements in `me`
                    e = iw[p]
                    p += one(E)
                    pj = pe[e]
                    ln = len[e]
                end

                knt2_updated = zero(V)
                knt2 = one(V)

                while knt2 <= ln              # search for different supervariables and add to new list
                    knt2_updated += one(V)
                    i = iw[pj]
                    pj += one(E)
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
                                    pe[j] = pdst
                                    pdst += one(E)
                                    lenj = len[j]

                                    for knt3 in zero(V):(lenj - two(V)) # L90
                                        pnt3 = convert(E, knt3)
                                        iw[pdst + pnt3] = iw[psrc + pnt3]
                                    end

                                    pdst += convert(E, lenj - one(V))
                                    psrc += convert(E, lenj - one(V))
                                end
                            end

                            p1 = pdst         # move the new partially-constructed element
                            psrc = pme1

                            while psrc < pfree # L100
                                iw[pdst] = iw[psrc]
                                pdst += one(E)
                                psrc += one(E)
                            end

                            pme1 = p1
                            pfree = pdst
                            pj = pe[e]
                            p = pe[me]
                        end

                        degme += nvi          # `i` is a principal variable not yet placed in Lme; store `i` in new list
                        nv[i] = -nvi
                        iw[pfree] = i
                        pfree += one(E)

                        if degree[i] != n2    # remove variable `i` from degree link list (only if `i` in V0)
                            ilast = last[i]
                            inext = next[i]

                            if !iszero(inext)
                                last[inext] = ilast
                            end

                            if !iszero(ilast)
                                next[ilast] = inext
                            else
                                if wf[i] > norig
                                    deg = min(div(wf[i] - norig, pas) + norig, nbbuck)
                                else
                                    deg = wf[i]
                                end

                                head[deg + one(V)] = inext
                            end
                        end
                    end

                    knt2 += one(V)
                end                           # L110:

                if e != me                    # set tree pointer and flag to indicate element `e` is absorbed into new element `me` (the parent of `e` is `me`)
                    pe[e] = convert(E, -me)
                    w[e] = zero(I)
                end

                knt1 += one(V)
            end                               # L120:

            pme2 = pfree - one(E)
            newmem = pfree - pme1             # this element takes `newmem` new memory in `iw` (possibly zero)
            mem += newmem
            maxmem = max(maxmem, mem)
        end

        degree[me] = degme                    # `me` has now been converted into an element in `iw[pme1...pme2]`
        pe[me] = pme1                         # `degme` holds the external degree of new element
        len[me] = convert(V, pme2 - pme1 + one(E))

        if wflg > maxint_n                    # make sure that `wflg` is not too large; `wflg + n` must not cause integer overflow
            for x in oneto(n)
                if !iszero(w[x])
                    w[x] = one(I)
                end
            end

            wflg = two(I)                     # L130:
        end

        for pme in pme1:pme2                  # compute `w[e] - wflg` = |Le\Lme| for all elements
            i = iw[pme]
            eln = elen[i]

            if ispositive(eln)                # note that `nv[i]` has been negated to denote `i` in Lme
                nvi = -nv[i]
                wnvi = wflg - convert(I, nvi)
                # L140:
                for p in pe[i]:(pe[i] + convert(E, eln) - one(E))
                    e = iw[p]
                    we = w[e]

                    if we >= wflg             # unabsorbed element `e` has been seen in this loop
                        we -= convert(I, nvi)
                    elseif !iszero(we)        # `e` is an unabsorbed element; this is the first time we have seen `e` in all of scan 1
                        we = convert(I, degree[e]) + wnvi
                        wf[e] = zero(E)
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
            wf3 = zero(E)
            wf4 = zero(E)
            nvi = -nv[i]

            for p in p1:p2                    # scan the element list associated with supervariable `i`
                e = iw[p]
                dext = convert(V, w[e] - wflg) # `dext` = |Le\Lme|

                if ispositive(dext)
                    if iszero(wf[e])          # first time we meet `e`: compute `wf[e]` which is the surface associated to element `e`
                        wf[e] = convert(E, dext) * convert(E, twice(degree[e]) - dext - one(V))
                    end

                    wf4 += wf[e]
                    deg += dext
                    iw[pn] = e
                    pn += one(E)
                    hash += convert(E, e)
                elseif iszero(dext)           # aggressive absorption: `e` is not adjacent to `me`, but |Le\Lme| is 0, so absorb it into `me`
                    pe[e] = convert(E, -me)
                    w[e] = zero(I)
                end
            end                               # L160:

            elen[i] = pn - p1 + one(E)        # count the number of elements in `i` (including `me`)
            p3 = pn

            for p in (p2 + one(E)):(p1 + convert(E, len[i]) - one(E)) # scan the supervariables in the list associated with `i`
                j = iw[p]
                nvj = nv[j]

                if ispositive(nvj)            # `j` is unabsorbed, and not in Lme; add to degree and add to new list
                    deg += nvj
                    wf3 += nvj
                    iw[pn] = j
                    pn += one(E)
                    hash += convert(E, j)
                end
            end

            if degree[i] == n2
                deg = n2
            end

            if iszero(deg)                    # mass elimination
                pe[i] = convert(E, -me)
                nvi = -nv[i]
                degme -= nvi
                nvpiv += nvi
                nel += nvi
                nv[i] = zero(V)
                elen[i] = zero(V)
            else                              # update the upper-bound degree of `i`
                if degree[i] != n2            # `i` does not belong to halo
                    if degree[i] < deg        # our approx degree is loose; we cannot subtract `wf[i]`
                        wf4 = zero(E)
                        wf3 = zero(E)
                    else
                        degree[i] = deg
                    end
                end

                wf[i] = wf4 + convert(E, twice(nvi)) * wf3
                iw[pn] = iw[p3]               # add `me` to the list for `i`
                iw[p3] = iw[p1]
                iw[p1] = me                   # add new element to front of list
                len[i] = pn - p1 + one(E)     # store the new length of the list in `len[i]`

                if deg != n2                  # place in hash bucket; save hash key of `i` in `last[i]`
                    hash = (hash % hmod) + one(E)
                    j = head[hash + one(E)]

                    if !ispositive(j)         # the degree list is empty; hash head is `-j`
                        next[i] = -j
                        head[hash + one(E)] = -i
                    else                     # degree list is not empty; use `last[head[hash + 1]]` as hash head
                        next[i] = last[j]
                        last[j] = i
                    end

                    last[i] = hash
                end
            end
        end                                  # L180:

        degree[me] = degme

        if degme > dmax                      # clear the counter array `w` by incrementing `wflg`
            dmax = degme
        end

        wflg += dmax

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

            if isnegative(nv[i]) && (degree[i] != n2) # `i` is a principal variable in Lme
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

                if iszero(i)
                    continue
                end

                @label L200                  # while loop L200:

                if !iszero(next[i])
                    ln = len[i]
                    eln = elen[i]

                    for p in (pe[i] + one(E)):(pe[i] + ln - one(E))
                        w[iw[p]] = wflg
                    end

                    jlast = i
                    j = next[i]

                    @label L220              # while loop L220:

                    if !iszero(j)
                        if len[j] != ln      # jump if `i` and `j` do not have same size data structure
                            @goto L240
                        end

                        if elen[j] != eln    # jump if `i` and `j` do not have same number adj elts
                            @goto L240
                        end

                        for p in (pe[j] + one(E)):(pe[j] + convert(E, ln) - one(E))
                            if w[iw[p]] != wflg # jump if an entry `[iw[p]]` is in `j` but not in `i`
                                @goto L240
                            end              # L230:
                        end

                        pe[j] = convert(E, -i) # found it! `j` can be absorbed into `i`

                        if wf[j] > wf[i]
                            wf[i] = wf[j]
                        end

                        nv[i] += nv[j]       # both `nv[i]` and `nv[j]` are negated since they are in Lme
                        nv[j] = zero(V)
                        elen[j] = zero(V)
                        j = next[j]          # delete `j` from hash bucket
                        next[jlast] = j

                        @goto L220

                        @label L240

                        jlast = j            # `j` cannot be absorbed into `i`
                        j = next[j]

                        @goto L220
                    end

                    wflg += one(I)
                    i = next[i]

                    if !iszero(i)
                        @goto L200
                    end
                end
            end
        end

        p = pme1                             # restore degree lists and remove nonprincipal supervariable from element
        nleft = totel - nel

        for pme in pme1:pme2
            i = iw[pme]
            nvi = -nv[i]

            if ispositive(nvi)               # `i` is a principal variable in Lme; restore `nv[i]` to signify that `i` is principal
                nv[i] = nvi

                if degree[i] != n2           # compute the external degree (add size of current elem)
                    deg = degree[i]

                    if degree[i] + degme > nleft
                        rmf1 = convert(FLOAT, deg) * convert(FLOAT, deg - one(V) + twice(degme)) - convert(FLOAT, wf[i])
                        degree[i] = nleft - nvi
                        deg = degree[i]
                        rmf = convert(FLOAT, deg) * convert(FLOAT, deg - one(V)) - convert(FLOAT, degme - nvi) * convert(FLOAT, degme - nvi - one(V))
                        rmf = min(rmf, rmf1)
                    else
                        degree[i] += degme - nvi
                        rmf = convert(FLOAT, deg) * convert(FLOAT, deg - one(V) + twice(degme)) - convert(FLOAT, wf[i])
                    end

                    rmf /= convert(FLOAT, nvi + one(V))

                    if rmf < dummy
                        wf[i] = anint(E, rmf)
                    elseif rmf / convert(FLOAT, n) < dummy
                        wf[i] = anint(E, rmf / convert(FLOAT, n))
                    else
                        wf[i] = idummy
                    end

                    if !ispositive(wf[i])
                        wf[i] = one(E)
                    end

                    deg = wf[i]

                    if deg > norig
                        deg = min(div(deg - norig, pas) + norig, nbbuck)
                    end

                    inext = head[deg + one(V)]

                    if !iszero(inext)
                        last[inext] = i
                    end

                    next[i] = inext
                    last[i] = zero(V)
                    head[deg + one(V)] = i

                    if mindeg > deg          # save the new degree, and find the minimum degree
                        mindeg = deg
                    end
                end

                iw[p] = i                    # place the supervariable in the element pattern
                p += one(E)
            end
        end # L260:

        nv[me] = nvpiv + degme               # finalize the new element
        len[me] = p - pme1

        if iszero(len[me])                   # there is nothing left of the current pivot element
            pe[me] = zero(E)
            w[me] = zero(I)
        end

        if !iszero(newmem)                   # element was not constructed in place: deallocate part of it
            pfree = p
            mem = mem - newmem + len[me]
        end
    end                                      # end while (selecting pivots)

    if ispositive(nbflag)                    # begin halo v2
        deg = mindeg - one(V); me = zero(V)

        @inbounds while deg < nbbuck + one(V) && !ispositive(me)
            deg += one(V); me = head[deg + one(V)]
        end

        mindeg = deg
        nelme = -one(V) - nel

        @inbounds for x in oneto(n)
            if ispositive(pe[x]) && isnegative(elen[x]) # `x` is an unabsorbed element
                pe[x] = convert(E, -me)
            elseif degree[x] == n2          # `x` is a dense row, absorb it in ME (mass elimination)
                nel += nv[x]
                pe[x] = convert(E, -me)
                elen[x] = zero(V)
                nv[x] = zero(V)             # patch 12/12/98 <PA+FP> (old: `n + 1`)
            end
        end

        @inbounds elen[me] = nelme          # `me` is the root node
        @inbounds nv[me] = n - nreal        # correct value of `nv` is principal variable = `nbflag`
        @inbounds pe[me] = zero(E)
    end

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
                    elen[j] = k
                    k += one(V)
                end

                j = jnext
            end

            elen[e] = -k
        end
    end                                     # L290:

    return ncmpa
end

end
