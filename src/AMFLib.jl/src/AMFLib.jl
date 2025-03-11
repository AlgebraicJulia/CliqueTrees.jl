# Copyright (C) 2002, International Business Machines
# Corporation and others.  All Rights Reserved.
# This code is licensed under the terms of the Eclipse Public License (EPL).

#==============================================================================#
#=      Ordering code - courtesy of Anshul Gupta                              =#
#=      (C) Copyright IBM Corporation 1997, 2009.  All Rights Reserved.       =#
#==============================================================================#

#=  A compact no-frills Approximate Minimum Local Fill ordering code.

    References:

[1] Ordering Sparse Matrices Using Approximate Minimum Local Fill.
    Edward Rothberg, SGI Manuscript, April 1996.
[2] An Approximate Minimum Degree Ordering Algorithm.
    T. Davis, P. Amestoy, and I. Duff, TR-94-039, CIS Department,
    University of Florida, December 1994.
=#
#===============================================================================#

module AMFLib

using Base: oneto, @propagate_inbounds

export amf

const MAXIW = typemax(Int) - 100000

function amf(
    xadj::AbstractVector{E}, adjncy::AbstractVector{V}; speed::Integer=1
) where {V,E}
    EE = promote_type(Int32, E)
    nnz = length(adjncy)
    neqns = convert(V, length(xadj) - 1)
    adjln = convert(EE, 2nnz + 4neqns + 10000)
    locaux = convert(EE, nnz + 1)

    return amf!(
        neqns,
        adjln,
        Vector{EE}(xadj),
        copyto!(Vector{V}(undef, adjln), 1, adjncy, 1, nnz),
        locaux,
        Val(Int(speed)),
    )
end

"""
    amf!(neqns, adjln, xadj, adjncy, locaux, speed)

A compact no-frills Approximate Minimum Local Fill ordering code.

input parameters:

  - `neqns`: number of equations

  - `adjln`: length of `adjncy`
  - `(xadj, adjncy)`: adjacency structure
  - `locaux`: first free index in `adjncy`
  - `speed`: fill approximation strategy

      + `Val(1)`: slow
      + `Val(2)`: medium
      + `Val(3)`: fast

output parameters:

  - `(perm, invp)`: the approximate minimum fill ordering
"""
function amf!(
    neqns::V,
    adjln::E,
    xadj::AbstractVector{E},
    adjncy::AbstractVector{V},
    locaux::E,
    speed::Val,
) where {V,E}
    dgree = Vector{V}(undef, neqns)
    head = Vector{V}(undef, neqns)
    snxt = Vector{V}(undef, neqns)
    perm = Vector{V}(undef, neqns)
    invp = Vector{V}(undef, neqns)
    varbl = Vector{V}(undef, neqns)
    lsize = Vector{V}(undef, neqns)
    flag = Vector{Int}(undef, neqns)

    for i in oneto(neqns)
        lsize[i] = dgree[i] = xadj[i + one(V)] - xadj[i]
        head[i] = invp[i] = perm[i] = snxt[i] = zero(V)
        varbl[i] = one(V)
        flag[i] = 1
    end

    erscore = initscore(dgree, speed)

    maxmum = zero(V)
    mindeg = one(V)
    fltag = 2
    nm1 = neqns - one(V)
    counter = one(V)

    @inbounds for l in oneto(neqns)
        j = erscore[l]

        if ispositive(j)
            nnode = head[j]

            if !iszero(nnode)
                perm[nnode] = l
            end

            snxt[l] = nnode
            head[j] = l
        else
            invp[l] = -counter
            counter += one(V)
            flag[l] = 0
            xadj[l] = zero(E)
        end
    end

    @inbounds while counter <= neqns
        deg = mindeg

        while !ispositive(head[deg])
            deg += one(V)
        end

        nodeg = zero(V)
        node = head[deg]
        mindeg = deg
        nnode = snxt[node]

        if !iszero(nnode)
            perm[nnode] = zero(V)
        end

        head[deg] = nnode
        nodeln = invp[node]
        numpiv = varbl[node]
        invp[node] = -counter
        counter += numpiv
        varbl[node] = -numpiv

        if !iszero(nodeln)
            j4 = locaux
            i5 = lsize[node] - nodeln
            i2 = nodeln + one(V)
            l = xadj[node]

            for i6 in oneto(i2)
                if i6 == i2
                    tn = node
                    i0 = l
                    scln = i5
                else
                    tn = adjncy[l]
                    l += one(E)
                    i0 = xadj[tn]
                    scln = lsize[tn]
                end

                for i7 in oneto(scln)
                    i = adjncy[i0]
                    i0 += one(E)
                    numii = varbl[i]

                    if ispositive(numii)
                        if locaux > adjln
                            lsize[node] -= i6
                            xadj[node] = l

                            if iszero(lsize[node])
                                xadj[node] = zero(E)
                            end

                            xadj[tn] = i0
                            lsize[tn] = scln - i7

                            if iszero(lsize[tn])
                                xadj[tn] = zero(E)
                            end

                            for j in oneto(neqns)
                                i4 = xadj[j]

                                if ispositive(i4)
                                    xadj[j] = adjncy[i4]
                                    adjncy[i4] = -j
                                end
                            end

                            i9 = j4 - one(E)
                            j6 = one(E)
                            j7 = one(E)

                            while j6 <= i9
                                j = -adjncy[j6]
                                j6 += one(E)

                                if ispositive(j)
                                    adjncy[j7] = xadj[j]
                                    xadj[j] = j7
                                    j7 += one(E)
                                    j8 = convert(E, lsize[j]) - one(E) + j7

                                    while j7 < j8
                                        adjncy[j7] = adjncy[j6]
                                        j7 += one(E)
                                        j6 += one(E)
                                    end
                                end
                            end

                            j0 = j7

                            for j4_ in j4:(locaux - one(E))
                                adjncy[j7] = adjncy[j4_]
                                j7 += one(E)
                            end

                            j4 = j0
                            locaux = j7
                            i0 = xadj[tn]
                            l = xadj[node]
                        end

                        adjncy[locaux] = i
                        locaux += one(E)
                        varbl[i] = -numii
                        nodeg += numii
                        ipp = perm[i]
                        nnode = snxt[i]

                        if !iszero(ipp)
                            snxt[ipp] = nnode
                        else
                            head[erscore[i]] = nnode
                        end

                        if !iszero(nnode)
                            perm[nnode] = ipp
                        end
                    end
                end

                if tn != node
                    flag[tn] = 0
                    xadj[tn] = -node
                end
            end

            j5 = locaux
            currloc = convert(V, j5 - j4)

        else
            j4 = xadj[node]
            i1 = j4 + lsize[node]
            j5 = j4

            for jj in j4:(i1 - one(E))
                i = adjncy[jj]
                numii = varbl[i]

                if ispositive(numii)
                    nodeg += numii
                    varbl[i] = -numii
                    adjncy[j5] = i
                    ipp = perm[i]
                    nnode = snxt[i]
                    j5 += one(E)

                    if !iszero(ipp)
                        snxt[ipp] = nnode
                    else
                        head[erscore[i]] = nnode
                    end

                    if !iszero(nnode)
                        perm[nnode] = ipp
                    end
                end
            end

            currloc = zero(V)
        end

        xadj[node] = j4
        lsize[node] = j5 - j4
        dgree[node] = nodeg

        if maxmum < nodeg
            maxmum = nodeg
        end

        if isnegative(fltag + neqns) || fltag + neqns > MAXIW
            for i in oneto(neqns)
                if !iszero(flag[i])
                    flag[i] = 1
                end
            end

            fltag = 2
        end

        for j3 in j4:(j5 - one(E))
            i = adjncy[j3]
            j = invp[i]

            if ispositive(j)
                numii = -varbl[i]
                i4 = fltag - numii
                ii2 = xadj[i] + convert(E, j)

                for l in xadj[i]:(ii2 - one(E))
                    tn = adjncy[l]
                    j9 = flag[tn]

                    if j9 >= fltag
                        j9 -= convert(Int, numii)
                    elseif !iszero(j9)
                        j9 = convert(Int, dgree[tn]) + convert(Int, i4)
                    end

                    flag[tn] = j9
                end
            end
        end

        for j3 in j4:(j5 - one(E))
            i = adjncy[j3]
            i5 = deg = zero(V)

            i4 = xadj[i]
            j1 = i4 + convert(E, invp[i])
            j0 = i4

            for l in i4:(j1 - one(E))
                tn = adjncy[l]
                jjj8 = flag[tn]

                if !iszero(jjj8)
                    deg += convert(V, jjj8 - fltag)
                    adjncy[i4] = tn
                    i5 += tn
                    i4 += one(E)

                    while i5 >= nm1
                        i5 -= nm1
                    end
                end
            end

            j2 = i4
            invp[i] = j2 - j0 + one(E)
            ii2 = j0 + lsize[i]

            for l in j1:(ii2 - one(E))
                j = adjncy[l]
                numii = varbl[j]

                if ispositive(numii)
                    deg += numii
                    adjncy[i4] = j
                    i5 += j
                    i4 += one(E)

                    while i5 >= nm1
                        i5 -= nm1
                    end
                end
            end

            if isone(invp[i]) && j2 == i4
                numii = -varbl[i]
                xadj[i] = -node
                nodeg -= numii
                counter += numii
                numpiv += numii
                invp[i] = zero(V)
                varbl[i] = zero(V)
            else
                if dgree[i] > deg
                    dgree[i] = deg
                end

                adjncy[i4] = adjncy[j2]
                adjncy[j2] = adjncy[j0]
                adjncy[j0] = node
                lsize[i] = i4 - j0 + one(E)
                i5 += one(V)

                j = head[i5]

                if ispositive(j)
                    snxt[i] = perm[j]
                    perm[j] = i
                else
                    snxt[i] = -j
                    head[i5] = -i
                end

                perm[i] = i5
            end
        end

        dgree[node] = nodeg

        if maxmum < nodeg
            maxmum = nodeg
        end

        fltag += maxmum

        for j3 in j4:(j5 - one(E))
            i = adjncy[j3]

            if isnegative(varbl[i])
                i5 = perm[i]
                j = head[i5]

                if !iszero(j)
                    if isnegative(j)
                        head[i5] = zero(V)
                        i = -j
                    else
                        i = perm[j]
                        perm[j] = zero(V)
                    end

                    while !iszero(i)
                        if iszero(snxt[i])
                            i = zero(V)
                        else
                            k = invp[i]
                            scln = lsize[i]
                            ii2 = xadj[i] + convert(E, scln)

                            for l in (xadj[i] + one(E)):(ii2 - one(E))
                                flag[adjncy[l]] = fltag
                            end

                            jpp = i
                            j = snxt[i]

                            while !iszero(j)
                                if lsize[j] == scln && invp[j] == k
                                    ii2 = xadj[j] + convert(E, scln)
                                    jj8 = true

                                    for l in (xadj[j] + one(E)):(ii2 - one(E))
                                        if flag[adjncy[l]] != fltag
                                            jj8 = false
                                            break
                                        end
                                    end

                                    if jj8
                                        xadj[j] = -i
                                        varbl[i] += varbl[j]
                                        varbl[j] = zero(V)
                                        invp[j] = zero(V)

                                        j = snxt[j]
                                        snxt[jpp] = j
                                    else
                                        jpp = j
                                        j = snxt[j]
                                    end
                                else
                                    jpp = j
                                    j = snxt[j]
                                end
                            end

                            fltag += 1
                            i = snxt[i]
                        end
                    end
                end
            end
        end

        jjjj8 = nm1 - counter
        jj = j4

        for j3 in j4:(j5 - one(E))
            i = adjncy[j3]
            numii = varbl[i]

            if isnegative(numii)
                varbl[i] = abs(numii)
                j9 = dgree[i] + nodeg
                deg = min(jjjj8, j9) + numii

                if !ispositive(deg)
                    deg = one(V)
                end

                dgree[i] = deg
                deg = score(xadj, adjncy, dgree, invp, deg, i, speed)
                erscore[i] = deg

                nnode = head[deg]

                if !iszero(nnode)
                    perm[nnode] = i
                end

                snxt[i] = nnode
                perm[i] = zero(V)

                head[deg] = i
                adjncy[jj] = i
                jj += one(E)

                if deg < mindeg
                    mindeg = deg
                end
            end
        end

        if !iszero(currloc)
            locaux = jj
        end

        varbl[node] = numpiv + nodeg
        lsize[node] = jj - j4

        if iszero(lsize[node])
            flag[node] = 0
            xadj[node] = zero(E)
        end
    end

    @inbounds for l in oneto(neqns)
        if iszero(invp[l])
            i = convert(V, -xadj[l])

            while !isnegative(invp[i])
                i = convert(V, -xadj[i])
            end

            tn = i
            k = -invp[tn]
            i = l

            while !isnegative(invp[i])
                nnode = convert(V, -xadj[i])
                xadj[i] = -tn

                if iszero(invp[i])
                    invp[i] = k
                    k += one(V)
                end

                i = nnode
            end

            invp[tn] = -k
        end
    end

    @inbounds for l in oneto(neqns)
        i = abs(invp[l])
        invp[l] = i
        perm[i] = l
    end

    return perm, invp
end

@propagate_inbounds function score(
    xadj::Vector{E},
    adjncy::Vector{V},
    dgree::Vector{V},
    invp::Vector{V},
    deg::V,
    i::V,
    ::Val{1},
) where {V,E}
    @boundscheck checkbounds(xadj, i)
    @boundscheck checkbounds(invp, i)
    @inbounds l = xadj[i]
    @inbounds i4 = l + convert(E, invp[i])

    k = zero(V)

    @inbounds while l < i4
        i5 = dgree[adjncy[l]]

        if k < i5
            k = i5
        end

        l += one(E)
    end

    x = Float64(k - one(V))
    y = Float64(deg)
    y = y * y - y
    x = y - (x * x - x)

    if x < 1.1
        x = 1.1
    end

    return trunc(V, sqrt(x))
end

@propagate_inbounds function score(
    xadj::AbstractVector{E},
    adjncy::AbstractVector{V},
    dgree::AbstractVector{V},
    invp::AbstractVector{V},
    deg::V,
    i::V,
    ::Val{2},
) where {V,E}
    @boundscheck checkbounds(xadj, i)
    @inbounds x = Float64(dgree[adjncy[xadj[i]]] - one(V))
    y = Float64(deg)
    y = y * y - y
    x = y - (x * x - x)

    if x < 1.1
        x = 1.1
    end

    return trunc(V, sqrt(x))
end

function score(
    xadj::AbstractVector{E},
    adjncy::AbstractVector{V},
    dgree::AbstractVector{V},
    invp::AbstractVector{V},
    deg::V,
    i::V,
    ::Val{3},
) where {V,E}
    return deg
end

function initscore(dgree::AbstractVector, ::Union{Val{1},Val{2}})
    return copy(dgree)
end

function initscore(dgree::AbstractVector, ::Val{3})
    return dgree
end

function ispositive(i::Integer)
    return i > zero(i)
end

function isnegative(i::Integer)
    return i < zero(i)
end

function istwo(i::Integer)
    return i == two(i)
end

function two(i::Integer)
    return one(i) + one(i)
end

end
