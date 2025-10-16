function pr3(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return pr3(weights, graph, convert(W, width))
end

function pr3(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V}
    @assert nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = zero(W)

    @inbounds for v in oneto(n)
        totdeg += weights[v]
    end

    marker = FVector{V}(undef, n)    
    stack0 = FVector{V}(undef, n)
    stack1 = FVector{V}(undef, n)
    stack2 = FVector{V}(undef, n)
    stack3 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    stack5 = FVector{V}(undef, n)
    index0 = FVector{V}(undef, n)
    index1 = FVector{V}(undef, n)
    index2 = FVector{V}(undef, n)
    index3 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)

    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    source = FVector{V}(undef, m)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    invptr = FVector{E}(undef, m)

    kernel, stack, inject, width = pr3_impl!(marker, stack0, stack1, stack2, stack3, stack4, stack5, 
        index0, index1, index2, index3, tmpptr, degree, number, source,
        target, begptr, endptr, invptr, totdeg, weights, graph, width)

    return kernel, stack, inject, width
end

"""
    pr3_impl!(marker, stack0, stack1, stack2, stack3, stack4,
        stack5, index0, index1, index2, index3, tmpptr, degree,
        number, source, target, begptr, endptr, invptr, totdeg,
        weight, graph, width)

Pre-processing for Triangulation of Probabilistic Networks
Bodlaender, Koster, Eijkhof, and van der Gaag

Preprocessing Rules for Triangulation of Probabilistic Networks
Bodlaender, Koster, Eijkhof, and van der Gaag

Safe Reduction Rules for Weighted Treewidth
Eijkhof, Bodlaender, and Koster

Preprocess a graph by applying a set of *safe* reduction
rules.
  - islet
  - twig
  - series
  - triangle
  - buddy
  - cube
The algorithm runs in a loop, testing each degree 0, 1, 2,
and 3 vertex to see if it can be eliminated according to one
of the rules. The algorithm terminates when no more vertices
can be eliminated.

The output is a reduced graph R and a sequence (v₁, ..., vₙ) of
eliminated vertices. Any minimum-treewidth elimination ordering
(w₁, ..., wₘ) of R can be appended to the sequence to create
a minimum-treewidth elimination ordering of the input graph.
   (v₁, ..., vₙ, w₁, ..., wₘ).
If the input graph has treewidth at most three, then the
algorithm will eliminate every vertex, and the reduced graph
R will have no vertices.

The algorithm employs a data structure called a *quotient graph.*
It is a directed graph with two types of vertices: elements and
supernodes. The arcs in the graph obey the following invariant
  - every supernode w has at most one predecessor v, which must
    be an element
We say that an element w is *reachable* by another element v if
there exists a path (v, x₁, ..., xₙ, w) from v to w through
supernodes {x₁, ..., xₙ}. Reachability is a symmetric, irreflexive
relation on the set of elements.

input parameters:
  - `totdeg`: total vertex weight
  - `weight`: vertex weight
  - `graph`: input graph
  - `width`: treewidth lower bound

output parameters:
  - `kernel`: reduced graph
  - `stack4`: stack of eliminated vertices
  - `inject`: mapping from the vertices of `kernel` to the vertices
              of `graph`
  - `width`: treewidth lower bound

working arrays:
  - miscellaneous:
    - `marker`: marker array
    - `stack5`: vertex stack
    - `tmpptr`: temporary pointer array
  - vertex-degree bucket queue:
    - `stack0`: stack of degree 0 vertices
    - `stack1`: stack of degree 1 vertices
    - `stack2`: stack of degree 2 vertices
    - `stack3`: stack of degree 3 vertices
    - `index0`: the index of a vertex in `stack0`
    - `index1`: the index of a vertex in `stack1`
    - `index2`: the index of a vertex in `stack2`
    - `index3`: the index of a vertex in `stack3`
  - quotient graph:
    - `source`: the source vertex of an edge
    - `target`: the target vertex of an edge
      - positive vertices are elements
      - negative vertices are supernodes
    - `begptr`: the first edge incident to a vertex
    - `endptr`: the final edge incident to a vertex
    - `invptr`: the reverse of an edge
"""
function pr3_impl!(
        marker::AbstractVector{V},
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        index0::AbstractVector{V},
        index1::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        tmpptr::AbstractVector{E},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        totdeg::W,
        weight::AbstractVector{W},
        graph::AbstractGraph{V},
        width::W,
    ) where {W, V, E}
    @assert nv(graph) <= length(marker)
    @assert nv(graph) <= length(stack0)
    @assert nv(graph) <= length(stack1)
    @assert nv(graph) <= length(stack2)
    @assert nv(graph) <= length(stack3)
    @assert nv(graph) <= length(stack4)
    @assert nv(graph) <= length(stack5)
    @assert nv(graph) <= length(index0)
    @assert nv(graph) <= length(index1)
    @assert nv(graph) <= length(index2)
    @assert nv(graph) <= length(index3)
    @assert nv(graph) < length(tmpptr)
    @assert nv(graph) <= length(degree)
    @assert nv(graph) <= length(number)
    @assert de(graph) <= length(source)
    @assert de(graph) <= length(target)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) <= length(endptr)
    @assert de(graph) <= length(invptr)

    # `n` is the number of vertices in the input graph
    n = nv(graph)

    # initialize vertex-degree bucket queue and quotient graph
    hi0, hi1, hi2, hi3, mindeg = pr3_init!(marker, stack0, stack1, stack2,
        stack3, index0, index1, index2, index3, tmpptr, degree, number,
        source, target, begptr, endptr, invptr, totdeg, weight, graph)

    # the weighted treewidth of the input graph is no less
    # than its minimum weighted degree
    width = max(width, mindeg)

    # `tag` is used to mark vertices
    tag = one(V)

    # `lo4` is the previous number of eliminated vertices
    lo4 = -one(V)

    # `hi4` is the current number of eliminated vertices
    hi4 = zero(V)

    # if a vertex was eliminated during the previous loop... 
    @inbounds while lo4 < hi4
        # update `lo4`
        lo4 = hi4

        # test every degree 0 vertex
        width, hi4 = pr3_0!(stack0, stack4, degree, number, width, hi0, hi4)

        # every degree 0 vertex has been eliminated
        hi0 = zero(V)

        # test every degree 1 vertex
        width, hi0, hi2, hi3, hi4 = pr3_1!(stack0, stack1, stack2, stack3,
            stack4, stack5, index0, index1, index2, index3, degree, number, source,
            target, begptr, endptr, invptr, weight, width, hi0, hi1, hi2, hi3, hi4)

        # every degree 1 vertex has been eliminated
        hi1 = zero(V)

        # test every degree 2 vertex
        width, hi0, hi1, hi2, hi3, hi4 = pr3_2!(stack0, stack1,
            stack2, stack3, stack4, stack5, index0, index1, index2, index3,
            degree, number, source, target, begptr, endptr, invptr, weight,
            width, hi0, hi1, hi2, hi3, hi4)

        # test every degree 3 vertex
        width, tag, hi0, hi1, hi2, hi3, hi4 = pr3_3!(marker, stack0, stack1,
            stack2, stack3, stack4, stack5, index0, index1, index2, index3,
            degree, number, source, target, begptr, endptr, invptr, weight,
            width, tag, hi0, hi1, hi2, hi3, hi4)
    end

    # construct the reduced graph R = (V, E):
    #  - V is the set of elements in the quotient graph
    #  - E contains an arc (v, w) if w is reachable by v in the quotient graph
    m, n = pr3_make!(stack4, stack5, target, begptr, endptr,
        invptr, stack0, number, tmpptr, source, hi4, n)

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(n, n, m, tmpptr, source)
    return kernel, stack4, stack0, width
end

function pr3_make!(
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        inj::AbstractVector{V},
        prj::AbstractVector{V},
        ptr::AbstractVector{E},
        tgt::AbstractVector{V},
        hi4::V,
        n::V,
    ) where {V, E}

    # mark eliminated vertices with -1
    @inbounds for i in oneto(hi4)
        w = stack4[i]; prj[w] = -one(V)
    end

    # `v` is a vertex in the reduced graph
    v = one(V)

    # for every vertex `w`...
    @inbounds for w in oneto(n)

        # if `w` was not eliminated
        if !isnegative(prj[w])
            # associate `v` and `w`
            prj[w] = v
            inj[v] = w

            # increment `v`
            v += one(V)
        end
    end

    # `v` is a vertex in the reduced graph
    v = one(V)

    # `p` is the first arc incident to `v`
    @inbounds ptr[v] = p = one(E)

    # for all vertices `v` in the reduced graph...
    @inbounds while v + hi4 <= n
        # `w` is the corresponding element in the
        # quotient graph
        w = inj[v]

        p =  pr3_reach!(p, stack5,
            target, begptr, endptr, invptr, w) do p, _, x

            # `x` is an element reachable by `w`
            @inbounds tgt[p] = prj[x]; p += one(E)
            return p
        end

        # update `v` and `p`
        v += one(V); ptr[v] = p
    end

    # `m` is the number of arcs in the reduced graph
    m = p - one(E)

    # `n` is the number of vertices in the reduced graph
    n = v - one(V)
    return m, n
end

function pr3_init!(
        marker::AbstractVector{V},
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        index0::AbstractVector{V},
        index1::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        tmpptr::AbstractVector{E},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        totdeg::W,
        weight::AbstractVector{W},
        graph::AbstractGraph{V},
    ) where {W, V, E}
    
    # `n` is the number of vertices in the graph
    n = nv(graph); nn = n + one(V)

    # `hi0` is the top of the stack of degree 0 vertices
    # `hi1` is the top of the stack of degree 1 vertices
    # `hi2` is the top of the stack of degree 2 vertices
    # `hi3` is the top of the stack of degree 3 vertices    
    hi0 = hi1 = hi2 = hi3 = zero(V)
    
    # `mindeg` is the minimum weighted degree
    mindeg = totdeg
    
    # `p` is the current arc
    p = one(E)

    @inbounds for v in vertices(graph)
        marker[v] = zero(V)
        tmpptr[v] = begptr[v] = endptr[v] = p

        # `deg` is the weighted degree of `v`
        deg = weight[v]
        
        # `num` is the unweighted degree of `v`
        num = zero(V)

        # for all neighbors `w` of `v`...
        for w in neighbors(graph, v)
            # ignore self loops
            if v != w
                # `p` is the arc (`v`, `w`)
                source[p] = v; p += one(E)

                # increase the weighted degree of `v` by
                # the weight of `w`
                deg += weight[w]

                # increment the degree of `v`
                num += one(V)
            end
        end
        
        # if `v` has degree 0 ...
        if iszero(num)
            # ... add it to the stack of degree 0 vertices
            index0[v] = hi0 += one(V); stack0[hi0] = v

        # if `v` has degree 1 ...
        elseif isone(num)
            # ... add it to the stack of degree 1 vertices
            index1[v] = hi1 += one(V); stack1[hi1] = v

        # if `v` has degree 2 ...
        elseif istwo(num)
            # ... add it to the stack of degree 2 vertices
            index2[v] = hi2 += one(V); stack2[hi2] = v

        # if `v` has degree 3 ...
        elseif isthree(num)
            # ... add it to the stack of degree 3 vertices
            index3[v] = hi3 += one(V); stack3[hi3] = v
        end

        # update the minimum weighted degree
        mindeg = min(mindeg, deg)

        # store the weighted degree of `v`
        degree[v] = deg

        # store the degree of `v`
        number[v] = num
    end

    # for all arcs (`v`, `w`)...
    @inbounds for v in vertices(graph), w in neighbors(graph, v)
        # ignore self loops
        if v != w
            # `q` is the arc (`w`, `v`)
            q = endptr[w]; target[q] = v; endptr[w] = q + one(E)
        end
    end

    # for all vertices `v`...
    @inbounds for v in vertices(graph)
        # the arcs {`p`, ..., `pend` - 1} are incident
        # to `v`
        p = begptr[v]; pend = endptr[v]

        while p < pend
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `q` is the arc (`w`, `v`)
            q = tmpptr[w]; invptr[p] = q; tmpptr[w] = q + one(E)
            p += one(E)
        end
    end

    if ispositive(n)
        @inbounds begptr[nn] = endptr[n]
    end

    return hi0, hi1, hi2, hi3, mindeg
end

function pr3_0!(
        stack0::AbstractVector{V},
        stack4::AbstractVector{V},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        width::W,
        hi0::V,
        hi4::V,
    ) where {W, V}

    # for all elements with degree 0...
    @inbounds while ispositive(hi0)
        # `v` is an element with degree 0
        hi0, v = pr3_stack_pop!(stack0, hi0)

        # add `v` to the stack of eliminated vertices
        hi4 = pr3_stack_add!(stack4, hi4, v)

        # `v` is simplicial: update the lower bound
        width = max(width, degree[v])
    end
    
    return width, hi4
end

function pr3_1!(
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        index0::AbstractVector{V},
        index1::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        width::W,
        hi0::V,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
    ) where {W, V, E}
    
    # for all elements with degree 1...
    @inbounds while ispositive(hi1)
        # `v` is an element with degree 1
        hi1, v = pr3_stack_pop!(stack1, hi1)

        # add `v` to the stack of eliminated vertices
        hi4 = pr3_stack_add!(stack4, hi4, v)

        # `v` is simplicial: update the lower bound
        width = max(width, degree[v])
        
        # `w` is the unique element reachable by `v`
        w = zero(V)
        p = zero(E)

        p, w = pr3_reach!((p, w), stack5,
            target, begptr, endptr, invptr, v) do _, p, w
            return (p, w)
        end
                
        if isone(number[w])
            # if `w` has degree 1, remove it from the stack
            # of degree 1 elements
            hi1 = pr3_stack_del!(stack1, index1, hi1, w)

            # add `w` to the stack of degree 0 elements
            hi0 = pr3_stack_add!(stack0, index0, hi0, w)
        elseif istwo(number[w])
            # if `w` has degree 2, remove it from the stack
            # of degree 2 elements
            hi2 = pr3_stack_del!(stack2, index2, hi2, w)

            # add `w` to the stack of degree 1 elements
            hi1 = pr3_stack_add!(stack1, index1, hi1, w)
        elseif isthree(number[w])
            # if `w` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, w)

            # add `w` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, w)
        elseif isfour(number[w])
            # if `w` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, w)
        end

        # remove `v` from the reachable set of `w`
        pr3_reach_del!(source, target, endptr, invptr, invptr[p])

        # decrement the degree of `w`
        number[w] -= one(V)

        # decrease the weighted degree of `w` by the weight
        # of `v`
        degree[w] -= weight[v]
    end
    
    return width, hi0, hi2, hi3, hi4
end

function pr3_2!(
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        index0::AbstractVector{V},
        index1::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        width::W,
        hi0::V,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
    ) where {W, V, E}
    tol = tolerance(W)
    
    # `i` is an index into the stack of degree 2 elements
    i = one(V)

    # for all elements with degree 2    
    @inbounds while i <= hi2
        # `v` is an element with degree 2
        v = stack2[i]
       
        # `w` and `ww` are the elements reachable by `v` 
        w = ww = zero(V)
        p = pp = zero(E)
        
        p, w, pp, ww = pr3_reach!((p, w, pp, ww),
            stack5, target, begptr, endptr, invptr, v) do (p, w, pp, ww), ppp, www
            
            if iszero(w)
                p, w = ppp, www
            else
                pp, ww = ppp, www
            end
            
            return (p, w, pp, ww)    
        end

        # sort `w` and `ww` by degree        
        if number[ww] < number[w]
            p, w, pp, ww = pp, ww, p, w
        end

        # if `flag` is zero, then `ww` is reachable by `w`
        flag = pr3_reach!(ww, stack5,
            target, begptr, endptr, invptr, w) do ww, qq, xx
            
            if ww == xx
                ww = zero(V)
            end

            return ww
        end
        
        if iszero(flag)
            # w ─── ww
            # │  ╱
            # v

            # remove `v` from the stack of degree 2 elements
            hi2 = pr3_stack_del!(stack2, index2, hi2, v)

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_stack_add!(stack4, hi4, v)

            # `v` is simplicial: update the lower bound
            width = max(width, degree[v])

            if istwo(number[w])
                # if `w` has degree 2, remove it from the stack
                # of degree 2 elements
                hi2 = pr3_stack_del!(stack2, index2, hi2, w)

                # add `w` to the stack of degree 1 elements
                hi1 = pr3_stack_add!(stack1, index1, hi1, w)
            elseif isthree(number[w])
                # if `w` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, w)

                # add `w` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, w)
            elseif isfour(number[w])
                # if `w` has degree 4, add it to the stack
                # of degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, w)
            end

            if istwo(number[ww])
                # if `ww` has degree 2, remove it from the stack
                # of degree 2 elements
                hi2 = pr3_stack_del!(stack2, index2, hi2, ww)

                # add `ww` to the stack of degree 1 elements
                hi1 = pr3_stack_add!(stack1, index1, hi1, ww)
            elseif isthree(number[ww])
                # if `ww` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, ww)

                # add `ww` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, ww)
            elseif isfour(number[ww])
                # if `ww` has degree 4, add it to the stack
                # of degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, ww)
            end

            # remove `v` from the reachable sets of `w` and `ww`
            pr3_reach_del!(source, target, endptr, invptr, invptr[p])
            pr3_reach_del!(source, target, endptr, invptr, invptr[pp])

            # decrement the degree of `w` and `ww`
            number[w] -= one(V)
            number[ww] -= one(V)

            # decrease the weighted degree of `w` and `ww` by the
            # weight of `v`
            degree[w] -= weight[v]
            degree[ww] -= weight[v]
        elseif degree[v] < width + tol && min(weight[w], weight[ww]) < weight[v] + tol
            # w     ww
            # │  ╱
            # v

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_stack_add!(stack4, hi4, v)

            # remove `v` from the stack of degree 2 vertices
            hi2 = pr3_stack_del!(stack2, index2, hi2, v)

            pinv = invptr[p]
            ppinv = invptr[pp]
            
            # replace `v` with `ww` in the reachable set of `w`
            target[pinv] = ww; invptr[pinv] = ppinv

            # replace `v` with `w` in the reachable set of `ww`
            target[ppinv] = w; invptr[ppinv] = pinv

            # increase the weighted degree of `w` by the weight
            # of `ww` and decrease it by the weight of `v`          
            degree[w] -= (weight[v] - weight[ww])

            # increase the weighted degree of `ww` by the weight
            # of `w` and decrease it by the weight of `v`          
            degree[ww] -= (weight[v] - weight[w])
        else
            i += one(V)
        end
    end
    
    return width, hi0, hi1, hi2, hi3, hi4
end

function pr3_3!(
        marker::AbstractVector{V},
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        index0::AbstractVector{V},
        index1::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        width::W,
        tag::V,
        hi0::V,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `i` is an index into the stack of degree 3 elements
    i = one(V)

    # for all elements with degree 3...    
    @inbounds while i <= hi3
        # `v` is an element with degree 3
        v = stack3[i]

        # `w`, `ww`, and `www` are the elements reachable by `v`
        p, w, pp, ww, ppp, www = pr3_3_reach!(stack5,
            target, begptr, endptr, invptr, v)

        # sort `w`, `ww`, and `www` by degree
        (p, w), (pp, ww), (ppp, www) = sortthree((p, w), (pp, ww), (ppp, www)) do (p, w)
            @inbounds wnum = number[w]
            return wnum
        end

        # mark the elements reachable by `w` and not `ww` with `tag`
        # mark the elements reachable by `w` and `ww` with `tag` + 1
        # mark the elements reachable by `ww` and not `w` with `tag` + 2
        pr3_3_mark!(marker, stack5, target,
            begptr, endptr, invptr, tag, w, ww)

        # `f` is true if `ww` is reachable by `w`
        f = marker[ww] == tag

        # `ff` is true if `w` is reachable by `www`
        ff = tag <= marker[www] <= tag + one(V)

        # `fff` is true if `www` is reachable by `ww`
        fff = tag + one(V) <= marker[www]

        # sort `f`, `ff`, and `fff` by true value
        if f
            pp, ppp = ppp, pp
            ww, www = www, ww
            f, ff = ff, f
        end
        
        if ff
            p, pp = pp, p
            w, ww = ww, w
            ff, fff = fff, ff
        end
        
        if f
            pp, ppp = ppp, pp
            ww, www = www, ww
            f, ff = ff, f
        end
        
        if f
            # w ─── ww
            # │  ╳  │
            # v ─── www

            # remove `v` from the stack of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, v)

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_stack_add!(stack4, hi4, v)

            # `v` is simplicial: update the lower bound
            width = max(width, degree[v])

            if isthree(number[w])
                # if `w` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, w)

                # add `w` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, w)
            elseif isfour(number[w])
                # if `w` has degree 4, add it to the stack of
                # degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, w)
            end

            if isthree(number[ww])
                # if `ww` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, ww)

                # add `ww` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, ww)
            elseif isfour(number[ww])
                # if `ww` has degree 4, add it to the stack of
                # degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, ww)
            end

            if isthree(number[www])
                # if `www` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, www)

                # add `www` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, www)
            elseif isfour(number[www])
                # if `www` has degree 4, add it to the stack of
                # degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, www)
            end

            # remove `v` from the reachable sets of `w`, `ww`, and `www`
            pr3_reach_del!(source, target, endptr, invptr, invptr[p])
            pr3_reach_del!(source, target, endptr, invptr, invptr[pp])
            pr3_reach_del!(source, target, endptr, invptr, invptr[ppp])

            # decrement the degrees of `w`, `ww`, and `www`
            number[w] -= one(V)
            number[ww] -= one(V)
            number[www] -= one(V)

            # decrease the weighted degrees of `w`, `ww`, and `www`
            # by the weight of `v`
            degree[w] -= weight[v]
            degree[ww] -= weight[v]
            degree[www] -= weight[v]
        elseif ff && degree[v] < width + tol && min(weight[w], weight[ww]) < weight[v] + tol
            # w     ww
            # │  ╳  │
            # v ─── www

            # remove `v` from the stack of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, v)

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_stack_add!(stack4, hi4, v)

            if isthree(number[www])
                # if `www` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, www)

                # add `www` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, www)
            elseif isfour(number[www])
                # if `www` has degree 4, add it to the stack of
                # degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, www)
            end

            # remove `v` from the reachable set of `www`
            pr3_reach_del!(source, target, endptr, invptr, invptr[ppp])

            # decrement the degree of `www`
            number[www] -= one(V)

            # decrease the weighted degree of `www` by the
            # weight of `v`
            degree[www] -= weight[v]
            
            pinv = invptr[p]
            ppinv = invptr[pp]
            
            # replace `v` with `ww` in the reachable set of `w`
            target[pinv] = ww; invptr[pinv] = ppinv

            # replace `v` with `w` in the reachable set of `ww`
            target[ppinv] = w; invptr[ppinv] = pinv

            # increase the weighted degree of `w` by the weight of
            # `ww` and decrease it by the weight of `v`        
            degree[w] -= (weight[v] - weight[ww])

            # increase the weighted degree of `ww` by the weight of
            # `w` and decrease it by the weight of `v`        
            degree[ww] -= (weight[v] - weight[w])
        elseif fff && degree[v] < width + tol && weight[w] < weight[v] + tol
            # w     ww
            # │  ╱  │
            # v ─── www

            # remove `v` from the stack of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, v)

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_stack_add!(stack4, hi4, v)

            if isone(number[w])
                # if `w` has degree 1, remove it from the stack
                # of degree 1 elements
                hi1 = pr3_stack_del!(stack1, index1, hi1, w)

                # add `w` to the stack of degree 2 elements
                hi2 = pr3_stack_add!(stack2, index2, hi2, w)
            elseif istwo(number[w])
                # if `w` has degree 2, remove it from the stack
                # of degree 2 elements
                hi2 = pr3_stack_del!(stack2, index2, hi2, w)

                # add `w` to the stack of degree 3 elements
                hi3 = pr3_stack_add!(stack3, index3, hi3, w)
            elseif isthree(number[w])
                # if `w` has degree 3, remove it from the stack
                # of degree 3 elements
                hi3 = pr3_stack_del!(stack3, index3, hi3, w)
            end
            
            # increment the degree of `w`
            number[w] += one(V)
            
            pinv = invptr[p]
            ppinv = invptr[pp]
            pppinv = invptr[ppp]
     
            # turn `v` into a supernode       
            target[pinv] = -v

            # replace `v` with `w` in the reachable sets of `ww` and `www`
            target[ppinv] = w
            target[pppinv] = w

            # increase the weighted degree of `w` by the weights of
            # `ww` and `www` and decrease it by the weight of `v`        
            degree[w] -= (weight[v] - weight[ww] - weight[www])

            # increase the weighted degree of `ww` by the weight
            # of `w` and decrease it by the weight of `v`
            degree[ww] -= (weight[v] - weight[w])

            # increase the weighted degree of `www` by the weight
            # of `w` and decrease it by the weight of `v`
            degree[www] -= (weight[v] - weight[w])

            # remove `w` from the reachable set of `v`
            pr3_reach_del!(source, target, endptr, invptr, p)
        else
            flag = false

            if !fff && degree[v] < width + tol
                # w     ww
                # │  ╱  
                # v ─── www

                # search for a buddy `vv`
                vv = pr3_buddy_reach!(marker, stack5, degree, number, target,
                    begptr, endptr, invptr, weight, width, tag, v, w, ww, www)

                # if a buddy was found ...
                if ispositive(vv)
                    # w ────────── vv
                    # │         ╱  │
                    # │     ww     │
                    # │  ╱         │
                    # v ────────── www

                    # remove `v` and `vv` from the stack of degree 3 elements
                    hi3 = pr3_stack_del!(stack3, index3, hi3, v)
                    hi3 = pr3_stack_del!(stack3, index3, hi3, vv)

                    # add `v` and `vv` to the stack of eliminated vertices
                    hi4 = pr3_stack_add!(stack4, hi4, v)
                    hi4 = pr3_stack_add!(stack4, hi4, vv)

                    # eliminate `v` and `vv`
                    pr3_buddy!(stack5, degree, source, target, begptr, endptr,
                        invptr, weight, v, vv, p, w, pp, ww, ppp, www)

                    flag = true
                end
            end

            if !fff && !flag && isthree(number[w]) && isthree(number[ww]) && isthree(number[www]) && max(degree[w], degree[ww], degree[www]) < width + tol
                # search for a cube `x`, `y`, and `z`
                q, qq, r, rr, s, ss, x, y, z = pr3_cube_reach!(stack5,
                    target, begptr, endptr, invptr, weight, v, w, ww, www)

                # if a cube was found...
                if ispositive(q)
                    #       x
                    #    ╱     ╲
                    # w           ww
                    # │  ╲     ╱  │
                    # │     v     │
                    # z     │     y
                    #    ╲  │  ╱
                    #      www

                    # `v` will be simplicial after eliminating `w`, `ww`, and `www`:
                    # update the lower bound
                    width = max(width, degree[v])

                    # remove `w`, `ww`, `www`, and `v` from the stack
                    # of degree 3 elements
                    hi3 = pr3_stack_del!(stack3, index3, hi3, w)
                    hi3 = pr3_stack_del!(stack3, index3, hi3, ww)
                    hi3 = pr3_stack_del!(stack3, index3, hi3, www)
                    hi3 = pr3_stack_del!(stack3, index3, hi3, v)

                    # add `w`, `ww`, `www`, and `v` to the stack of
                    # eliminated vertices
                    hi4 = pr3_stack_add!(stack4, hi4, w)
                    hi4 = pr3_stack_add!(stack4, hi4, ww)
                    hi4 = pr3_stack_add!(stack4, hi4, www)
                    hi4 = pr3_stack_add!(stack4, hi4, v)

                    # eliminate `w`, `ww`, `www`, and `v`
                    hi2, hi3 = pr3_cube!(marker, stack2, stack3, stack5, index2, index3, 
                        number, degree, source, target, begptr, endptr, invptr,
                        weight, tag, hi2, hi3, w, ww, www, q, qq, r, rr, s, ss, x, y, z)

                    flag = true
                end
            end

            if !flag
                i += one(V)
            end
        end

        tag += six(V)
    end

    return width, tag, hi0, hi1, hi2, hi3, hi4
end

function pr3_buddy!(
        stack5::AbstractVector{V},
        degree::AbstractVector{W},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        v::V,
        vv::V,
        p::E,
        w::V,
        pp::E,
        ww::V,
        ppp::E,
        www::V,
    ) where {W, V, E}
    # w ────────── vv
    # │         ╱  │
    # │     ww     │
    # │  ╱         │
    # v ────────── www
    q = qq = qqq = zero(E)

    q, qq, qqq = pr3_reach!((q, qq, qqq),
        stack5, target, begptr, endptr, invptr, vv) do (q, qq, qqq), qnxt, wnxt

        if w == wnxt
            q = qnxt
        elseif ww == wnxt
            qq = qnxt
        elseif www == wnxt
            qqq = qnxt
        end

        return (q, qq, qqq)
    end

    @inbounds pinv = invptr[p]
    @inbounds ppinv = invptr[pp]
    @inbounds pppinv = invptr[ppp]

    @inbounds qinv = invptr[q]
    @inbounds qqinv = invptr[qq]
    @inbounds qqqinv = invptr[qqq]

    # replace `v` with `ww` in the reachable set of `w`   
    @inbounds target[pinv] = ww
    @inbounds invptr[pinv] = qqinv

    # replace `v` with `www` in the reachable set of `ww`
    @inbounds target[ppinv] = www
    @inbounds invptr[ppinv] = qqqinv

    # replace `v` with `w` in the reachable set of `www`
    @inbounds target[pppinv] = w
    @inbounds invptr[pppinv] = qinv

    # replace `vv` with `www` in the reachable set of `w` 
    @inbounds target[qinv] = www
    @inbounds invptr[qinv] = pppinv

    # replace `vv` with `w` in the reachable set of `ww`
    @inbounds target[qqinv] = w
    @inbounds invptr[qqinv] = pinv

    # replace `vv` with `ww` in the reachable set of `www`
    @inbounds target[qqqinv] = ww
    @inbounds invptr[qqqinv] = ppinv

    # decrease the weighted degree of `w` by the weights of `v` 
    # and `vv`, and increase it by the weights of `ww` and `www`
    @inbounds degree[w] -= (weight[v] + weight[vv] - weight[ww] - weight[www])

    # decrease the weighted degree of `ww` by the weights of `v` 
    # and `vv`, and increase it by the weights of `w` and `www`
    @inbounds degree[ww] -= (weight[v] + weight[vv] - weight[w] - weight[www])

    # decrease the weighted degree of `www` by the weights of `v` 
    # and `vv`, and increase it by the weights of `w` and `ww`
    @inbounds degree[www] -= (weight[v] + weight[vv] - weight[w] - weight[ww])                
    return
end

function pr3_3_reach!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    p = pp = ppp = zero(E)
    w = ww = www = zero(V)
    
    p, w, pp, ww, ppp, www = pr3_reach!((p, w, pp, ww, ppp, www),
        stack5, target, begptr, endptr, invptr, v) do (p, w, pp, ww, ppp, www), pnxt, wnxt
        
        if iszero(w)
            p, w = pnxt, wnxt
        elseif iszero(ww)
            pp, ww = pnxt, wnxt
        elseif iszero(www)
            ppp, www = pnxt, wnxt
        end
        
        return (p, w, pp, ww, ppp, www)    
    end

    return (p, w, pp, ww, ppp, www)
end

function pr3_3_mark!(
        marker::AbstractVector{V},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::V,
        w::V,
        ww::V,
    ) where {V, E}
    # mark elements reachable by `w` with `tag`
    pr3_reach!(nothing, stack5,
        target, begptr, endptr, invptr, w) do _, _, x
        @inbounds marker[x] = tag
        return
    end

    # mark elements reachable by `ww` and `w` with `tag` + 1
    # mark elements reachable by `ww` and not `w` with `tag` + 2
    pr3_reach!(nothing, stack5,
        target, begptr, endptr, invptr, ww) do _, _, xx
        @inbounds xxtag = marker[xx]

        if xxtag == tag
            @inbounds marker[xx] = tag + one(V)
        else
            @inbounds marker[xx] = tag + two(V)
        end

        return
    end

    return
end

function pr3_buddy_reach!(
        marker::AbstractVector{V},
        stack5::AbstractVector{V},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        width::W,
        tag::V,
        v::V, 
        w::V,
        ww::V,
        www::V,
    ) where {V, E, W}
    tol = tolerance(W)

    # `vwgt` is the weight of `v`
    @inbounds vwgt = weight[v]

    # `wwgt` is the weight of `w`
    @inbounds wwgt = weight[w]

    # `wwwgt` is the weight of `ww`
    @inbounds wwwgt = weight[ww]

    # `wwwwgt` is the weight of `www`
    @inbounds wwwwgt = weight[www]

    # sort the weights of `w`, `ww`, and `www`
    minwgt, medwgt, maxwgt = sortthree(wwgt, wwwgt, wwwwgt)

    # `vv` is a buddy of `v`
    vv = zero(V)

    vv = pr3_reach!(vv, stack5,
        target, begptr, endptr, invptr, www) do vv, _, vnxt

        if iszero(vv) && v != vnxt
            @inbounds nxtdeg = degree[vnxt]

            if nxtdeg < width + tol
                @inbounds nxtnum = number[vnxt]

                if isthree(nxtnum)
                    @inbounds nxttag = marker[vnxt]

                    if nxttag == tag + one(V)
                        @inbounds nxtwgt = weight[vnxt]

                        if minwgt < min(vwgt, nxtwgt) + tol && medwgt < max(vwgt, nxtwgt) + tol
                            vv = vnxt
                        end
                    end
                end
            end
        end

        return vv
    end

    return vv
end

function pr3_cube_reach!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        v::V, 
        w::V,
        ww::V,
        www::V,
    ) where {V, E, W}
    tol = tolerance(W)

    # `x`, `xx` and `xxx` are elements reachable by `w`
    q, x, qq, xx, qqq, xxx = pr3_3_reach!(stack5,
        target, begptr, endptr, invptr, w)        

    # `y`, `yy` and `yyy` are elements reachable by `ww`
    r, y, rr, yy, rrr, yyy = pr3_3_reach!(stack5,
        target, begptr, endptr, invptr, ww)        

    # `z`, `zz`, and `zzz` are elements reachable by `www`
    s, z, ss, zz, sss, zzz = pr3_3_reach!(stack5,
        target, begptr, endptr, invptr, www)        

    # ensure that `v` = `xxx`
    (q, x), (qq, xx), (qqq, xxx) = sortthree((q, x), (qq, xx), (qqq, xxx)) do (_, x)
        return x == v
    end

    # ensure that `v` = `yyy`
    (r, y), (rr, yy), (rrr, yyy) = sortthree((r, y), (rr, yy), (rrr, yyy)) do (_, y)
        return y == v
    end

    # ensure that `v` = `zzz`
    (s, z), (ss, zz), (sss, zzz) = sortthree((s, z), (ss, zz), (sss, zzz)) do (_, z)
        return z == v
    end

    # copy `z` to avoid boxing
    zc = z

    # ensure that `z` = `xx`
    (q, x), (qq, xx) = sorttwo((q, x), (qq, xx)) do (_, x)
        x == zc
    end

    # copy `x` to avoid boxing
    xc = x

    # ensure that `x` = `yy`
    (r, y), (rr, yy) = sorttwo((r, y), (rr, yy)) do (_, y)
        y == xc
    end

    # `wwgt` is the weight of `w`
    @inbounds wwgt = weight[w]

    # `wwwgt` is the weight of `ww`
    @inbounds wwwgt = weight[ww]

    # `wwwwgt` is the weight of `www`
    @inbounds wwwwgt = weight[www]

    # `xwgt` is the weight of `x`
    @inbounds xwgt = weight[x]

    # `ywgt` is the weight of `y`
    @inbounds ywgt = weight[y]

    # `zwgt` is the weight of `x`
    @inbounds zwgt = weight[z]

    if x != yy || y != zz || z != xx || (
            (wwgt + tol <= xwgt || wwwgt + tol <= ywgt || wwwwgt + tol <= zwgt) &&
            (wwgt + tol <= zwgt || wwwgt + tol <= xwgt || wwwwgt + tol <= ywgt)
        )

        q = qq = r = rr = s = ss = zero(E)
        x = y = z = zero(V)
    end

    return (q, qq, r, rr, s, ss, x, y, z)
end

function pr3_cube!(
        marker::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack5::AbstractVector{V},
        index2::AbstractVector{V},
        index3::AbstractVector{V},
        number::AbstractVector{V},
        degree::AbstractVector{W},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        tag::V,
        hi2::V,
        hi3::V,
        w::V, ww::V, www::V,
        q::E, qq::E, r::E, rr::E,
        s::E, ss::E, x::V, y::V, z::V,
    ) where {W, V, E}
    #       x
    #    ╱     ╲
    # w           ww
    # │  ╲     ╱  │
    # │     v     │
    # z     │     y
    #    ╲  │  ╱
    #      www

    # mark elements reachable by `x` and not `y` with `tag` + 3
    # mark elements reachable by `y` and `x` with `tag` + 4
    # mark elements reachable by `y` and not `x` with `tag` + 5
    pr3_3_mark!(marker, stack5, target, begptr,
        endptr, invptr, tag + three(V), x, y)

    @inbounds xtag = marker[x]
    @inbounds ztag = marker[z]

    @inbounds xnum = number[x]
    @inbounds ynum = number[y]
    @inbounds znum = number[z]

    @inbounds xdeg = degree[x]
    @inbounds ydeg = degree[y]
    @inbounds zdeg = degree[z]

    @inbounds xwgt = weight[x]
    @inbounds ywgt = weight[y]
    @inbounds zwgt = weight[z]

    @inbounds wwgt = weight[w]
    @inbounds wwwgt = weight[ww]
    @inbounds wwwwgt = weight[www]

    @inbounds qinv = invptr[q]
    @inbounds qqinv = invptr[qq]
   
    if tag + three(V) <= ztag <= tag + four(V)  
        # w ─── x
        # │  ╱
        # z

        # remove `w` from the reachable sets of `x` and `z`
        pr3_reach_del!(source, target, endptr, invptr, qinv)
        pr3_reach_del!(source, target, endptr, invptr, qqinv)

        if isthree(xnum)
            # if `x` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, x)

            # add `x` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, x)
        elseif isfour(xnum)
            # if `x` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, x)
        end

        if isthree(znum)
            # if `z` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, z)

            # add `x` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, z)
        elseif isfour(znum)
            # if `z` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, z)
        end

        # decrement the degree of `x` and `z`
        xnum -= one(V)
        znum -= one(V)

        # decrease the weighted degree of `x` and `z` by the
        # weight of `w`
        xdeg -= wwgt
        zdeg -= wwgt
    else
        # w ─── x
        # │
        # z

        # replace `w` with `z` in the reachable set of `x`
        @inbounds target[qinv] = z; invptr[qinv] = qqinv

        # replace `w` with `x` in the reachable set of `z`
        @inbounds target[qqinv] = x; invptr[qqinv] = qinv

        # increase the weighted degree of of `x` by the weight
        # of `z` and decrease it by the weight of `w`
        xdeg -= (wwgt - zwgt)

        # increase the weighted degree of `z` by the weight
        # of `x` and decrease it by the weight of `w`
        zdeg -= (wwgt - xwgt)
    end

    @inbounds rinv = invptr[r]
    @inbounds rrinv = invptr[rr]
  
    if tag + four(V) <= xtag
        # ww ─── y
        #  │  ╱
        #  x

        # remove `ww` from the reachable sets of `y` and `x`
        pr3_reach_del!(source, target, endptr, invptr, rinv)
        pr3_reach_del!(source, target, endptr, invptr, rrinv)

        if isthree(ynum)
            # if `y` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, y)

            # add `y` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, y)
        elseif isfour(ynum)
            # if `y` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, y)
        end

        if isthree(xnum)
            # if `x` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, x)

            # add `x` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, x)
        elseif isfour(xnum)
            # if `x` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, x)
        end

        # decrement the degree of `y` and `x`
        ynum -= one(V)
        xnum -= one(V)

        # decrease the weighted degree of `y` and `x` by the
        # weight of `ww`
        ydeg -= wwwgt
        xdeg -= wwwgt
    else
        # ww ─── y
        #  │
        #  x

        # replace `ww` with `x` in the reachable set of `y`
        @inbounds target[rinv] = x; invptr[rinv] = rrinv

        # replace `ww` with `y` in the reachable set of `x`
        @inbounds target[rrinv] = y; invptr[rrinv] = rinv

        # increase the weighted degree of of `y` by the weight
        # of `x` and decrease it by the weight of `ww`
        ydeg -= (wwwgt - xwgt)

        # increase the weighted degree of `x` by the weight
        # of `y` and decrease it by the weight of `ww`
        xdeg -= (wwwgt - ywgt)
    end

    @inbounds sinv = invptr[s]
    @inbounds ssinv = invptr[ss]
   
    if tag + four(V) <= ztag
        # www ─── z
        #   │  ╱
        #   y

        # remove `www` from the reachable sets of `z` and `y`
        pr3_reach_del!(source, target, endptr, invptr, sinv)
        pr3_reach_del!(source, target, endptr, invptr, ssinv)

        if isthree(znum)
            # if `z` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, z)

            # add `x` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, z)
        elseif isfour(znum)
            # if `z` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, z)
        end

        if isthree(ynum)
            # if `y` has degree 3, remove it from the stack
            # of degree 3 elements
            hi3 = pr3_stack_del!(stack3, index3, hi3, y)

            # add `y` to the stack of degree 2 elements
            hi2 = pr3_stack_add!(stack2, index2, hi2, y)
        elseif isfour(ynum)
            # if `y` has degree 4, add it to the stack
            # of degree 3 elements
            hi3 = pr3_stack_add!(stack3, index3, hi3, y)
        end

        # decrement the degree of `z` and `y`
        znum -= one(V)
        ynum -= one(V)

        # decrease the weighted degree of `z` and `y` by the
        # weight of `www`
        zdeg -= wwwwgt
        ydeg -= wwwwgt
    else
        # www ─── z
        #   │
        #   y

        # replace `www` with `y` in the reachable set of `z`
        @inbounds target[sinv] = y; invptr[sinv] = ssinv

        # replace `www` with `z` in the reachable set of `x`
        @inbounds target[ssinv] = z; invptr[ssinv] = sinv

        # increase the weighted degree of of `z` by the weight
        # of `y` and decrease it by the weight of `www`
        zdeg -= (wwwwgt - ywgt)

        # increase the weighted degree of `y` by the weight
        # of `z` and decrease it by the weight of `www`
        ydeg -= (wwwwgt - zwgt)
    end

    @inbounds number[x] = xnum
    @inbounds number[y] = ynum
    @inbounds number[z] = znum

    @inbounds degree[x] = xdeg
    @inbounds degree[y] = ydeg
    @inbounds degree[z] = zdeg

    return hi2, hi3
end

function pr3_stack_del!(stack::AbstractVector{V}, index::AbstractVector{V}, hi::V, v::V) where {V}
    @inbounds i = index[v]
    
    if i < hi
        @inbounds x = stack[i] = stack[hi]; index[x] = i
    end

    hi -= one(V)
    return hi
end

function pr3_stack_add!(stack::AbstractVector{V}, index::AbstractVector{V}, hi::V, v::V) where {V}
    @inbounds hi = index[v] = pr3_stack_add!(stack, hi, v)
    return hi
end

function pr3_stack_add!(stack::AbstractVector{V}, hi::V, v::V) where {V}
    @inbounds hi += one(V); stack[hi] = v
    return hi
end

function pr3_stack_pop!(stack::AbstractVector{V}, hi::V) where {V}
    @inbounds v = stack[hi]; hi -= one(V)
    return hi, v
end

function pr3_reach_del!(
        source::AbstractVector{V},
        target::AbstractVector{V},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        p::E
    ) where {V, E}
    @inbounds pend = endptr[source[p]] -= one(E)

    if p < pend
        @inbounds x = target[p] = target[pend]

        if ispositive(x)
            @inbounds pinv = invptr[p] = invptr[pend]
            @inbounds invptr[pinv] = p
        end
    end

    return
end

function sorttwo(x, y)
    return sorttwo(identity, x, y)
end

function sorttwo(f::Function, x, y)
    if f(y) < f(x)
        x, y = y, x
    end

    return x, y
end

function sortthree(x, y, z)
    return sortthree(identity, x, y, z)
end

function sortthree(f::Function, x, y, z)
    x, y = sorttwo(f, x, y)
    y, z = sorttwo(f, y, z)
    x, y = sorttwo(f, x, y)
    return (x, y, z)
end

function pr3_reach!(
        combine::Function,
        result,
        vstack::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    # push `v` to the stack
    @inbounds num = one(V); vstack[num] = v

    @inbounds while ispositive(num)
        # `vv` is a supernode adjacent to `v`
        vv = vstack[num]; num -= one(V)

        # the arcs {`pp`, ..., `ppend` - 1} are incident
        # to `vv`
        pp = begptr[vv]; ppend = endptr[vv]

        while pp < ppend
            # `ww` is adjacent to `vv` and reachable by `v`
            ww = target[pp]

            # if `ww` is an element, fold it into `result`
            if ispositive(ww)
                result = combine(result, pp, ww)
                pp += one(E)

            # otherwise, `ww` is a supernode
            else
                # `ppnxt` is the largest possible value of `ppend`
                ppnxt = begptr[vv + one(V)]

                # the arcs {`qq`, ..., `qqend` - 1} are incident
                # to `ww`
                ww = -ww; qq = begptr[ww]; qqend = endptr[ww]

                # while there is space, move neighbors of `ww` 
                # into the neighborhood of `vv`
                while ppend < ppnxt && qq < qqend
                    qqend -= one(E)

                    # `xx` is adjacent to `ww` and reachable by `v`
                    xx = target[ppend] = target[qqend]

                    if ispositive(xx)
                        # `xx` is an element; `ppinv` is the arc
                        # (`xx`, `v`)
                        ppinv = invptr[ppend] = invptr[qqend]
                        invptr[ppinv] = ppend
                    end

                    ppend += one(E)
                end

                endptr[ww] = qqend

                # if `ww` has more than two neighbors, push
                # it to the stack
                if qq < qqend - one(E)
                    num += one(V); vstack[num] = ww
                    pp += one(E)

                # if `ww` has only one neighbor `xx`, replace
                # `ww` with `xx` in the neighborhood of `vv` 
                elseif qq == qqend - one(E)
                    xx = target[pp] = target[qq]

                    if ispositive(xx)
                        # `xx` is an element; `ppinv` is the arc
                        # (`xx`, `v`)
                        ppinv = invptr[pp] = invptr[qq]
                        invptr[ppinv] = pp
                    end

                # if `ww` has no neighbors, replace it with a vertex
                # `xx` in the neighborhood of `vv`
                else
                    ppend -= one(E)

                    if pp < ppend
                        # `xx` is adjacent to `vv` and reachable by `v`
                        xx = target[pp] = target[ppend]

                        if ispositive(xx)
                            # `xx` is an element; `ppinv` is the arc
                            # (`xx`, `v`)
                            ppinv = invptr[pp] = invptr[ppend]
                            invptr[ppinv] = pp
                        end
                    end
                end
            end
        end

        endptr[vv] = ppend
    end

    return result
end
