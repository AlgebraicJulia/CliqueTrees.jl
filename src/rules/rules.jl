function compressreduce(rules::Integer, weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W; verbose::Bool = false) where {W <: Number, V <: Integer}
    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    if rules == 1
        degree = FVector{W}(undef, n)
        number = FVector{V}(undef, n)
        status = FVector{UInt8}(undef, n)
        stack1 = FVector{V}(undef, n)
        stack4 = FVector{V}(undef, n)
        inject = FVector{V}(undef, n)
        ptr = FVector{E}(undef, nn)
        tgt = FVector{V}(undef, m)

        return compressreduce(weights, graph, width; verbose) do w, g, wd
            pr1_impl!(w, degree, number, status, stack1, stack4, inject, ptr, tgt, wd, g)
        end
    end

    if rules == 2
        degree = FVector{W}(undef, n)
        number = FVector{V}(undef, n)
        status = FVector{UInt8}(undef, n)
        target = FVector{V}(undef, m)
        begptr = FVector{E}(undef, nn)
        stack1 = FVector{V}(undef, n)
        stack2 = FVector{V}(undef, n)
        stack7 = FVector{V}(undef, n)
        stack4 = FVector{V}(undef, n)
        inject = FVector{V}(undef, n)
        ptr = FVector{E}(undef, nn)
        tgt = FVector{V}(undef, m)

        return compressreduce(weights, graph, width; verbose) do w, g, wd
            pr2_impl!(w, degree, number, status, target, begptr, stack1, stack2,
                stack7, stack4, inject, ptr, tgt, wd, g)
        end
    end

    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    status = FVector{UInt8}(undef, n)
    marker = FVector{V}(undef, n)
    source = FVector{V}(undef, m)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    invptr = FVector{E}(undef, m)
    stack1 = FVector{V}(undef, n)
    stack2 = FVector{V}(undef, n)
    stack3 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    stack5 = FVector{V}(undef, n)
    stack6 = FVector{V}(undef, n)
    stack7 = FVector{V}(undef, n)
    stack8 = FVector{V}(undef, n)
    stack0 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)

    if rules == 3
        return compressreduce(weights, graph, width; verbose) do w, g, wd
            totdeg = zero(W)

            @inbounds for v in vertices(g)
                totdeg += w[v]
            end

            pr3_impl!(
                w, degree, number, status, marker, source, target, begptr,
                endptr, invptr, stack1, stack2, stack3, stack4, stack5, stack6,
                stack7, stack8, stack0, tmpptr, totdeg, wd, g)
        end
    elseif rules == 4
        weight1 = FVector{W}(undef, n)
        sfillin = FVector{Int}(undef, n)
        ssource = FVector{V}(undef, m)
        starget = FVector{V}(undef, m)
        sbegptr = FVector{E}(undef, nn)
        stmpptr = FVector{E}(undef, nn)

        return compressreduce(weights, graph, width; verbose) do w, g, wd
            pr4_impl!(
                degree, number, status, marker, source, target, begptr, endptr, invptr,
                stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
                sfillin, ssource, starget, sbegptr, stmpptr, weight1, w, g, wd)
        end
    else
        weight1 = FVector{W}(undef, n)
        afillin = FVector{Int}(undef, n)
        amarker = FVector{Int}(undef, n)
        amarker2 = FVector{Int}(undef, n)
        asource = FVector{V}(undef, m)
        atarget = FVector{V}(undef, m)
        abegptr = FVector{E}(undef, nn)
        atmpptr = FVector{E}(undef, nn)
        atgt = FVector{V}(undef, m)

        return compressreduce(weights, graph, width; verbose) do w, g, wd
            pr5_impl!(
                degree, number, status, marker, source, target, begptr, endptr, invptr,
                stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
                afillin, amarker, amarker2, asource, atarget, abegptr, atmpptr, atgt,
                weight1, w, g, wd)
        end
    end
end

function compressreduce(reduce::F, weights::AbstractVector{W}, graph00::AbstractGraph{V}, width00::W; verbose::Bool = false) where {F <: Function, W <: Number, V <: Integer}
    E = etype(graph00); n00 = nv(graph00); m00 = de(graph00); nn00 = n00 + one(V)

    work00 = FScalar{V}(undef)
    work01 = FVector{V}(undef, nn00)
    work02 = FVector{V}(undef, n00)
    work03 = FVector{V}(undef, n00)
    work04 = FVector{V}(undef, n00)
    work05 = FVector{V}(undef, n00)
    work06 = FVector{V}(undef, n00)
    work07 = FVector{V}(undef, n00)
    work08 = FVector{E}(undef, nn00)
    work09 = FVector{V}(undef, m00)

    graph10 = simplegraph!(work08, work09, graph00)

    ptr11 = FVector{V}(undef, nn00)
    tgt11 = FVector{V}(undef, n00)
    ptr21 = FVector{V}(undef, nn00)
    tgt21 = FVector{V}(undef, n00)

    @inbounds for v00 in oneto(nn00)
        ptr11[v00] = v00
    end

    @inbounds for v00 in oneto(n00)
        tgt11[v00] = v00
    end

    project11 = BipartiteGraph{V, V}(n00, n00, n00, ptr11, tgt11)

    weights10 = FVector{W}(undef, n00)
    weights20 = FVector{W}(undef, n00)

    @inbounds for v00 in oneto(n00)
        weights10[v00] = weights[v00]
    end

    width10 = width00
    inject03 = FVector{V}(undef, n00); n03 = zero(V)

    if verbose
        println("| iter | \\|V\\| | \\|E\\| |")
        println("|------|------|------|")
    end

    lo = n00; hi = n00 + one(V); iter = 0

    @inbounds while lo < hi
        iter += 1
        hi = lo
        n10 = nv(graph10)

        if verbose
            println("| ", iter, " | ", n10, " | ", half(ne(graph10)), " |")
        end
        # V11
        #  ↓ inject11
        # V10
        #  ↑ inject12
        # V12
        #  ↓ project20
        # V20
        graph12, inject11, inject12, width20 = reduce(weights10, graph10, width10)
        graph20, project20 = compress_impl!(work00, work01, work02, work03, work04, work05, work06, work07, work08, work09, graph12, Val(true))
        n12 = nv(graph12); n20 = nv(graph20)

        for v11 in oneto(n10 - n12)
            v10 = inject11[v11]

            for v00 in neighbors(project11, v10)
                n03 += one(V); inject03[n03] = v00
            end
        end
        #            inject12
        #          V10  ←   V12
        # project11 ↑        ↓ project20
        #          V00  →   V20
        #           project21
        project21 = BipartiteGraph{V, V}(n00, n20, n00 - n03, ptr21, tgt21)
        pointers(project21)[begin] = p = one(V)

        for v20 in vertices(graph20)
            w20 = zero(W); vv20 = v20 + one(V)

            for v12 in neighbors(project20, v20)
                v10 = inject12[v12]
                w20 += weights10[v10]

                for v00 in neighbors(project11, v10)
                    targets(project21)[p] = v00; p += one(V)
                end
            end

            weights20[v20] = w20
            pointers(project21)[vv20] = p
        end

        lo = n20; graph10 = graph20; width10 = width20; project11 = project21
        weights10, weights20 = weights20, weights10
        ptr11, ptr21 = ptr21, ptr11
        tgt11, tgt21 = tgt21, tgt11
    end

    return weights10, graph10, view(inject03, oneto(n03)), project11, width10
end

include("pr1.jl")
include("pr2.jl")
include("pr3.jl")
include("pr4.jl")
include("pr5.jl")

