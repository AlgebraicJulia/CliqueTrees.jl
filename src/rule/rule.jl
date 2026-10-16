mutable struct RuleSolver{W <: Number, V <: Integer, E <: Integer}
    work00    ::FScalar{V}
    work01    ::FVector{V}
    work02    ::FVector{V}
    work03    ::FVector{V}
    work04    ::FVector{V}
    work05    ::FVector{E}
    work06    ::FVector{V}
    graph10   ::FBipartiteGraph{V, E}
    weights10 ::FVector{W}
    weights20 ::FVector{W}
    width10   ::W
    ptr11     ::FVector{V}
    tgt11     ::FVector{V}
    ptr21     ::FVector{V}
    tgt21     ::FVector{V}
    project11 ::FBipartiteGraph{V, V}
    inject03  ::FVector{V}
    n03       ::V
end

struct RuleResult{W <: Number, V <: Integer}
    inject  ::FVector{V}
    ninject ::V
    project ::FBipartiteGraph{V, V}
    width   ::W
end

function RuleSolver(weights::AbstractVector{W}, graph00::AbstractGraph{V}, width00::W) where {W <: Number, V <: Integer}
    E = etype(graph00); n00 = nv(graph00); m00 = de(graph00); nn00 = n00 + one(V)

    work00 = FScalar{V}(undef)
    work01 = FVector{V}(undef, nn00)
    work02 = FVector{V}(undef, n00)
    work03 = FVector{V}(undef, n00)
    work04 = FVector{V}(undef, n00)
    work05 = FVector{E}(undef, nn00)
    work06 = FVector{V}(undef, m00)

    graph10 = simplegraph!(work05, work06, graph00)

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

    inject03 = FVector{V}(undef, n00)

    return RuleSolver{W, V, E}(
        work00, work01, work02, work03, work04, work05, work06,
        graph10, weights10, weights20, width00,
        ptr11, tgt11, ptr21, tgt21, project11,
        inject03, zero(V))
end

function RuleResult(s::RuleSolver{W, V}) where {W, V}
    return RuleResult{W, V}(s.inject03, s.n03, s.project11, s.width10)
end

# Reduce a graph with the rules of level `rules`, compressing true twins
# between rounds, until a round removes no vertex. At level 5, `ext` selects
# the star-forest and twin rules of `asr` (`ext = 0x00`: neither). If `minfrac`
# is positive, stop as soon as a round removes fewer than `minfrac` times the
# vertices it started with: the last round, which finds little or nothing,
# costs as much as a full pass over the graph.
function compressreduce(rules::Integer, weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W; verbose::Bool = false, ext::UInt8 = ASR_EXT_DEFAULT, minfrac::Real = 0.0) where {W <: Number, V <: Integer}
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

        return compressreduce(weights, graph, width; verbose, minfrac) do w, g, wd
            pr1_impl!(w, degree, number, status, stack1, stack4, inject, ptr, tgt, wd, g)
        end
    elseif rules == 2
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

        return compressreduce(weights, graph, width; verbose, minfrac) do w, g, wd
            pr2_impl!(w, degree, number, status, target, begptr, stack1, stack2,
                stack7, stack4, inject, ptr, tgt, wd, g)
        end
    else
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
            return compressreduce(weights, graph, width; verbose, minfrac) do w, g, wd
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

            return compressreduce(weights, graph, width; verbose, minfrac) do w, g, wd
                pr4_impl!(
                    degree, number, status, marker, source, target, begptr, endptr, invptr,
                    stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
                    sfillin, ssource, starget, sbegptr, stmpptr, weight1, w, g, wd)
            end
        else
            # level 5: the rules of `asr`, with the star-forest and twin rules
            # selected by `ext` (see pr5.jl)
            ws = ASRWork{V, E}(n)
            weight1 = FVector{W}(undef, n)
            afillin = FVector{Int}(undef, n)
            amarker = FVector{Int}(undef, n)
            amarker2 = FVector{Int}(undef, n)
            asource = FVector{V}(undef, m)
            atarget = FVector{V}(undef, m)
            abegptr = FVector{E}(undef, nn)
            atmpptr = FVector{E}(undef, nn)
            atgt = FVector{V}(undef, m)

            return compressreduce(weights, graph, width; verbose, minfrac) do w, g, wd
                pr5_impl!(
                    degree, number, status, marker, source, target, begptr, endptr, invptr,
                    stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
                    afillin, amarker, amarker2, asource, atarget, abegptr, atmpptr, atgt,
                    weight1, w, g, wd, ext, ws)
            end
        end
    end
end

function compressreduce(reduce::F, weights::AbstractVector{W}, graph00::AbstractGraph{V}, width00::W; verbose::Bool = false, minfrac::Real = 0.0) where {F <: Function, W <: Number, V <: Integer}
    return rule_solve!(reduce, RuleSolver(weights, graph00, width00); verbose, minfrac)
end

function rule_solve!(reduce::F, s::RuleSolver{W, V}; verbose::Bool = false, minfrac::Real = 0.0) where {F <: Function, W, V}
    if verbose
        println()
        println("| iter |       n |       m |")
        println("|------|---------|---------|")
    end

    lo = nv(s.graph10); hi = lo + one(V); iter = 0

    while lo < hi && (iszero(iter) || hi - lo >= minfrac * hi)
        iter += 1
        hi = lo

        if verbose
            println("| ", lpad(iter, 4), " | ", lpad(nv(s.graph10), 7), " | ", lpad(half(ne(s.graph10)), 7), " |")
        end

        lo, _ = rule_step!(reduce, s)
    end

    return s.weights10, s.graph10, RuleResult(s)
end

function rule_step!(reduce::F, s::RuleSolver{W, V, E}) where {F <: Function, W, V, E}
    s.graph10, s.width10, s.project11, s.n03, n, m = rule_step_impl!(
        reduce, s.work00, s.work01, s.work02, s.work03, s.work04, s.work05, s.work06,
        s.graph10, s.weights10, s.weights20, s.width10,
        s.ptr21, s.tgt21, s.project11, s.inject03, s.n03)

    s.weights10, s.weights20 = s.weights20, s.weights10
    s.ptr11, s.ptr21 = s.ptr21, s.ptr11
    s.tgt11, s.tgt21 = s.tgt21, s.tgt11

    return n, m
end

function rule_step_impl!(
        reduce::F,
        work00::FScalar{V},
        work01::FVector{V},
        work02::FVector{V},
        work03::FVector{V},
        work04::FVector{V},
        work05::FVector{E},
        work06::FVector{V},
        graph10::FBipartiteGraph{V, E},
        weights10::FVector{W},
        weights20::FVector{W},
        width10::W,
        ptr21::FVector{V},
        tgt21::FVector{V},
        project11::FBipartiteGraph{V, V},
        inject03::FVector{V},
        n03::V,
    ) where {F <: Function, W, V, E}
    n00 = nov(project11); n10 = nv(graph10)
    # V11
    #  ↓ inject11
    # V10
    #  ↑ inject12
    # V12
    #  ↓ project20
    # V20
    graph12, inject11, inject12, width20 = reduce(weights10, graph10, width10)
    graph20, project20 = compress_impl!(work00, work01, work02, work03, work04, work05, work06, graph12, Val(:T))
    n12 = nv(graph12); n20 = nv(graph20)

    @inbounds for v11 in oneto(n10 - n12)
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

    @inbounds for v20 in vertices(graph20)
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

    return graph20, width20, project21, n03, n20, half(ne(graph20))
end

function rule_pull(result::RuleResult{W, V}, order1::AbstractVector) where {W, V}
    return rule_pull_impl(result.inject, result.ninject, result.project, order1)
end

function rule_pull_impl(
        inject::FVector{V},
        ninject::V,
        project::FBipartiteGraph{V, V},
        order1::AbstractVector,
    ) where {V}
    order = Vector{V}(undef, nov(project)); i = zero(V)

    @inbounds for k in oneto(ninject)
        i += one(V); order[i] = inject[k]
    end

    @inbounds for j in order1
        for v in neighbors(project, j)
            i += one(V); order[i] = v
        end
    end

    return order
end

include("pr1.jl")
include("pr2.jl")
include("pr3.jl")
include("pr4.jl")
include("pr5.jl")

