function writegr(io::IO, graph::AbstractGraph)
    m = 0

    for v in vertices(graph), w in neighbors(graph, v)
        if v != w && (!is_directed(graph) || v < w)
            m += 1
        end
    end

    n = nv(graph)
    println(io, "p tw $n $m")

    for v in vertices(graph), w in neighbors(graph, v)
        if v != w && (!is_directed(graph) || v < w)
            println(io, "$v $w")
        end
    end

    return
end

function readgr(io::IO)
    lines = readlines(io)

    # remove comments
    filter!(lines) do line
        !startswith(line, "c")
    end

    # parse statistics
    words = split(popfirst!(lines))
    nv = parse(Int, words[3])
    ne = parse(Int, words[4])

    # parse graph
    p = 0
    I = Vector{Int}(undef, 2ne)
    J = Vector{Int}(undef, 2ne)
    
    while !isempty(lines)
        words = split(popfirst!(lines))
        
        i = parse(Int, words[1])
        j = parse(Int, words[2])
        
        p += 1; I[p] = i; J[p] = j
        p += 1; I[p] = j; J[p] = i
    end
    
    V = ones(Int, 2ne)
    return BipartiteGraph(sparse(I, J, V))
end


function readtd(io::IO)
    lines = readlines(io)

    # remove comments
    filter!(lines) do line
        !startswith(line, "c")
    end

    isempty(lines) && error("Empty File")

    # parse statistics
    words = split(lines[1])
    nb = parse(Int, words[3])
    tw = parse(Int, words[4])
    nv = parse(Int, words[5])

    # parse hypergraph
    index = Vector{Int}(undef, nb)
    pointer = Vector{Int}(undef, nb + 1)
    target = Vector{Int}(undef, 0)
    pointer[begin] = p = 1

    for i in oneto(nb)
        words = split(lines[i + 1])

        j = parse(Int, words[2])
        index[j] = i

        for word in words[3:end]
            p += 1
            push!(target, parse(Int, word))
        end

        pointer[i + 1] = p
    end

    hypergraph = BipartiteGraph(nv, nb, p - 1, pointer, target)

    # parse tree
    I = sizehint!(Int[], nb)
    J = sizehint!(Int[], nb)

    for line in lines[(nb + 2):end]
        words = split(line)
        i = parse(Int, words[1])
        j = parse(Int, words[2])
        push!(I, index[i], index[j])
        push!(J, index[j], index[i])
    end

    root = 1
    parent = dfs_parents(BipartiteGraph(sparse(I, J, J, nb, nb)), root)
    parent[root] = 0

    tree = Parent(nb, parent)
    return hypergraph, tree
end

function readeo(io::IO, ::Type{V}) where {V}
    hypergraph, tree = readtd(io)

    treeindex = postorder(tree)
    treeorder = invperm(treeindex)

    n = nov(hypergraph)
    index = zeros(V, n)
    i = n + 1

    for b in Iterators.reverse(treeorder), v in neighbors(hypergraph, b)
        if iszero(index[v])
            index[v] = i -= 1
        end
    end

    return index
end
