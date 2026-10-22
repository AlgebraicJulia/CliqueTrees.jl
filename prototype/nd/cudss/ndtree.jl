# cuDSS ND partition tree for any elimination order (NOTES.md, "partition tree").
# cuDSS (0.8) factorizes a user_perm ~10x slower unless CUDSS_DATA_USER_ND_PARTITION_TREE is also given: 2^L - 1
# block sizes (L = CUDSS_CONFIG_ND_NLEVELS, default 10) of a complete binary separator tree. Layout (checked on
# cuDSS's own perm + tree, laptop): array = level L-1 (2^(L-1) leaf domains) left to right, then each level up, root
# separator last; the permutation lists the blocks in that same order (NOT a postorder: a postordered perm with these
# sizes made cuDSS fail with an illegal address).
# nd_tree(ptr, adj, order; L): etree of the order, postorder it (same fill), then split recursively: a node owns a set
# of independent etree subtrees; its separator is the chain of single-child vertices at the top of the largest one
# (down to and including its first branching vertex), its children are the remaining independent subtrees, split into
# two size-balanced groups (never an empty separator with nonempty children: cuDSS's own trees never have one and a
# tree with such nodes failed in cuDSS on the laptop). Below level L-1 a node keeps everything.
# Returns (new elimination order: a reordering of independent subtrees, same fill; tree::Vector{Int32}).

# elimination tree in position space: par[k] = parent position of position k (0 = root)
function etree_pos(ptr, adj, order)
    n = length(ptr) - 1
    pos = Vector{Int}(undef, n); for (k, v) in enumerate(order); pos[v] = k; end
    par = zeros(Int, n); anc = zeros(Int, n)
    @inbounds for k in 1:n
        v = order[k]
        for q in ptr[v]:(ptr[v + 1] - 1)
            j = pos[adj[q]]
            j < k || continue
            while j != 0 && j != k          # walk up with path compression
                nx = anc[j]; anc[j] = k
                nx == 0 && (par[j] = k)
                j = nx
            end
        end
    end
    return par
end

function postorder_pos(par)
    n = length(par)
    head = zeros(Int, n); nxt = zeros(Int, n); roots = Int[]
    for k in n:-1:1            # children lists in increasing position order
        p = par[k]
        if p == 0; push!(roots, k); else; nxt[k] = head[p]; head[p] = k; end
    end
    reverse!(roots)
    post = Vector{Int}(undef, n); t = 0
    stack = Int[]; it = zeros(Int, n)
    for r in roots
        push!(stack, r); it[r] = head[r]
        while !isempty(stack)
            v = stack[end]; c = it[v]
            if c != 0
                it[v] = nxt[c]; push!(stack, c); it[c] = head[c]
            else
                pop!(stack); t += 1; post[t] = v
            end
        end
    end
    return post        # post[i] = old position of the i-th vertex in postorder
end

function nd_tree(ptr, adj, order; L::Int = 10, minsz::Int = 0)
    n = length(ptr) - 1
    par0 = etree_pos(ptr, adj, order)
    post = postorder_pos(par0)
    newpos = Vector{Int}(undef, n); for (i, k) in enumerate(post); newpos[k] = i; end
    po = [order[k] for k in post]                  # postordered elimination order (same fill)
    par = zeros(Int, n)
    for k in 1:n; par0[k] == 0 || (par[newpos[k]] = newpos[par0[k]]); end
    # children and subtree sizes in the postordered tree (subtree of k = positions k - sz[k] + 1 : k)
    nch = zeros(Int, n); sz = ones(Int, n)
    for k in 1:n; p = par[k]; p == 0 || (nch[p] += 1; sz[p] += sz[k]); end
    cptr = zeros(Int, n + 1); cptr[1] = 1
    for k in 1:n; cptr[k + 1] = cptr[k] + nch[k]; end
    fill_ = copy(cptr); ch = Vector{Int}(undef, n)
    for k in 1:n; p = par[k]; p == 0 || (ch[fill_[p]] = k; fill_[p] += 1); end
    roots = [k for k in 1:n if par[k] == 0]
    tree = zeros(Int32, 2^L - 1)
    blk = [Int[] for _ in 1:(2^L - 1)]           # positions of each heap node's block
    idx(h, lev) = (sum(2^l for l in (lev + 1):(L - 1); init = 0) + (h - 2^lev) + 1)
    # independent subtrees -> two groups, largest first into the lighter group
    function split2(K)
        A = Int[]; B = Int[]; sa = 0; sb = 0
        for k in sort(K; by = k -> -sz[k])
            if sa <= sb; push!(A, k); sa += sz[k]; else; push!(B, k); sb += sz[k]; end
        end
        return A, B
    end
    emit_sub(k, h) = append!(blk[h], (k - sz[k] + 1):k)
    # node h at level lev owns the independent subtrees R; children first, then its separator (postorder layout)
    function rec(R, h, lev)
        isempty(R) && return
        if lev == L - 1 || sum(sz[r] for r in R) <= minsz      # leaf, or kept whole (children empty)
            tree[idx(h, lev)] = sum(sz[r] for r in R)
            foreach(k -> emit_sub(k, h), sort(R)); return
        end
        r = R[argmax([sz[x] for x in R])]
        c = r; s = 1
        while nch[c] == 1
            c = ch[cptr[c]]; s += 1
        end
        # separator = chain c..r (positions, single-child links); the rest of R and c's children are independent
        K = vcat(ch[cptr[c]:(cptr[c + 1] - 1)], [x for x in R if x != r])
        tree[idx(h, lev)] = s
        A, B = split2(K)
        rec(A, 2h, lev + 1); rec(B, 2h + 1, lev + 1)
        append!(blk[h], c:r)
    end
    rec(roots, 1, 0)
    # cuDSS layout: the permutation is the blocks in tree-array order (leaf level first, root separator last)
    out = Int[]; sizehint!(out, n)
    for lev in (L - 1):-1:0, h in 2^lev:(2^(lev + 1) - 1)
        append!(out, blk[h])
    end
    @assert length(out) == n && sum(tree) == n
    return [po[k] for k in out], tree
end

# check: blocks of the permutation in tree-array order; every edge must join a node and one of its ancestors
function check_layout(ptr, adj, order, tree; L::Int = 10)
    n = length(ptr) - 1
    idx(h, lev) = (sum(2^l for l in (lev + 1):(L - 1); init = 0) + (h - 2^lev) + 1)
    off = cumsum(vcat(0, Int.(tree)))
    off[end] == n || return "sizes do not sum to n"
    node = zeros(Int, n)
    for lev in 0:(L - 1), h in 2^lev:(2^(lev + 1) - 1)
        a = idx(h, lev)
        for k in (off[a] + 1):off[a + 1]; node[order[k]] = h; end
    end
    isanc(a, b) = (while b > a; b >>= 1; end; a == b)
    bad = 0
    for u in 1:n, q in ptr[u]:(ptr[u + 1] - 1)
        a, b = node[u], node[adj[q]]
        (isanc(a, b) || isanc(b, a)) || (bad += 1)
    end
    return bad == 0 ? "ok" : "$bad edges between unrelated tree nodes"
end
