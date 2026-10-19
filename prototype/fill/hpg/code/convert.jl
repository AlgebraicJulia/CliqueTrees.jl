# usage: julia convert.jl in.mtx out.bin   (lattice names: julia convert.jl lat2_512 out.bin)
include(joinpath(@__DIR__, "fcommon.jl"))
function readmtx(path)
    b = read(path); L = length(b); p = 1
    nextline(p) = (while p <= L && b[p] != 0x0a; p += 1; end; p + 1)
    function nint(p)
        while b[p] < 0x30 || b[p] > 0x39; p += 1; end
        x = 0
        while p <= L && 0x30 <= b[p] <= 0x39; x = 10x + (b[p] - 0x30); p += 1; end
        return x, p
    end
    while b[p] == UInt8('%'); p = nextline(p); end
    r, p = nint(p); c, p = nint(p); nz, p = nint(p); p = nextline(p)
    @assert r == c
    I = Int32[]; J = Int32[]; sizehint!(I, nz); sizehint!(J, nz)
    for _ in 1:nz
        i, p = nint(p); j, p = nint(p); p = nextline(p)
        i != j && (push!(I, i); push!(J, j))
    end
    return r, I, J
end
function main(inp, outp)
    if isfile(inp)
        n, I, J = readmtx(inp)
    else
        g = gen_graph(inp); n = nv(g); I, J = edges_of(g)
    end
    serialize(outp, (n, I, J)); println("$inp -> $outp n=$n m=$(length(I))")
end
main(ARGS[1], ARGS[2])
