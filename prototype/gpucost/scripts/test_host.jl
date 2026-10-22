# CPU reference check: ordering -> bucket tree -> exact cost; writes .tree files for verify.py.
#   julia --project=env scripts/test_host.jl <outdir> <ord files...>
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
out = ARGS[1]; mkpath(out)
nets = Dict{String, Any}()
for f in ARGS[2:end]
    name = split(basename(f), "__")[1]
    net = get!(() -> readnet(name), nets, name)
    o = open_last(net, parse.(Int, split(readlines(f)[2])))
    tr = bucket_tree(net, o); cost!(tr, net)
    tag = replace(basename(f), ".ord" => "")
    Host.write_tree(joinpath(out, tag * ".tree"), tr; meta = Dict("log2C" => tr.log2C, "W" => tr.W))
    println(tag, "  log10C=", round(tr.log2C * log10(2); digits = 3), " W=", tr.W)
end
