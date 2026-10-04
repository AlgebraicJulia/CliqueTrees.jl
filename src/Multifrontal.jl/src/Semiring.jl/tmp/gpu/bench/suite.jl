# The benchmark graphs by structure, so that every measurement covers every kind of graph the solver
# meets (the method's cost depends on the separators: grids and meshes are its best case, social and
# power-law graphs its worst). All are undirected with integer weights 1:100 (data/mtx; bench/export_mtx.jl,
# bench/export_more.jl and bench/export_real.jl write them).
#
#   include("bench/suite.jl"); suite(:quick)     # one or two graphs per class, laptop-sized
#   suite(:all)                                  # every graph up to ~80k vertices
#   suite(:large)                                # the big ones (HPG GPUs only)

const SUITE = [
    # class                       quick                               all
    ("3D grid",                   ["grid3d-25"],                      ["grid3d-25", "grid3d-30", "grid3d-40"]),
    ("2D grid",                   ["grid2d-150"],                     ["grid2d-150", "grid2d-180", "grid2d-250"]),
    ("2D mesh",                   ["delaunay_n14", "fe_sphere"],      ["fe_4elt2", "delaunay_n14", "fe_sphere", "delaunay_n15", "t60k", "delaunay_n16"]),
    ("3D / structural mesh",      ["bcsstk30"],                       ["bcsstk30", "fe_body", "wing", "fe_tooth"]),
    ("random geometric",          ["rgg_n_2_15_s0"],                  ["rgg_n_2_15_s0", "rgg_n_2_16_s0"]),
    ("road",                      String[],                           ["luxembourg_osm"]),
    ("power grid / circuit",      ["power", "memplus"],               ["power", "memplus"]),
    ("collaboration / social",    ["ca-GrQc", "ca-CondMat"],          ["ca-GrQc", "ca-HepTh", "PGPgiantcompo", "ca-AstroPh", "ca-CondMat", "email-Enron", "cond-mat-2005"]),
    ("internet / power law",      ["as-22july06"],                    ["as-22july06", "kron_g500-logn16"]),
]

const LARGE = ["luxembourg_osm", "grid2d-380", "grid2d-450", "grid3d-52", "grid3d-58"]

# the graphs of a selection, smallest first (by file size), and the class of each
function suite(which::Symbol = :quick)
    names = which === :quick ? reduce(vcat, (q for (_, q, _) in SUITE)) :
            which === :all ? reduce(vcat, (a for (_, _, a) in SUITE)) :
            which === :large ? LARGE : error("suite: :quick, :all or :large")
    dir = joinpath(@__DIR__, "..", "data", "mtx")
    names = filter(g -> isfile(joinpath(dir, g * ".mtx")), unique(names))
    return sort(names; by = g -> filesize(joinpath(dir, g * ".mtx")))
end

graph_class(g) = something(findfirst(c -> g in c[3] || g in c[2], SUITE) |> i -> isnothing(i) ? nothing : SUITE[i][1], "other")
