using CliqueTrees
using Documenter

makedocs(;
    modules=[CliqueTrees],
    format=Documenter.HTML(),
    sitename="CliqueTrees.jl",
    doctest=false,
    checkdocs=:none,
    pages=["CliqueTrees.jl" => "index.md", "Library Reference" => "api.md"],
)

deploydocs(;
    target="build", repo="github.com/AlgebraicJulia/CliqueTrees.jl.git", branch="gh-pages"
)
