using SuiteSparseMatrixCollection
db = ssmc_db()
names = ["GHS_psdef/jnlbrng1","Schmid/thermal1","UTEP/Dubcova3","AMD/G2_circuit","Wissgott/parabolic_fem","McRae/ecology2","HB/bcsstk17","TKK/cbuckle","Simon/olafu","Boeing/ct20stif","Nasa/nasasrb","GHS_psdef/s3dkt3m2","Boeing/crystm03","Botonakis/thermomech_TC","Botonakis/thermomech_dM","GHS_psdef/apache1"]
for nm in names
    g, n = split(nm, "/")
    r = db[(db.group .== g) .& (db.name .== n), :]
    p = fetch_ssmc(g, n, format = "MM"); println(p); for d in (p isa AbstractString ? [p] : p); for f in readdir(d, join=true); endswith(f, ".mtx") && cp(f, joinpath(@__DIR__, "data", basename(f)), force=true); end; end
    println(nm)
end
