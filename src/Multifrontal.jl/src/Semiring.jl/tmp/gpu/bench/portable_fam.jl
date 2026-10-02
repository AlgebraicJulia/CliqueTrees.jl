# kernel families of a CUPTI trace (shared by bench/portable.jl and experiments)
const FAMILY = [
    r"^sgemx_kernel" => "gemm", r"^layered_down" => "L_layered", r"^rowmajor_down" => "L_layered",
    r"^downward_kernel" => "L_top", r"^upward_kernel" => "U_top", r"^upward_path" => "U_path",
    r"^strsx_" => "trsm_diag", r"^sgetrf_diag" => "lu_diag", r"^amalgamate" => "merge",
    r"^gpu_fill|^\[set" => "fill", r"^\[copy" => "copy", r"." => "other"]
family(name) = (k = match(r"^([A-Za-z_0-9!#\[\] ]+)", name); s = isnothing(k) ? name : k[1]; first(f for (r, f) in FAMILY if occursin(r, s)))

function families(res)
    d = res.device; out = Dict{String, Tuple{Float64, Int}}()
    for i in eachindex(d.id)
        f = family(d.name[i]); t, c = get(out, f, (0.0, 0))
        out[f] = (t + d.stop[i] - d.start[i], c + 1)
    end
    return out
end

