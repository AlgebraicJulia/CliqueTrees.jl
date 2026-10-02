# ===== device profile =====
#
# Everything the kernels and the schedule need to know about the GPU, read from the driver once per
# device instead of hard-coded for the GPUs we happen to test on. `measure!` adds three numbers that
# the attributes do not give reliably: sustained DRAM bandwidth, sustained min-plus throughput, and
# the cost of one kernel launch. They are used to report efficiency as a fraction of what this GPU
# can do (bench/portable.jl), never to change results.

mutable struct DeviceProfile
    name::String
    capability::VersionNumber
    nsm::Int                    # streaming multiprocessors
    threads_sm::Int             # resident threads per SM
    warps_sm::Int
    blocks_sm::Int              # resident blocks per SM
    regs_sm::Int                # 32-bit registers per SM
    shmem_sm::Int               # shared memory per SM (bytes)
    shmem_block::Int            # shared memory per block with opt-in (bytes)
    l2::Int                     # L2 cache (bytes)
    clock::Float64              # SM clock (Hz) as reported; boost clocks may be higher
    bandwidth_spec::Float64     # DRAM bandwidth from memory clock and bus width (bytes/s); 0 if unknown
    # measured (0 until measure!)
    bandwidth::Float64          # sustained device-to-device copy (bytes read + written per second)
    minplus::Float64            # sustained Float32 min-plus multiply-adds per second
    launch::Float64             # seconds per empty kernel launch, back to back
end

const PROFILES = Dict{Int, DeviceProfile}()
const PROFILES_LOCK = ReentrantLock()

attr(d, a, default = 0) = try CUDA.attribute(d, a) catch; default end

function DeviceProfile(d::CuDevice = CUDA.device())
    nsm = attr(d, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    thr = attr(d, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    memclk = attr(d, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)          # kHz (deprecated in newer CUDA; may be 0)
    bus = attr(d, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    return DeviceProfile(CUDA.name(d), CUDA.capability(d), nsm, thr, thr ÷ 32,
        attr(d, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, 16),
        attr(d, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, 65536),
        attr(d, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 49152),
        attr(d, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 49152),
        attr(d, CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
        1e3 * attr(d, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE),
        2e3 * memclk * bus / 8,
        0.0, 0.0, 0.0)
end

# the profile of the current device (attributes only; call measure! for the measured numbers)
function device_profile(d::CuDevice = CUDA.device())
    lock(PROFILES_LOCK) do
        get!(() -> DeviceProfile(d), PROFILES, CUDA.deviceid(d))
    end
end

# resident threads that fill the GPU once
fill_threads(p::DeviceProfile = device_profile()) = p.nsm * p.threads_sm

# resident warps per SM for a kernel with `regs` registers per thread and blocks of `block` threads
function resident_warps(p::DeviceProfile, regs::Integer, block::Integer)
    wpb = cld(block, 32)
    perwarp = cld(max(regs, 1) * 32, 256) * 256                        # register allocation granularity
    byregs = (p.regs_sm ÷ perwarp) ÷ wpb
    return min(p.warps_sm ÷ wpb, byregs, p.blocks_sm) * wpb
end

# ----- microbenchmarks -----

@inline fmin(x::Float32, y::Float32) = ccall("llvm.minnum.f32", llvmcall, Float32, (Float32, Float32), x, y)

function minplus_bench_kernel!(out, x, ::Val{R}) where {R}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    a = x[1] + Float32(i & 7); b = x[2]
    c1 = 1f0; c2 = 2f0; c3 = 3f0; c4 = 4f0; c5 = 5f0; c6 = 6f0; c7 = 7f0; c8 = 8f0     # 8 independent chains

    for _ in 1:R
        c1 = fmin(c1, a + b); c2 = fmin(c2, a + 2f0 * b); c3 = fmin(c3, a + 3f0 * b); c4 = fmin(c4, a + 4f0 * b)
        c5 = fmin(c5, a + 5f0 * b); c6 = fmin(c6, a + 6f0 * b); c7 = fmin(c7, a + 7f0 * b); c8 = fmin(c8, a + 8f0 * b)
        a = fmin(a + 1f0, c1)
    end

    s = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    s == -1f0 && (out[1] = s)                                          # never true; keeps the work
    return
end

empty_kernel!() = return

# fills in bandwidth, minplus and launch (about one second); idempotent unless force = true
function measure!(p::DeviceProfile = device_profile(); force::Bool = false)
    (p.bandwidth > 0 && !force) && return p
    # DRAM: a copy much larger than L2
    n = clamp(8 * p.l2, 256 << 20, CUDA.free_memory() ÷ 4) ÷ 8
    x = CUDA.zeros(Float32, n); y = similar(x)
    copyto!(y, x); CUDA.synchronize()
    t = minimum(CUDA.@elapsed(copyto!(y, x)) for _ in 1:5)
    p.bandwidth = 2 * sizeof(x) / t
    CUDA.unsafe_free!(x); CUDA.unsafe_free!(y)
    # min-plus: 8 independent chains per thread, 4 waves of the whole GPU
    R = 2048
    out = CUDA.zeros(Float32, 1); v = CuVector{Float32}([1f0, 2f0])
    k = @cuda launch = false minplus_bench_kernel!(out, v, Val(R))
    tb = 256; nb = 4 * p.nsm * (p.threads_sm ÷ tb)
    k(out, v, Val(R); threads = tb, blocks = nb); CUDA.synchronize()
    t = minimum(CUDA.@elapsed(k(out, v, Val(R); threads = tb, blocks = nb)) for _ in 1:5)
    p.minplus = (8 * R) * tb * nb / t                                  # one min-plus multiply-add = one minnum after the add
    # launch overhead
    e = @cuda launch = false empty_kernel!()
    e(); CUDA.synchronize()
    t = @elapsed (for _ in 1:1000; e(); end; CUDA.synchronize())
    p.launch = t / 1000
    return p
end

function Base.show(io::IO, p::DeviceProfile)
    print(io, p.name, " (sm_", p.capability.major, p.capability.minor, "): ", p.nsm, " SMs × ", p.threads_sm, " threads, ",
        p.regs_sm ÷ 1024, "K regs, ", p.shmem_sm ÷ 1024, " KB shared/SM, L2 ", p.l2 >> 20, " MB")
    p.bandwidth_spec > 0 && print(io, ", DRAM ", round(Int, p.bandwidth_spec / 1e9), " GB/s spec")
    p.bandwidth > 0 && print(io, " | measured: copy ", round(Int, p.bandwidth / 1e9), " GB/s, min-plus ",
        round(p.minplus / 1e12; digits = 1), " T/s, launch ", round(p.launch * 1e6; digits = 1), " µs")
end
