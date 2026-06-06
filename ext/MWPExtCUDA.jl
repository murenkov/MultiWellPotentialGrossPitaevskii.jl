module MWPExtCUDA

using MultiWellPotentialGrossPitaevskii
using CUDA
import DiffEqGPU

function _initialise_cuda()
    if CUDA.has_cuda()
        CUDA.device!(0)
        @info "MWPExtCUDA: using CUDA device 0 — $(CUDA.name(CUDA.device()))"
    else
        @warn "MWPExtCUDA: no CUDA-capable GPU found"
    end
    return nothing
end

_get_cuda_backend() = (CUDA.device!(0); CUDA.CUDABackend())

function MultiWellPotentialGrossPitaevskii._get_solver(::MultiWellPotentialGrossPitaevskii.GPU)
    return DiffEqGPU.GPUVern9(), DiffEqGPU.EnsembleGPUKernel(_get_cuda_backend())
end

function __init__()
    return _initialise_cuda()
end

end
