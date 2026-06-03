module MWPExtCUDA

using MultiWellPotentialGrossPitaevskii
using CUDA
import DiffEqGPU

_get_cuda_backend() = CUDA.CUDABackend()

function MultiWellPotentialGrossPitaevskii._get_solver(::MultiWellPotentialGrossPitaevskii.GPU)
    return DiffEqGPU.GPUVern9(), DiffEqGPU.EnsembleGPUKernel(_get_cuda_backend())
end

end
