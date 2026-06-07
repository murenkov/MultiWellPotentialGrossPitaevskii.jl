"""
    _get_solver(backend::Backend) -> (alg, ensemble_alg)

Return an ODE solver algorithm and an ensemble algorithm for the given `backend`.

Extensions should specialize on their own `Backend` subtypes and return a tuple
`(alg, ensemble_alg)` compatible with `DiffEqBase.solve`. The default CPU
implementation returns `(Vern9(), EnsembleCPUArray())`. The GPU extension
MWPExtCUDA.jl specializes on `GPU`, selects CUDA device 0, and returns
`(GPUVern9(), EnsembleGPUKernel(CUDABackend()))`.
"""
function _get_solver(::CPU)
    return DE.Vern9(), DiffEqGPU.EnsembleCPUArray()
end

"""
    _initial_conditions(Cs, ps, tspan) -> (u0_vec, ps, tspan, s)

Build initial condition vectors from asymptotic amplitudes `Cs` for the
Gross–Pitaevskii ODE.

# Arguments
- `Cs`: vector of asymptotic amplitudes `C`
- `ps`: [`MultiWellParams`](@ref) containing potential parameters
- `tspan`: `(t₀, tₑ)` integration interval

# Returns
A tuple `(u0_vec, ps, tspan, s)` where `u0_vec` is a vector of
`SVector{2, T}` initial states `(u, u′)`, `ps`/`tspan` are potentially
modified for reverse-time integration, and `s` is the direction sign
(`+1` or `-1`).
"""
function _initial_conditions(
        Cs,
        ps::MultiWellParams{T, N},
        tspan::Tuple{T, T},
    ) where {T <: Real, N}
    (t₀, tₑ) = tspan
    ω = ps.ω
    s = sign(tₑ - t₀)
    u = Cs .* exp(s * √(-ω) * t₀)
    uₓ = s * √(-ω) .* u

    # Issue: https://github.com/SciML/DiffEqGPU.jl/issues/352
    if tₑ < t₀
        tspan = (-t₀, tₑ)
        (t₀, tₑ) = tspan
        ps = MultiWellParams(ps.ω, ps.as, -reverse(ps.ds))
        u = Cs .* exp(-s * √(-ω) * t₀)
        uₓ = -s * √(-ω) .* u
    end

    u0_vec = [SA.SVector{2, T}(x, y) for (x, y) in zip(u, uₓ)]
    return u0_vec, ps, tspan, s
end

"""
    _build_ensemble_problem(u0_vec, ps, tspan) -> EnsembleProblem

Wrap a [`MultiWellPotentialProblem`](@ref) in an `EnsembleProblem` for
batch solving across multiple initial conditions.

# Arguments
- `u0_vec`: vector of `SVector{2, T}` initial states
- `ps`: [`MultiWellParams`](@ref) containing potential parameters
- `tspan`: `(t₀, tₑ)` integration interval

# Returns
A `SciMLBase.EnsembleProblem` whose `prob_func` dispatches on
`sim_id` to select the initial condition from `u0_vec`.
"""
function _build_ensemble_problem(u0_vec, ps::MultiWellParams{T, N}, tspan) where {T <: Real, N}
    u0 = SA.@SVector T[0.0, 0.0]
    base_prob = MultiWellPotentialProblem(ps, u0, tspan)
    return SciMLBase.EnsembleProblem(
        base_prob;
        prob_func = (prob, ctx) -> DE.remake(prob, u0 = u0_vec[ctx.sim_id]),
        output_func = (sol, ctx) -> (sol.u[end], false),
        safetycopy = false,
    )
end

"""
    finish_points(Cs, ps, tspan; backend = CPU())

Integrate the Gross–Pitaevskii ODE from asymptotic initial conditions using
GPU-accelerated (CUDA) or CPU ensemble solving.

# Arguments
- `Cs`: vector of asymptotic amplitudes `C` for initial conditions
- `ps`: [`MultiWellParams`](@ref) containing potential parameters
- `tspan`: `(t₀, tₑ)` integration interval
- `backend`: solver backend — [`CPU`](@ref) (OrdinaryDiffEq, default) or [`GPU`](@ref) (CUDA via DiffEqGPU)

# Returns
A `DataFrame` with columns `C` (initial amplitude), `u` (final position),
and `ux` (final velocity).
"""
function finish_points(
        Cs,
        ps::MultiWellParams{T, N},
        tspan::Tuple{T, T};
        backend = CPU(),
    ) where {T <: Real, N}
    u0_vec, ps, tspan, s = _initial_conditions(Cs, ps, tspan)
    eproblem = _build_ensemble_problem(u0_vec, ps, tspan)

    if backend isa GPU && !applicable(_get_solver, backend)
        error("GPU backend requires CUDA.jl to be loaded. Add `using CUDA` to activate the GPU extension.")
    end
    alg, ensemble_alg = _get_solver(backend)
    solutions = DE.solve(
        eproblem, alg, ensemble_alg;
        dt = T(0.1),
        trajectories = length(u0_vec),
        adaptive = false,
        verbose = SciMLLogging.None(),
        save_everystep = false,
        save_on = false,
        save_start = true,
        save_end = true,
    )

    # Issue: https://github.com/SciML/DiffEqGPU.jl/issues/352
    return DataFrame(C = Cs, u = first.(solutions.u), ux = s * last.(solutions.u))
end

"""
    find_parametric_curves(Cs, ps; backend = CPU())

Compute the parametric curves `γ₋ = (u₋, u₋′)` and `γ₊ = (u₊, u₊′)` for
a set of asymptotic amplitudes.

# Arguments
- `Cs`: vector of asymptotic amplitudes
- `ps`: [`MultiWellParams`](@ref) containing potential parameters
- `backend`: solver backend — [`CPU`](@ref) or [`GPU`](@ref), passed to [`finish_points`](@ref)

# Returns
A `DataFrame` with columns `C`, `um`, `uxm`, `up`, `uxp` — the matched
`(u, u′)` pairs for negative (`γ₋`) and positive (`γ₊`) branches.
"""
function find_parametric_curves(Cs, ps::MultiWellParams{T, N}; backend = CPU()) where {T <: Real, N}
    pairs₋ = finish_points(Cs, ps, (T(-10.0), T(0.0)); backend = backend)
    filter!(row -> regular([row.u, row.ux]), pairs₋)

    pairs₊ = copy(pairs₋)
    pairs₊.ux = -pairs₊.ux
    filter!(row -> regular([row.u, row.ux]), pairs₊)

    rename!(pairs₋, :u => :um, :ux => :uxm)
    rename!(pairs₊, :u => :up, :ux => :uxp)
    return innerjoin(pairs₋, pairs₊; on = :C)
end
