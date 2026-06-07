# Tolerances for intersection finding
const COINCIDENT_ATOL = 1.0e-10
const CROSSING_ATOL = 1.0e-12
const DENOM_EPS = 1.0e-15

"""
    MultiWellParams{T, N}

Parameters for the multi-well potential Gross–Pitaevskii equation.

# Fields
- `ω`: chemical potential (energy offset)
- `as`: `SVector{N,T}` of well amplitudes
- `ds`: `SVector{N,T}` of well center positions
"""
struct MultiWellParams{T, N}
    ω::T
    as::SA.SVector{N, T}
    ds::SA.SVector{N, T}
end

"""
    MultiWellPotentialProblem(ps::MultiWellParams, u0, tspan; kwargs...)
    MultiWellPotentialProblem(ω, as, ds, u0, tspan; kwargs...)

Construct an [`ODEProblem`](@extref SciMLBase.ODEProblem) for the multi-well
potential Gross–Pitaevskii equation.

Accepts either a pre-built [`MultiWellParams`](@ref) or raw parameters
`ω`, `as`, `ds`. Returning a plain `ODEProblem` ensures full compatibility
with OrdinaryDiffEq integrators, `remake`, and ensemble solvers.

# Examples
```julia
prob = MultiWellPotentialProblem(-1.0, [1.0, 0.5], [-2.0, 2.0], [0.0, 0.0], (-10.0, 0.0))
sol = solve(prob, Vern9())
```
"""
function MultiWellPotentialProblem(ps::MultiWellParams{T, N}, u0, tspan; kwargs...) where {T, N}
    return ODEProblem(multiwell_potential_equation, u0, tspan, ps; kwargs...)
end

function MultiWellPotentialProblem(ω, as, ds, u0, tspan; kwargs...)
    return MultiWellPotentialProblem(
        MultiWellParams(ω, SA.SVector{length(as)}(as), SA.SVector{length(ds)}(ds)),
        u0, tspan; kwargs...
    )
end

"""
    ParametricCurve{T}

A parametric curve with coordinates `t`, `x(t)`, and `y(t)`.

# Fields
- `t`: parameter vector
- `x`: x-coordinate vector
- `y`: y-coordinate vector
"""
struct ParametricCurve{T}
    t::Vector{T}
    x::Vector{T}
    y::Vector{T}
end


abstract type Backend end
struct CPU <: Backend end
struct GPU <: Backend end
