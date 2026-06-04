module MultiWellPotentialGrossPitaevskii

import StaticArrays as SA
import OrdinaryDiffEq as DE
import DiffEqGPU
import SciMLBase
import SciMLLogging
import DataFrames: DataFrame, rename!, innerjoin

import Polynomials
import Interpolations
import Roots

export MultiWellParams, MultiWellPotentialProblem, multiwell_potential_equation
export singular, regular
export every_nth, monotonicity_intervals
export find_interpolations_intersections, find_polynomials_intersections
export find_intersections, nonlinear_range, fmt
export finish_points, find_parametric_curves
export ParametricCurve
export V₁, V, plot_u_ux_diagram
export Backend, CPU, GPU

"""
    V₁(t, A)

Single-well potential function.

# Arguments
- `t`: position coordinate
- `A`: well amplitude

# Returns
The potential value `-A * sech(t)^2`.
"""
@inline function V₁(t, A)
    return -A * sech(t)^2
end

"""
    V(t, as, ds)

Multi-well potential function composed of a sum of inverted sech² wells.

# Arguments
- `t`: position coordinate
- `as`: amplitudes of each well
- `ds`: displacements (center positions) of each well

# Returns
The potential value `-sum(as[i] * sech(t - ds[i])^2)`.
"""
@inline function V(t, as, ds)
    return -sum(as .* sech.(t .- ds) .^ 2)
end

"""
    singular(x)

Check if a value or array is "singular" (non-finite or diverging).

# Arguments
- `x`: value or array to check

# Returns
`true` if any element is `NaN` or has absolute value exceeding 10.0.
"""
function singular(x)
    for val in x
        if isnan(val) || abs(val) > 10.0
            return true
        end
    end
    return false
end

"""
    regular(x)

Inverse of [`singular`](@ref).

# Arguments
- `x`: value or array to check

# Returns
`true` if the value is finite and bounded (absolute value ≤ 10.0).
"""
regular(x) = !singular(x)

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
    return DE.ODEProblem(multiwell_potential_equation, u0, tspan, ps; kwargs...)
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

"""
    multiwell_potential_equation(u, p, t)

Right-hand side of the Gross–Pitaevskii ODE `u'' + (ω - V(t)) u - u³ = 0`
written as a first-order system `du/dt = [u₂, -(ω - V) u₁ + u₁³]`.

# Arguments
- `u`: state vector `[u, u′]`
- `p`: [`MultiWellParams`](@ref) containing `ω`, well amplitudes `as`, and displacements `ds`
- `t`: time (position) coordinate

# Returns
Time derivative `[u′, u″]` as an `SVector{2, T}`.
"""
@inline function multiwell_potential_equation(u::AbstractVector{T}, p::MultiWellParams{T, N}, t::T)::SA.SVector{2, T} where {T <: Real, N}
    ω = p.ω
    as = p.as
    ds = p.ds

    V = zero(T)
    @inbounds for k in 1:N
        diff = t - ds[k]
        sech_val = one(T) / cosh(diff)
        V -= as[k] * sech_val * sech_val
    end

    du₁ = u[2]
    du₂ = -(ω - V) * u[1] + u[1]^3
    return SA.SVector{2, T}(du₁, du₂)
end

function _get_solver(::CPU)
    return DE.Vern9(), DiffEqGPU.EnsembleCPUArray()
end

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

function _build_ensemble_problem(u0_vec, ps::MultiWellParams{T, N}, tspan) where {T <: Real, N}
    u0 = SA.@SVector T[0.0, 0.0]
    base_prob = MultiWellPotentialProblem(ps, u0, tspan)
    return SciMLBase.EnsembleProblem(
        base_prob;
        prob_func = (prob, ctx) -> DE.remake(prob, u0 = u0_vec[ctx.sim_id]),
        output_func = (sol, ctx) -> (sol[end], false),
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
    every_nth(iter, n)

Return a lazy iterator over every `n`-th element of `iter`.

# Arguments
- `iter`: any iterable collection
- `n`: step size (positive integer)

# Returns
A generator yielding `iter[1], iter[1+n], iter[1+2n], …`.
"""
function every_nth(iter, n::Integer)
    return (v for (i, v) in enumerate(iter) if i % n == 0)
end

_adjacent_pairs(x) = zip(x[1:(end - 1)], x[2:end])

"""
    define_directions(x, y)

Classify the direction of each segment of a parametric curve `(x, y)`.

# Arguments
- `x`: vector of x-coordinates (must have length ≥ 2)
- `y`: vector of y-coordinates (same length as `x`)

# Returns
A `Vector{Symbol}` with direction labels (`:topright`, `:bottomleft`,
`:horizontal`, `:vertical`, `:zero`, etc.) for each segment.

# Throws
- `ArgumentError` if `length(x) != length(y)` or `length(x) < 2`
"""
function define_directions(x, y)::Vector{Symbol}
    if length(x) != length(y)
        throw(ArgumentError("length(x) != length(y)"))
    end
    if length(x) < 2
        throw(ArgumentError("need at least 2 points to define directions, got $(length(x))"))
    end

    directions = Vector{Symbol}(undef, length(x))

    pairs = zip(_adjacent_pairs(x), _adjacent_pairs(y))
    for (k, ((u₁, u₂), (ux₁, ux₂))) in enumerate(pairs)
        s₁ = sign(u₂ - u₁)
        s₂ = sign(ux₂ - ux₁)
        if s₁ == 0 && s₂ == 0
            directions[k] = :zero
        elseif s₁ == 0
            directions[k] = :vertical
        elseif s₂ == 0
            directions[k] = :horizontal
        elseif s₁ == 1.0 && s₂ == 1.0
            directions[k] = :topright
        elseif s₁ == 1.0 && s₂ == -1.0
            directions[k] = :bottomright
        elseif s₁ == -1.0 && s₂ == 1.0
            directions[k] = :topleft
        elseif s₁ == -1.0 && s₂ == -1.0
            directions[k] = :bottomleft
        else
            throw(ArgumentError("unexpected direction combination"))
        end
    end
    directions[end] = directions[end - 1]

    for k in 1:(length(directions) - 1)
        if directions[k + 1] == :vertical || directions[k + 1] == :horizontal
            directions[k + 1] = directions[k]
        end
    end

    return directions
end

"""
    constant_runs(xs)

Split an array into intervals of equal consecutive values (constant runs).

# Arguments
- `xs`: vector of values

# Returns
A vector of overlapping unit-range intervals `[l:r, …]` marking segments
where values are equal.
"""
function constant_runs(xs)
    if isempty(xs)
        throw(ArgumentError("empty input: need at least 1 element to define intervals"))
    end
    checkpoints = [1]
    for (k, (a, b)) in enumerate(_adjacent_pairs(xs))
        if a != b
            push!(checkpoints, k)
            push!(checkpoints, k)
        end
    end
    push!(checkpoints, length(xs))

    intervals = Vector{UnitRange{Int}}(undef, length(checkpoints) ÷ 2)
    for (k, (l, r)) in enumerate(Iterators.partition(checkpoints, 2))
        intervals[k] = l:r
    end

    return intervals
end

"""
    monotonicity_intervals(xs, ys)

Convenience composition of [`define_directions`](@ref) and [`constant_runs`](@ref).

Classify curve direction at each point via `define_directions(xs, ys)`, then
split the result into intervals of equal direction values.

# Arguments
- `xs`, `ys`: coordinate vectors of equal length

# Returns
A vector of overlapping unit-range intervals `[l:r, …]` marking segments
of constant direction.
"""
function monotonicity_intervals(xs, ys)
    return define_directions(xs, ys) |> constant_runs
end

"""
    fmt(x)

Round a number to 2 decimal places for display.

# Arguments
- `x`: numeric value

# Returns
Rounded value (same type as input).
"""
fmt(x) = round(x; digits = 2)

"""
    find_interpolations_intersections(i₁, i₂, x_range)

Find x-coordinates where two interpolation objects intersect.

# Arguments
- `i₁`, `i₂`: callable interpolation objects (e.g. `Interpolations.Extrapolation`)
- `x_range`: `(xmin, xmax)` search interval

# Returns
Vector of intersection x-values within `x_range`.
"""
function find_interpolations_intersections(i₁, i₂, x_range::Tuple{T, T}) where {T}
    if x_range[1] == x_range[2]
        return []
    end
    if x_range[1] > x_range[2]
        x_range = (x_range[2], x_range[1])
    end
    f(x) = i₁(x) - i₂(x)
    return Roots.find_zeros(f, x_range)
end

"""
    find_polynomials_intersections(p₁, p₂, x_range)

Find real roots of `p₁ - p₂` within a given interval.

# Arguments
- `p₁`, `p₂`: `Polynomials.Polynomial` objects
- `x_range`: `(xmin, xmax)` search interval

# Returns
Vector of real intersection x-values within `x_range`.
Returns empty vector if coefficients are non-finite.
"""
function find_polynomials_intersections(p₁::Polynomials.Polynomial, p₂::Polynomials.Polynomial, x_range::Tuple{T, T}) where {T}
    p = p₁ - p₂
    if any(!isfinite, p.coeffs)
        return []
    end
    roots = Polynomials.roots(p)
    if eltype(roots) <: Complex
        roots = filter(r -> isapprox(r.im, 0.0), roots)
        roots = map(r -> r.re, roots)
    end
    return filter(r -> x_range[1] < r < x_range[2], roots)
end

"""
    find_intersections(data; interpolation)

Find intersection points `(u, u′)` of the parametric curves `γ₋` and `γ₊`.

# Arguments
- `data`: `DataFrame` with columns `um`, `uxm`, `up`, `uxp` (e.g. from
  [`find_parametric_curves`](@ref))
- `interpolation`: `:Polynomial` (cubic fit, default) or `:Linear`
  (linear interpolation)

# Returns
Vector of `(u, u′)` tuples at curve intersections.
"""
function find_intersections(data::DataFrame; interpolation::Symbol = :Polynomial)
    # Split each parametric curve into direction-based monotonic segments
    # (overlapping unit ranges of constant direction). Pairwise comparison
    # of only those segments whose x-ranges overlap avoids unnecessary work.
    isₘ = monotonicity_intervals(data.um, data.uxm)
    isₚ = monotonicity_intervals(data.up, data.uxp)

    intersections = Tuple{Float64, Float64}[]
    for (iₘ, iₚ) in Iterators.product(isₘ, isₚ)
        um = data.um[iₘ]
        up = data.up[iₚ]
        if maximum(um) < minimum(up) || maximum(up) < minimum(um)
            continue
        end
        if length(um) < 2 || length(up) < 2
            continue
        end

        uxm = data.uxm[iₘ]
        uxp = data.uxp[iₚ]

        x_range = (max(minimum(um), minimum(up)), min(maximum(um), maximum(up)))

        if (interpolation == :Polynomial)
            fₘ = Polynomials.fit(um, uxm, 3)
            fₚ = Polynomials.fit(up, uxp, 3)

            if !(any(isfinite, fₘ.coeffs) && any(isfinite, fₚ.coeffs))
                continue
            end

            roots = find_polynomials_intersections(fₘ, fₚ, x_range)
        elseif (interpolation == :Linear)
            ums = sort(um)
            if ums != um
                uxm = reverse(uxm)
            end

            ups = sort(up)
            if ups != up
                uxp = reverse(uxp)
            end
            fₘ = Interpolations.linear_interpolation(ums, uxm)
            fₚ = Interpolations.linear_interpolation(ups, uxp)

            roots = find_interpolations_intersections(fₘ, fₚ, x_range)
        else
            throw(ArgumentError("unknown interpolation type: $interpolation"))
        end

        for u in roots
            ux = (fₘ(u) + fₚ(u)) / 2
            push!(intersections, (u, ux))
        end
    end

    intersections = _deduplicate(intersections)
    return intersections
end

function _deduplicate(v::Vector{T}) where {T}
    isempty(v) && return v
    sort!(v, by = x -> x[1])
    result = [v[1]]
    atol = sqrt(eps(eltype(T)))
    for i in 2:length(v)
        if !isapprox(v[i][1], v[i - 1][1], atol = atol)
            push!(result, v[i])
        end
    end
    return result
end

"""
    nonlinear_range(start, stop; length)

Create a range with cubic spacing (more points near zero).

# Arguments
- `start`, `stop`: range endpoints
- `length`: number of points (keyword argument)

# Returns
Vector of `length` points with density concentrated near zero.
"""
function nonlinear_range(start::T, stop::T; length::Integer) where {T}
    f, f_inv = (x -> x^(1 / 3)), (x -> x^3)

    x = range(sign(start) * f_inv(abs(start)), sign(stop) * f_inv(abs(stop)); length = length)
    return sign.(x) .* f.(abs.(x))
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

"""
    plot_u_ux_diagram(data; save_path, linewidth, title)

Plot the `(u, u′)` phase diagram showing the parametric curves `γ₋` and `γ₊`.

Requires Plots.jl and CSV.jl to be loaded (`using Plots, CSV`).

# Arguments
- `data`: `DataFrame` with columns `um`, `uxm`, `up`, `uxp`
- `save_path`: (optional) directory path for saving plot and CSV data
- `linewidth`: line width for the curves (default: `0.5`)
- `title`: (optional) plot title and filename stem

# Returns
A `Plots.Plot` object (when Plots.jl is loaded).
"""
function plot_u_ux_diagram(data; save_path = nothing, linewidth = 0.5, title = nothing)
    error(
        """
        plot_u_ux_diagram requires Plots.jl and CSV.jl.
        Load them with `using Plots, CSV` and re-run.
        """
    )
end

end
