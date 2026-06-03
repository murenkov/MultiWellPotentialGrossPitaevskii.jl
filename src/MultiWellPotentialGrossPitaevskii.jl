module MultiWellPotentialGrossPitaevskii

import StaticArrays as SA
import DifferentialEquations as DE
import DiffEqGPU
import CUDA
import DataFrames: DataFrame, rename!, innerjoin
import ResultTypes: Result, AssertionError, unwrap
import IterTools
import Polynomials
import Interpolations
import Roots
import Plots
import CSV
using LaTeXStrings

export MultiWellParams, multiwell_potential_equation
export singular, regular
export every_nth, define_directions, monotonicity_intervals
export find_interpolations_intersections, find_polynomials_intersections
export find_intersections, nonlinear_range, fmt
export finish_points, find_parametric_curves
export V₁, V, plot_u_ux_diagram

@inline function V₁(t, A)
    return -A[1] * sech(t)^2
end

@inline function V(t, as, ds)
    return -sum(as .* sech.(t .- ds) .^ 2)
end

singular(x) = any(isnan.(x)) || any(abs.(x) .> 10.0)

regular(x) = !singular(x)

struct MultiWellParams{T, N}
    ω::T
    as::SA.SVector{N, T}
    ds::SA.SVector{N, T}
end

@inline function multiwell_potential_equation(u::SA.SVector{2, T}, p::MultiWellParams{T, N}, t::T)::SA.SVector{2, T} where {T <: Real, N}
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

function finish_points(
        Cs,
        ps::MultiWellParams{T, N},
        tspan::Tuple{T, T},
    ) where {T <: Real, N}
    (t₀, tₑ) = tspan
    ω = ps.ω
    s = sign(tₑ - t₀)
    u = Cs .* exp(s * √(-ω) * t₀)
    uₓ = s * √(-ω) .* u

    (t₀, tₑ) = tspan
    if tₑ < t₀
        tspan = (-t₀, tₑ)
        (t₀, tₑ) = tspan
        ps = MultiWellParams(ps.ω, ps.as, -reverse(ps.ds))
        u = Cs .* exp(-s * √(-ω) * t₀)
        uₓ = -s * √(-ω) .* u
    end

    f = multiwell_potential_equation
    u0 = SA.@SVector T[0.0, 0.0]
    u0_vec = [SA.SVector{2, T}(x, y) for (x, y) in zip(u, uₓ)]

    eproblem = DE.EnsembleProblem(
        DE.ODEProblem(f, u0, tspan, ps);
        prob_func = (prob, ctx) -> DE.remake(prob, u0 = u0_vec[ctx.sim_id]),
        output_func = (sol, ctx) -> (sol[end], false),
        safetycopy = false,
    )
    solutions = DE.solve(
        eproblem,
        DiffEqGPU.GPUVern9(),
        DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());
        dt = T(0.1),
        trajectories = length(u0_vec),
        adaptive = false,
        verbose = true,
        save_everystep = false,
        save_on = false,
        save_start = true,
        save_end = true,
    )

    return DataFrame(C = Cs, u = first.(solutions.u), ux = s * last.(solutions.u))
end

function every_nth(iter, n::Integer)
    f = Iterators.filter(pairs(iter)) do (k, v)
        k % n == 0
    end
    return [v for (k, v) in f]
end

function define_directions(x, y)::Result{Vector{Symbol}, AssertionError}
    if length(x) != length(y)
        return AssertionError()
    end

    directions = Vector{Symbol}(undef, first(size(x)))

    pairs = zip(IterTools.partition(x, 2, 1), IterTools.partition(y, 2, 1))
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
            throw(ErrorException)
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

function monotonicity_intervals(xs)
    checkpoints = [1]
    for (k, (a, b)) in enumerate(IterTools.partition(xs, 2, 1))
        if a != b
            push!(checkpoints, k)
            push!(checkpoints, k)
        end
    end
    push!(checkpoints, length(xs))

    intervals = Vector(undef, length(checkpoints) ÷ 2)
    for (k, (l, r)) in enumerate(Iterators.partition(checkpoints, 2))
        intervals[k] = l:r
    end

    return intervals
end

fmt(x) = round(x; digits = 2)

function find_interpolations_intersections(i₁, i₂, x_range::Tuple{T, T}) where {T}
    if x_range[1] == x_range[2]
        return []
    end
    if x_range[1] < x_range[2]
        x_range = (x_range[2], x_range[1])
    end
    f(x) = i₁(x) - i₂(x)
    return Roots.find_zeros(f, x_range)
end

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

function find_intersections(data::DataFrame; interpolation::Symbol = :Polynomial)
    isₘ = unwrap(define_directions(data.um, data.uxm)) |> monotonicity_intervals
    isₚ = unwrap(define_directions(data.up, data.uxp)) |> monotonicity_intervals

    intersections = []
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
            throw(ErrorException)
        end

        for u in roots
            ux = (fₘ(u) + fₚ(u)) / 2
            push!(intersections, (u, ux))
        end
    end

    return intersections
end

function nonlinear_range(start::T, stop::T; length::Integer) where {T}
    f, f_inv = (x -> x^(1 / 3)), (x -> x^3)

    x = range(sign(start) * f_inv(abs(start)), sign(stop) * f_inv(abs(stop)); length = length)
    return sign.(x) .* f.(abs.(x))
end

function find_parametric_curves(Cs, ps)
    pairs₋ = finish_points(Cs, ps, (-10.0f0, 0.0f0))
    filter!(row -> regular([row.u, row.ux]), pairs₋)

    pairs₊ = copy(pairs₋)
    pairs₊.ux = -pairs₊.ux
    filter!(row -> regular([row.u, row.ux]), pairs₊)

    rename!(pairs₋, :u => :um, :ux => :uxm)
    rename!(pairs₊, :u => :up, :ux => :uxp)
    return innerjoin(pairs₋, pairs₊; on = :C)
end

function plot_u_ux_diagram(data; save_path = nothing, linewidth = 0.5, title = nothing)
    plot = Plots.plot(title = title, xlabel = L"u(0)", ylabel = L"u'(0)")
    plot = Plots.plot!(data.um, data.uxm; label = L"γ_-", linewidth = linewidth)
    plot = Plots.plot!(data.up, data.uxp; label = L"γ_+", linewidth = linewidth)

    if save_path != nothing
        if !isdir(save_path)
            mkdir(save_path)
        end
        CSV.write("$(save_path)/$(title)-diagram-data.csv", data)
        Plots.savefig(plot, "$(save_path)/$(title).svg")
    end

    return plot
end

end
