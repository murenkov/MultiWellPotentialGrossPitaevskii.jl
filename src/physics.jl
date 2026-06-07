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
    singular(x; cutoff = 10.0)

Check if a value or array is "singular" (non-finite or diverging).

# Arguments
- `x`: value or array to check
- `cutoff`: magnitude threshold (default: 10.0)

# Returns
`true` if any element is `NaN` or has absolute value exceeding `cutoff`.
"""
function singular(x; cutoff = 10.0)
    for val in x
        if isnan(val) || abs(val) > cutoff
            return true
        end
    end
    return false
end

"""
    regular(x; cutoff = 10.0)

Inverse of [`singular`](@ref).

# Arguments
- `x`: value or array to check
- `cutoff`: magnitude threshold (default: 10.0)

# Returns
`true` if the value is finite and bounded (absolute value ≤ `cutoff`).
"""
regular(x; cutoff = 10.0) = !singular(x; cutoff = cutoff)

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
