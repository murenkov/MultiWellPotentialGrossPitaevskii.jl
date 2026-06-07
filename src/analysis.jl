"""
    _every_nth(iter, n)

Return a lazy iterator over every `n`-th element of `iter`.

# Arguments
- `iter`: any iterable collection
- `n`: step size (positive integer)

# Returns
A generator yielding `iter[1], iter[1+n], iter[1+2n], …`.
"""
function _every_nth(iter, n::Integer)
    return (v for (i, v) in enumerate(iter) if i % n == 0)
end

"""
    _adjacent_pairs(x)

Return an iterator over consecutive pairs `(x[i], x[i+1])` for `i = 1:length(x)-1`.

# Arguments
- `x`: vector with at least 2 elements

# Returns
A zip iterator yielding tuples `(x[i], x[i+1])` for each consecutive pair.
"""
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
        if iszero(s₁) && iszero(s₂)
            directions[k] = :zero
        elseif iszero(s₁)
            directions[k] = :vertical
        elseif iszero(s₂)
            directions[k] = :horizontal
        elseif s₁ > 0 && s₂ > 0
            directions[k] = :topright
        elseif s₁ > 0 && s₂ < 0
            directions[k] = :bottomright
        elseif s₁ < 0 && s₂ > 0
            directions[k] = :topleft
        elseif s₁ < 0 && s₂ < 0
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
    _fmt(x)

Round a number to 2 decimal places for display.

# Arguments
- `x`: numeric value

# Returns
Rounded value (same type as input).
"""
_fmt(x) = round(x; digits = 2)

# Check if a value is approximately zero by testing x ≈ -x (which implies 2x ≈ 0).
# This is used to detect zero-velocity points for self-intersection handling.
_is_approximately_zero(x; atol) = isapprox(x, -x, atol = atol)

"""
    find_intersections(data)

Find intersection points `(u, u′)` of the parametric curves `γ₋` and `γ₊`.

# Arguments
- `data`: `DataFrame` with columns `um`, `uxm`, `up`, `uxp` (e.g. from
  [`find_parametric_curves`](@ref))

# Returns
Vector of `(u, u′)` tuples at curve intersections.
"""

function find_intersections(data::DataFrame)
    um, uxm = data.um, data.uxm
    up, uxp = data.up, data.uxp
    n = length(um)

    result = Tuple{Float64, Float64}[]

    # Phase 1: Coincident data points between γ₋ and γ₊
    # Sort by u and sweep clusters to avoid O(n²)
    perm = sortperm(um)
    i = 1
    while i <= n
        cluster_start = i
        ui = um[perm[i]]
        while i <= n && um[perm[i]] <= ui + COINCIDENT_ATOL
            i += 1
        end
        for a in cluster_start:(i - 1)
            aa = perm[a]
            uia = um[aa]
            uxia = uxm[aa]
            for b in a:(i - 1)
                bb = perm[b]
                # Allow self-intersection at zero-velocity points (uxia ≈ 0)
                if a != b || _is_approximately_zero(uxia; atol = COINCIDENT_ATOL)
                    if isapprox(uxia, -uxm[bb], atol = COINCIDENT_ATOL)
                        push!(result, (uia, uxia))
                    end
                end
            end
        end
    end

    # Phase 2: Segment crossings between γ₋ and γ₊
    # Spatial grid on u to avoid O(n²)
    if n ≥ 2
        u_min_all = min(minimum(um), minimum(up))
        u_max_all = max(maximum(um), maximum(up))
        range_len = u_max_all - u_min_all
        range_len == 0 && (range_len = 1.0)

        B = max(1, round(Int, sqrt(n)))
        bin_width = range_len / B

        bins = [Int[] for _ in 1:B]
        for j in 1:(n - 1)
            lo_j = min(up[j], up[j + 1])
            hi_j = max(up[j], up[j + 1])
            first_bin = max(1, floor(Int, (lo_j - u_min_all) / bin_width) + 1)
            last_bin = min(B, floor(Int, (hi_j - u_min_all) / bin_width) + 1)
            for b in first_bin:last_bin
                push!(bins[b], j)
            end
        end

        lk = ReentrantLock()
        Threads.@threads :static for i in 1:(n - 1)
            lo_i = min(um[i], um[i + 1])
            hi_i = max(um[i], um[i + 1])
            first_bin = max(1, floor(Int, (lo_i - u_min_all) / bin_width) + 1)
            last_bin = min(B, floor(Int, (hi_i - u_min_all) / bin_width) + 1)

            a1x, a1y = um[i], uxm[i]
            a2x, a2y = um[i + 1], uxm[i + 1]
            v1x, v1y = a2x - a1x, a2y - a1y

            for b in first_bin:last_bin
                for j in bins[b]
                    lo_j = min(up[j], up[j + 1])
                    hi_j = max(up[j], up[j + 1])
                    if hi_i < lo_j - CROSSING_ATOL || lo_i > hi_j + CROSSING_ATOL
                        continue
                    end

                    b1x, b1y = up[j], uxp[j]
                    b2x, b2y = up[j + 1], uxp[j + 1]

                    v2x, v2y = b2x - b1x, b2y - b1y
                    denom = v1x * v2y - v1y * v2x
                    if abs(denom) < DENOM_EPS
                        continue
                    end

                    dx = b1x - a1x
                    dy = b1y - a1y
                    t = (dx * v2y - dy * v2x) / denom
                    s = (dx * v1y - dy * v1x) / denom

                    if t > CROSSING_ATOL && t < 1 - CROSSING_ATOL && s > CROSSING_ATOL && s < 1 - CROSSING_ATOL
                        lock(lk) do
                            push!(result, (a1x + t * v1x, a1y + t * v1y))
                        end
                    end
                end
            end
        end
    end

    return _deduplicate(result)
end

"""
    _deduplicate(v) -> Vector

Deduplicate a sorted vector of tuples within a tolerance,
merging intersection points that are approximately equal.

# Arguments
- `v`: sorted vector of `(Float64, Float64)` tuples

# Returns
A new vector with approximate duplicates removed.
"""
function _deduplicate(v::Vector{T}) where {T}
    isempty(v) && return v
    sort!(v)
    result = [v[1]]
    atol = sqrt(eps(eltype(T)))
    for i in 2:length(v)
        if !(isapprox(v[i][1], v[i - 1][1], atol = atol) && isapprox(v[i][2], v[i - 1][2], atol = atol))
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
