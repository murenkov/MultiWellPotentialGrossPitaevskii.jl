using MultiWellPotentialGrossPitaevskii
using Test
import StaticArrays as SA
import OrdinaryDiffEq
import DiffEqGPU
import MultiWellPotentialGrossPitaevskii: define_directions, constant_runs, _deduplicate, _get_solver, every_nth

try
    import Plots
catch
    @warn "Plots not available; skipping plot tests"
end

try
    import CUDA
catch
    @warn "CUDA not available; skipping GPU tests"
end

@testset "MultiWellPotentialGrossPitaevskii" begin

    @testset "V₁" begin
        @test V₁(0.0, 1.0) == -1.0
        @test V₁(0.0, 2.0) == -2.0
        @test V₁(Inf, 1.0) == 0.0
        @test V₁(-Inf, 1.0) ≈ -0.0
        @test V₁(0.0, 5.0) == -5.0
    end

    @testset "V" begin
        @test V(0.0, [1.0], [0.0]) == -1.0
        @test V(Inf, [1.0, 1.0], [-1.0, 1.0]) == 0.0
        @test V(0.0, [1.0, 1.0], [0.0, 0.0]) == -2.0
        @test V(0.0, [2.0, 3.0], [0.0, 0.0]) == -5.0
    end

    @testset "singular / regular" begin
        @test singular([NaN, 1.0]) == true
        @test singular([1.0, Inf]) == true
        @test singular([1.0, 2.0]) == false
        @test singular([11.0]) == true
        @test regular([1.0, 2.0]) == true
        @test regular([NaN]) == false
        @test regular([Inf]) == false

        @test singular([5.0]; cutoff = 3.0) == true
        @test singular([5.0]; cutoff = 10.0) == false
        @test singular([5.0]; cutoff = 5.0) == false
        @test regular([5.0]; cutoff = 3.0) == false
        @test regular([5.0]; cutoff = 10.0) == true
        @test regular([5.0]; cutoff = 5.0) == true
    end

    @testset "MultiWellParams" begin
        as = SA.SVector(1.0, 2.0)
        ds = SA.SVector(-1.0, 1.0)
        p = MultiWellParams(-0.5, as, ds)
        @test p.ω == -0.5
        @test p.as == as
        @test p.ds == ds
        @test p isa MultiWellParams{Float64, 2}
    end

    @testset "multiwell_potential_equation" begin
        as = SA.SVector(0.0, 0.0)
        ds = SA.SVector(0.0, 0.0)
        p = MultiWellParams(0.0, as, ds)
        u = SA.SVector(1.0, 0.0)
        du = multiwell_potential_equation(u, p, 0.0)
        @test du isa SA.SVector{2, Float64}
        @test du[1] ≈ 0.0
        @test du[2] ≈ 1.0

        pω = MultiWellParams(-1.0, as, ds)
        du2 = multiwell_potential_equation(u, pω, 0.0)
        @test du2[2] ≈ 2.0

        as2 = SA.SVector(4.0)
        ds2 = SA.SVector(0.0)
        p2 = MultiWellParams(-1.0, as2, ds2)
        du3 = multiwell_potential_equation(u, p2, 0.0)
        V_at_0 = -4.0 * (1 / cosh(0))^2
        expected_du2 = -(-1.0 - V_at_0) * 1.0 + 1.0^3
        @test du3[2] ≈ expected_du2
    end

    @testset "every_nth" begin
        @test collect(every_nth(1:10, 2)) == [2, 4, 6, 8, 10]
        @test collect(every_nth(1:10, 3)) == [3, 6, 9]
        @test collect(every_nth(1:5, 1)) == [1, 2, 3, 4, 5]
        @test collect(every_nth(1:5, 10)) == []
        @test collect(every_nth([], 2)) == []
    end

    @testset "define_directions" begin
        d1 = define_directions([1, 2, 3], [1, 2, 3])
        @test d1 == [:topright, :topright, :topright]

        d2 = define_directions([3, 2, 1], [3, 2, 1])
        @test d2 == [:bottomleft, :bottomleft, :bottomleft]

        d3 = define_directions([1, 1, 1], [1, 2, 3])
        @test d3 == [:vertical, :vertical, :vertical]

        d4 = define_directions([1, 2, 3], [1, 1, 1])
        @test d4 == [:horizontal, :horizontal, :horizontal]

        @test_throws ArgumentError define_directions([1, 2], [1])
        @test_throws ArgumentError define_directions([1], [1])
        @test_throws ArgumentError define_directions(Float64[], Float64[])
    end

    @testset "constant_runs" begin
        intervals = constant_runs([:a, :a, :a])
        @test intervals == [1:3]

        intervals2 = constant_runs([:a, :b, :c])
        @test intervals2 == [1:1, 1:2, 2:3]

        intervals3 = constant_runs([:a, :a, :b, :b])
        @test intervals3 == [1:2, 2:4]

        @test_throws ArgumentError constant_runs([])
        @test constant_runs([:a]) == [1:1]
    end

    @testset "_deduplicate" begin
        @test _deduplicate(Tuple{Float64, Float64}[]) == []
        @test _deduplicate([(1.0, 2.0)]) == [(1.0, 2.0)]
        @test _deduplicate([(3.0, 1.0), (1.0, 2.0), (2.0, 3.0)]) == [(1.0, 2.0), (2.0, 3.0), (3.0, 1.0)]

        # Exact duplicate by both elements
        @test _deduplicate([(1.0, 1.0), (1.0, 1.0), (2.0, 3.0)]) == [(1.0, 1.0), (2.0, 3.0)]

        # Exact duplicate by first element only (different ux → kept)
        @test _deduplicate([(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)]) == [(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

        # Near-duplicate in both elements within sqrt(eps) tolerance
        @test _deduplicate([(1.0, 1.0), (1.0 + 1.0e-10, 1.0 + 1.0e-10), (2.0, 3.0)]) == [(1.0, 1.0), (2.0, 3.0)]

        # Distinguishable values kept
        @test _deduplicate([(1.0, 1.0), (1.0 + 1.0e-6, 2.0), (2.0, 3.0)]) == [(1.0, 1.0), (1.0 + 1.0e-6, 2.0), (2.0, 3.0)]
    end

    @testset "_get_solver" begin
        alg, ensemble_alg = _get_solver(CPU())
        @test alg isa OrdinaryDiffEq.Vern9
        @test ensemble_alg isa DiffEqGPU.EnsembleCPUArray
        if applicable(_get_solver, GPU())
            alg_gpu, ensemble_alg_gpu = _get_solver(GPU())
            @test alg_gpu isa DiffEqGPU.GPUVern9
            @test ensemble_alg_gpu isa DiffEqGPU.EnsembleGPUKernel
        else
            @test !applicable(_get_solver, GPU())
        end
    end

    @testset "monotonicity_intervals (composite: define_directions + constant_runs)" begin
        a = 0.01
        ts = range(0, 2pi, length = 200)
        xs = [exp(a * t) * cos(t) for t in ts]
        ys = [exp(a * t) * sin(t) for t in ts]

        intervals = monotonicity_intervals(xs, ys)
        @test intervals == [1:50, 50:100, 100:150, 150:200]
    end

    @testset "fmt" begin
        @test MultiWellPotentialGrossPitaevskii.fmt(3.14159) == 3.14
        @test MultiWellPotentialGrossPitaevskii.fmt(0.0) == 0.0
        @test MultiWellPotentialGrossPitaevskii.fmt(-1.234) == -1.23
        @test MultiWellPotentialGrossPitaevskii.fmt(100.0) == 100.0
    end

    @testset "nonlinear_range" begin
        r = nonlinear_range(0.0, 8.0; length = 5)
        @test length(r) == 5
        @test r[1] == 0.0
        @test r[end] ≈ 8.0
        @test all(diff(r) .>= 0)

        r2 = nonlinear_range(-8.0, 0.0; length = 5)
        @test length(r2) == 5
        @test r2[1] ≈ -8.0
        @test r2[end] == 0.0
        @test all(diff(r2) .>= 0)
    end

    @testset "find_intersections" begin
        using DataFrames
        data = DataFrame(
            um = [-2.0, -1.0, 0.0, 1.0, 2.0],
            uxm = [4.0, 1.0, 0.0, 1.0, 4.0],
            up = [-2.0, -1.0, 0.0, 1.0, 2.0],
            uxp = [-4.0, -1.0, 0.0, -1.0, -4.0],
            C = [1, 2, 3, 4, 5],
        )
        intersections = find_intersections(data)
        @test intersections isa Vector
        @test length(intersections) == 1
        @test intersections[1][1] ≈ 0.0
        @test intersections[1][2] ≈ 0.0
    end

    @testset "finish_points" begin
        using DataFrames

        @testset "Float32 MultiWellParams" begin
            Cs_range = 1_000
            Cs = range(-Cs_range, Cs_range; length = 10)
            as = SA.@SVector [8.66, 8.66, 8.66]
            ds = SA.@SVector [-π, 0, π]
            ps = MultiWellParams{Float32, 3}(-4.5, as, ds)
            result = finish_points(Cs, ps, (-10.0f0, 0.0f0))
            @test result isa DataFrame
            @test propertynames(result) == [:C, :u, :ux]
            @test size(result, 1) == 10
            expected_u = Float32[
                2.84003, 1.36035, 0.723363, 0.36345, 0.111335,
                -0.111335, -0.36345, -0.723363, -1.36035, -2.84003,
            ]
            expected_ux = Float32[
                3.42684, -0.41152, -0.81805, -0.592605, -0.20862,
                0.20862, 0.592605, 0.81805, 0.41152, -3.42684,
            ]
            for i in 1:10
                @test isapprox(result.u[i], expected_u[i]; atol = 0.001)
                @test isapprox(result.ux[i], expected_ux[i]; atol = 0.001)
            end
        end

        ps = MultiWellParams(-1.0, SA.SVector(0.0), SA.SVector(0.0))

        # Basic forward integration
        Cs = [0.01, 0.1]
        result = finish_points(Cs, ps, (-10.0, 0.0))
        @test result isa DataFrame
        @test propertynames(result) == [:C, :u, :ux]
        @test size(result, 1) == 2
        @test all(regular, [result.u result.ux])

        # For very small C, linear approximation holds: u ≈ C, ux ≈ √(-ω) * u
        @test result.u[1] ≈ 0.01 atol = 0.005
        @test result.u[2] ≈ 0.1  atol = 0.02
        @test result.ux[1] ≈ result.u[1] atol = 0.005
        @test result.ux[2] ≈ result.u[2] atol = 0.02

        # Reverse tspan (10.0, 0.0) — ux is negated by s = sign(0 - 10) = -1
        result_rev = finish_points(Cs, ps, (10.0, 0.0))
        @test result_rev isa DataFrame
        @test propertynames(result_rev) == [:C, :u, :ux]
        @test size(result_rev, 1) == 2
        @test all(regular, [result_rev.u result_rev.ux])
        @test result_rev.u ≈ result.u atol = 0.01
        @test result_rev.ux ≈ -result.ux atol = 0.01

        if applicable(_get_solver, GPU())
            result_gpu = finish_points(Cs, ps, (-10.0, 0.0); backend = GPU())
            @test result_gpu isa DataFrame
            @test propertynames(result_gpu) == [:C, :u, :ux]
            @test size(result_gpu, 1) == 2
            @test all(regular, [result_gpu.u result_gpu.ux])
            @test result_gpu.u ≈ result.u atol = 0.05
            @test result_gpu.ux ≈ result.ux atol = 0.05
        else
            @test_throws ErrorException finish_points(Cs, ps, (-10.0, 0.0); backend = GPU())
        end
    end

    @testset "find_parametric_curves" begin
        using DataFrames
        ps = MultiWellParams(-1.0, SA.SVector(0.0), SA.SVector(0.0))

        # Basic call
        Cs = [0.01, 0.1]
        result = find_parametric_curves(Cs, ps)
        @test result isa DataFrame
        @test propertynames(result) == [:C, :um, :uxm, :up, :uxp]
        @test 1 ≤ size(result, 1) ≤ 2
        @test all(regular, [result.um result.uxm result.up result.uxp])

        # For zero potential: um ≈ up and uxm ≈ -uxp
        @test result.um ≈ result.up atol = 0.005
        @test result.uxm ≈ -result.uxp atol = 0.005

        # Singular filtering: large Cs may produce NaN/inf
        Cs_large = [20.0, 50.0]
        result_large = find_parametric_curves(Cs_large, ps)
        @test result_large isa DataFrame
        @test propertynames(result_large) == [:C, :um, :uxm, :up, :uxp]
        @test size(result_large, 1) ≤ 2
    end

    @testset "parametric curve intersection counts" begin
        using DataFrames

        @testset "N=2 a≈8.66 ds=±π/2 ω∈[-6.1,-3.0]" begin
            as = SA.SVector(8.661554517943312, 8.661554517943312)
            ds = SA.SVector(-π / 2, π / 2)
            Cs = range(-500, 500; length = 10_000)
            for ω in (-6.1, -4.45, -3.0)
                @testset "ω=$ω" begin
                    ps = MultiWellParams(ω, as, ds)
                    data = find_parametric_curves(Cs, ps)
                    @test length(find_intersections(data)) == 9
                end
            end
        end

        @testset "N=2 a≈8.66 ds=±π/2 ω∈[-5.0,-4.5]" begin
            as = SA.SVector(8.661554517943312, 8.661554517943312)
            ds = SA.SVector(-π / 2, π / 2)
            Cs = range(-500, 500; length = 10_000)
            for ω in (-5.0, -4.8, -4.5)
                @testset "ω=$ω" begin
                    ps = MultiWellParams(ω, as, ds)
                    data = find_parametric_curves(Cs, ps)
                    @test length(find_intersections(data)) == 9
                end
            end
        end

        @testset "N=2 a≈11.55 ds=±π/2 ω∈[-8.6,-4.5]" begin
            as = SA.SVector(11.548739357257748, 11.548739357257748)
            ds = SA.SVector(-π / 2, π / 2)
            Cs = range(-500, 500; length = 10_000)
            for ω in (-8.6, -5.0, -4.5)
                @testset "ω=$ω" begin
                    ps = MultiWellParams(ω, as, ds)
                    data = find_parametric_curves(Cs, ps)
                    @test length(find_intersections(data)) == 9
                end
            end
        end

        @testset "N=3 a≈8.66 ds=[-π,0,π] ω∈[-4.7,-4.4]" begin
            as = SA.SVector(8.661554517943312, 8.661554517943312, 8.66155451794331)
            ds = SA.SVector(-π, 0, π)
            Cs = range(-7000, 7000; length = 10_000)
            for ω in (-4.7, -4.55, -4.4)
                @testset "ω=$ω" begin
                    ps = MultiWellParams(ω, as, ds)
                    data = find_parametric_curves(Cs, ps)
                    @test length(find_intersections(data)) == 27
                end
            end
        end

        @testset "N=2 a≈11.55 ds=±π/2 ω∈[-1.85,-1.75]" begin
            as = SA.SVector(11.548739357257748, 11.548739357257748)
            ds = SA.SVector(-π / 2, π / 2)
            Cs = range(-1000, 1000; length = 50_000)
            for ω in (-1.85, -1.75)
                @testset "ω=$ω" begin
                    ps = MultiWellParams(ω, as, ds)
                    data = find_parametric_curves(Cs, ps)
                    @test length(find_intersections(data)) == 25
                end
            end
        end
    end

    @testset "parametric curve intersection counts (GPU)" begin
        using DataFrames

        if applicable(_get_solver, GPU())
            @testset "N=2 a≈8.66 ds=±π/2 ω∈[-6.1,-3.0]" begin
                as = SA.SVector(8.661554517943312, 8.661554517943312)
                ds = SA.SVector(-π / 2, π / 2)
                Cs = range(-500, 500; length = 10_000)
                for ω in (-6.1, -4.45, -3.0)
                    @testset "ω=$ω" begin
                        ps = MultiWellParams(ω, as, ds)
                        data = find_parametric_curves(Cs, ps; backend = GPU())
                        @test length(find_intersections(data)) == 9
                    end
                end
            end

            @testset "N=3 a≈8.66 ds=[-π,0,π] ω∈[-4.7,-4.4]" begin
                as = SA.SVector(8.661554517943312, 8.661554517943312, 8.66155451794331)
                ds = SA.SVector(-π, 0, π)
                Cs = range(-7000, 7000; length = 10_000)
                for ω in (-4.7, -4.55, -4.4)
                    @testset "ω=$ω" begin
                        ps = MultiWellParams(ω, as, ds)
                        data = find_parametric_curves(Cs, ps; backend = GPU())
                        @test length(find_intersections(data)) == 27
                    end
                end
            end
        else
            @testset "GPU backend throws without CUDA" begin
                as = SA.SVector(8.661554517943312, 8.661554517943312)
                ds = SA.SVector(-π / 2, π / 2)
                Cs = range(-500, 500; length = 10_000)
                ps = MultiWellParams(-4.45, as, ds)
                @test_throws ErrorException find_parametric_curves(Cs, ps; backend = GPU())
            end
        end
    end

    function _test_plot_ext(fn)
        if isdefined(@__MODULE__, :Plots)
            ext = Base.get_extension(MultiWellPotentialGrossPitaevskii, :MWPExtPlots)
            if ext !== nothing
                fn()
            else
                @test_throws ErrorException fn()
            end
        else
            @test_throws ErrorException fn()
        end
    end

    @testset "plot_u_ux_diagram" begin
        using DataFrames
        data = DataFrame(
            C = [1.0, 2.0, 3.0],
            um = [1.0, 2.0, 3.0],
            uxm = [1.0, 2.0, 3.0],
            up = [1.0, 2.0, 3.0],
            uxp = [-1.0, -2.0, -3.0],
        )
        _test_plot_ext() do
            p = plot_u_ux_diagram(data)
            @test p isa Plots.Plot

            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = "")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = 42)

            @testset "path resolution instead of TOCTOU check" begin
                mktempdir() do dir
                    target = joinpath(dir, "realtarget")
                    mkpath(joinpath(dir, "escape"))
                    p = plot_u_ux_diagram(data; save_path = joinpath(dir, "sub", "..", "escape"), title = "resolved1")
                    @test p isa Plots.Plot
                    @test isfile(joinpath(dir, "escape", "resolved1-diagram-data.csv"))
                    @test isfile(joinpath(dir, "escape", "resolved1.svg"))

                    p2 = plot_u_ux_diagram(data; save_path = joinpath(dir, "..", basename(dir), "newdir"), title = "resolved2")
                    @test p2 isa Plots.Plot
                    @test isdir(joinpath(dir, "newdir"))
                    @test isfile(joinpath(dir, "newdir", "resolved2-diagram-data.csv"))
                    @test isfile(joinpath(dir, "newdir", "resolved2.svg"))
                end
            end

            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = 42)
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "foo/bar")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "foo\\bar")

            @testset "symlink resolution" begin
                mktempdir() do dir
                    real_sub = joinpath(dir, "realtarget")
                    mkpath(real_sub)
                    link = joinpath(dir, "link")
                    try
                        symlink(real_sub, link)
                        p = plot_u_ux_diagram(data; save_path = link, title = "symlinktest")
                        @test p isa Plots.Plot
                        @test isdir(real_sub)
                        @test isfile(joinpath(real_sub, "symlinktest-diagram-data.csv"))
                        @test isfile(joinpath(real_sub, "symlinktest.svg"))
                    catch e
                        if e isa SystemError
                            @warn "symlinks not supported on this platform; skipping"
                        else
                            rethrow(e)
                        end
                    end
                end
            end
        end
    end

    @testset "Property-based tests" begin
        @testset "nonlinear_range is monotonic" begin
            for _ in 1:200
                start = randn() * 10
                stop = randn() * 10
                len = rand(2:50)
                r = nonlinear_range(start, stop; length = len)
                @test length(r) == len
                @test r[1] ≈ start atol = 1.0e-10
                @test r[end] ≈ stop atol = 1.0e-10
                if start < stop
                    @test all(diff(r) .>= -1.0e-12)
                elseif start > stop
                    @test all(diff(r) .<= 1.0e-12)
                else
                    @test all(abs.(r .- start) .<= 1.0e-12)
                end
            end
        end

        @testset "_deduplicate result has no near-duplicates" begin
            for _ in 1:200
                n = rand(0:20)
                v = [(randn() * 10, randn() * 10) for _ in 1:n]
                result = _deduplicate(v)
                atol = sqrt(eps(Float64))
                # No two result elements should be close in both coordinates
                for i in 1:(length(result) - 1)
                    for j in (i + 1):length(result)
                        both_close = isapprox(result[i][1], result[j][1], atol = atol) &&
                            isapprox(result[i][2], result[j][2], atol = atol)
                        @test !both_close
                    end
                end
                # Every original element must have a close match in the result
                for elem in v
                    found = any(
                        isapprox(elem[1], r[1], atol = atol) && isapprox(elem[2], r[2], atol = atol)
                            for r in result
                    )
                    @test found
                end
            end
        end

        @testset "define_directions consistent with finite differences" begin
            for _ in 1:200
                n = rand(2:30)
                x = randn(n) * 10
                y = randn(n) * 10
                d = define_directions(x, y)
                @test length(d) == n
                @test d[end] == d[end - 1]
                # Compute raw direction for each segment from finite differences
                raw = Vector{Symbol}(undef, n - 1)
                for i in 1:(n - 1)
                    s1 = sign(x[i + 1] - x[i])
                    s2 = sign(y[i + 1] - y[i])
                    if iszero(s1) && iszero(s2)
                        raw[i] = :zero
                    elseif iszero(s1)
                        raw[i] = :vertical
                    elseif iszero(s2)
                        raw[i] = :horizontal
                    elseif s1 > 0 && s2 > 0
                        raw[i] = :topright
                    elseif s1 > 0 && s2 < 0
                        raw[i] = :bottomright
                    elseif s1 < 0 && s2 > 0
                        raw[i] = :topleft
                    else
                        raw[i] = :bottomleft
                    end
                end
                # Apply same post-processing as define_directions
                expected = vcat(raw, raw[end])
                for k in 1:(n - 1)
                    if expected[k + 1] == :vertical || expected[k + 1] == :horizontal
                        expected[k + 1] = expected[k]
                    end
                end
                @test d == expected
            end
        end
    end

end
