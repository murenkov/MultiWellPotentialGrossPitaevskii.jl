using MultiWellPotentialGrossPitaevskii
using Test
import StaticArrays as SA

try
    import Plots
catch
    @warn "Plots not available; skipping plot tests"
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

        @test_throws AssertionError define_directions([1, 2], [1])
    end

    @testset "monotonicity_intervals" begin
        intervals = monotonicity_intervals([:a, :a, :a])
        @test intervals == [1:3]

        intervals2 = monotonicity_intervals([:a, :b, :c])
        @test intervals2 == [1:1, 1:2, 2:3]

        intervals3 = monotonicity_intervals([:a, :a, :b, :b])
        @test length(intervals3) >= 1
    end

    @testset "define_directions + monotonicity_intervals" begin
        a = 0.01
        ts = range(0, 2pi, length = 200)
        xs = [exp(a * t) * cos(t) for t in ts]
        ys = [exp(a * t) * sin(t) for t in ts]

        dirs = define_directions(xs, ys)
        @test all(dirs[1:50] .== :topleft)
        @test all(dirs[51:100] .== :bottomleft)
        @test all(dirs[101:150] .== :bottomright)
        @test all(dirs[151:end] .== :topright)

        intervals = monotonicity_intervals(dirs)
        @test intervals == [1:50, 50:100, 100:150, 150:200]
    end

    @testset "fmt" begin
        @test fmt(3.14159) == 3.14
        @test fmt(0.0) == 0.0
        @test fmt(-1.234) == -1.23
        @test fmt(100.0) == 100.0
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

    @testset "find_polynomials_intersections" begin
        using Polynomials
        p1 = Polynomial([-1, 0, 1])   # x^2 - 1
        p2 = Polynomial([1, 0, -1])   # -x^2 + 1
        roots = find_polynomials_intersections(p1, p2, (-10.0, 10.0))
        @test length(roots) == 2
        @test -1.0 in roots || any(r -> isapprox(r, -1.0, atol = 1.0e-10), roots)
        @test 1.0 in roots || any(r -> isapprox(r, 1.0, atol = 1.0e-10), roots)

        roots2 = find_polynomials_intersections(p1, p2, (0.5, 10.0))
        @test length(roots2) == 1
        @test roots2[1] ≈ 1.0

        roots3 = find_polynomials_intersections(p1, p2, (2.0, 10.0))
        @test roots3 == []
    end

    @testset "find_interpolations_intersections" begin
        using Interpolations
        xs = 0.0:4.0
        f1 = linear_interpolation(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
        f2 = linear_interpolation(xs, [4.0, 3.0, 2.0, 1.0, 0.0])
        roots = find_interpolations_intersections(f1, f2, (0.0, 4.0))
        @test length(roots) >= 1
        @test any(r -> isapprox(r, 2.0, atol = 1.0e-10), roots)

        empty = find_interpolations_intersections(f1, f2, (1.0, 1.0))
        @test empty == []
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
        intersections = find_intersections(data; interpolation = :Polynomial)
        @test intersections isa Vector
    end

    @testset "finish_points" begin
        using DataFrames
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

        # GPU backend without CUDA
        @test_throws ErrorException finish_points(Cs, ps, (-10.0, 0.0); backend = GPU())
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
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = "../escape")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = "..\\escape")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = "sub/../escape")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = "sub\\..\\escape")

            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = 42)
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "foo/bar")
            @test_throws ArgumentError plot_u_ux_diagram(data; save_path = ".", title = "foo\\bar")
        end
    end

end
