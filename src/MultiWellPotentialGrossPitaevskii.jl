module MultiWellPotentialGrossPitaevskii

using Reexport
import StaticArrays as SA
import OrdinaryDiffEq as DE
import DiffEqGPU
using SciMLBase
import SciMLLogging
using DataFrames: DataFrame, rename!, innerjoin

using Polynomials
using Roots

@reexport using SciMLBase

export MultiWellParams, MultiWellPotentialProblem, multiwell_potential_equation
export singular, regular
export every_nth, monotonicity_intervals
export find_intersections, nonlinear_range
export finish_points, find_parametric_curves
export ParametricCurve
export V₁, V, plot_u_ux_diagram
export Backend, CPU, GPU

include("types.jl")
include("physics.jl")
include("computation.jl")
include("analysis.jl")
include("plot.jl")

end
