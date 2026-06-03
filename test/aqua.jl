using Pkg
Pkg.instantiate()

using Aqua
using MultiWellPotentialGrossPitaevskii

Aqua.test_all(
    MultiWellPotentialGrossPitaevskii;
    ambiguities = true,
    unbound_args = true,
    undefined_exports = true,
    project_extras = true,
    stale_deps = (ignore = [:Aqua, :JET],),
)

println("Aqua.jl tests passed successfully.")
