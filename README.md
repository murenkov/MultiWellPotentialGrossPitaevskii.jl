# MultiWellPotentialGrossPitaevskii

[![Build Status](https://github.com/murenkov/MultiWellPotentialGrossPitaevskii.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/murenkov/MultiWellPotentialGrossPitaevskii.jl/actions/workflows/CI.yml?query=branch%3Amain)

Analysis and simulation tools for the Gross-Pitaevskii equation with multi-well potentials.

## Exported API

### Physics
- `MultiWellParams{T,N}` — parameter struct holding frequency `ω`, well amplitudes `as`, and well positions `ds`
- `multiwell_potential_equation(u, p, t)` — ODE right-hand side
- `V₁(t, A)`, `V(t, as, ds)` — potential functions
- `singular(x)`, `regular(x)` — solution validity predicates

### Computation
- `finish_points(Cs, ps, tspan)` — GPU-batched ODE ensemble solver
- `find_parametric_curves(Cs, ps)` — compute parametric `(u, u′)` curves

### Analysis
- `every_nth(iter, n)` — sample every n-th element from an iterator
- `define_directions(x, y)` — classify curve direction at each point
- `monotonicity_intervals(xs)` — split a sequence into monotonic segments
- `find_intersections(data; interpolation)` — find curve intersections using polynomial or linear interpolation
- `find_polynomials_intersections(p₁, p₂, x_range)` — roots of polynomial difference
- `find_interpolations_intersections(i₁, i₂, x_range)` — roots of interpolation difference
- `nonlinear_range(start, stop; length)` — generate a range with nonlinear spacing
- `fmt(x)` — round to 2 decimal places

### Plotting
- `plot_u_ux_diagram(data; save_path, linewidth, title)` — plot the `(u(0), u′(0))` phase diagram

## Testing

Run the test suite with:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Tests cover all exported functions. GPU-dependent tests (`finish_points`, `find_parametric_curves`) are skipped automatically when CUDA is unavailable.
