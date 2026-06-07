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
