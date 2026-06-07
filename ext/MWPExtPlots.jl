module MWPExtPlots

using MultiWellPotentialGrossPitaevskii
import MultiWellPotentialGrossPitaevskii: ParametricCurve
using Plots, CSV, RecipesBase, LaTeXStrings
using DataFrames: DataFrame

RecipesBase.@recipe f(c::ParametricCurve) = (c.x, c.y)

function validate_path(save_path)
    if !(save_path isa AbstractString)
        throw(ArgumentError("save_path must be a string"))
    end
    if isempty(save_path)
        throw(ArgumentError("save_path must not be empty"))
    end
    save_path = normpath(save_path)
    if ispath(save_path)
        return realpath(save_path)
    end
    parent = dirname(save_path)
    if ispath(parent)
        return joinpath(realpath(parent), basename(save_path))
    end
    if !isabspath(save_path)
        save_path = joinpath(pwd(), save_path)
    end
    return save_path
end

function MultiWellPotentialGrossPitaevskii.plot_u_ux_diagram(data::DataFrame; save_path = nothing, linewidth = 0.5, title = nothing)
    curve₋ = ParametricCurve(data.C, data.um, data.uxm)
    curve₊ = ParametricCurve(data.C, data.up, data.uxp)

    plot = Plots.plot(title = title, xlabel = L"u(0)", ylabel = L"u'(0)")
    plot = Plots.plot!(curve₋; label = L"γ_-", linewidth = linewidth)
    plot = Plots.plot!(curve₊; label = L"γ_+", linewidth = linewidth)

    if save_path != nothing
        save_path = validate_path(save_path)

        if title != nothing
            if !(title isa AbstractString)
                throw(ArgumentError("title must be a string"))
            end
            if isempty(title)
                throw(ArgumentError("title must not be empty"))
            end
            if occursin(r"[/\\]", title)
                throw(ArgumentError("title must not contain path separators"))
            end
        end

        if !isdir(save_path)
            mkdir(save_path)
        end
        CSV.write(joinpath(save_path, "$(title)-diagram-data.csv"), data)
        Plots.savefig(plot, joinpath(save_path, "$(title).svg"))
    end

    return plot
end

end
