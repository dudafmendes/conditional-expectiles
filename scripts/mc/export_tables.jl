using Printf
using JLD2

function ensure_result_dirs()
    raw_dir = joinpath(@__DIR__, "..", "..", "results", "mc", "raw")
    tables_dir = joinpath(@__DIR__, "..", "..", "results", "mc", "tables")
    mkpath(raw_dir)
    mkpath(tables_dir)
    return raw_dir, tables_dir
end

function _fmt(x; upper_limit = 1e3, lower_limit = 1e-4)
    if x isa Number
        # 1. Handle edge cases like Inf or NaN so the string split doesn't fail
        if !isfinite(x)
            return isnan(x) ? "NaN" : (x > 0 ? "\$\\infty\$" : "\$-\\infty\$")
        end

        # 2. Use standard formatting for zero and numbers within your "safe" range
        if x == 0 || (abs(x) >= lower_limit && abs(x) < upper_limit)
            return @sprintf("%.4f", x)
        else
            # 3. Handle large/small numbers with scientific notation
            sci_str = @sprintf("%.4e", x)
            mantissa, exp_str = split(sci_str, "e")

            # Parse the exponent as an integer to strip formatting like "+04" down to "4"
            exponent = parse(Int, exp_str)

            # Interpolate into a LaTeX-friendly math mode string
            # Note: \$ escapes the dollar sign, \\ escapes the backslash for \times
            return "\$$(mantissa) \\times 10^{$(exponent)}\$"
        end
    else
        return string(x)
    end
end
_row(fields) = join(string.(fields), " & ") * " \\\\"

function format_dist(d_str::String)
    if startswith(d_str, "Normal")
        return "Normal"
    elseif startswith(d_str, "TDist")
        # Extract the degrees of freedom (e.g., from "TDist{Float64}(df=8.0)")
        m = match(r"=([0-9\.]+)", d_str)
        if m !== nothing
            df = round(Int, parse(Float64, m.captures[1]))
            return "\$t_{$(df)}\$" # Renders as t_8 or t_4 in LaTeX
        else
            return "\$t\$"
        end
    else
        return d_str
    end
end

function _design_key(s)
    d = s["design"]
    return (d["dist"], d["persistence"], d["n"])
end

function _sorted_summaries(summaries)
    return sort(summaries; by = _design_key)
end

function point_table_tex(summaries)
    lines = String[]
    push!(lines, "\\begin{tabular}{lllrrrrrrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & Bias(\$\\hat\\omega\$) & RMSE(\$\\hat\\omega\$) & Bias(\$\\hat\\alpha_1\$) & RMSE(\$\\hat\\alpha_1\$) & Bias(\$\\hat\\beta_1\$) & RMSE(\$\\hat\\beta_1\$) & Bias(\$\\hat\\xi_\\tau\$) & RMSE(\$\\hat\\xi_\\tau\$) \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        p = s["point"]
        push!(lines, _row([
            format_dist(d["dist"]),
            d["persistence"],
            d["n"],
            _fmt(p["omega_bias"]),
            _fmt(p["omega_rmse"]),
            _fmt(p["alpha1_bias"]),
            _fmt(p["alpha1_rmse"]),
            _fmt(p["beta1_bias"]),
            _fmt(p["beta1_rmse"]),
            _fmt(p["xi_bias"]),
            _fmt(p["xi_rmse"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function expectile_ci_table_tex(summaries)
    lines = String[]
    push!(lines, "\\begin{tabular}{lllrrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & Cov.(\$\\xi_\\tau\$) & Avg. SE(\$\\xi_\\tau\$) & MC SD(\$\\hat\\xi_\\tau\$) & Valid Var. Frac. \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        iv = s["intervals_xi"]
        push!(lines, _row([
            format_dist(d["dist"]),
            d["persistence"],
            d["n"],
            _fmt(iv["xi_coverage_95"]),
            _fmt(iv["xi_avg_asymptotic_se"]),
            _fmt(iv["xi_mc_sd"]),
            _fmt(iv["xi_valid_vfrac"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function predicted_expectile_table_tex(summaries)
    lines = String[]
    push!(lines, "\\begin{tabular}{lllrrrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & Bias(CE) & RMSE(CE) & Cov.(CE) & Avg. SE(CE) & Valid Var. Frac. \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        p = s["point"]
        iv = s["intervals_ce"]
        push!(lines, _row([
            format_dist(d["dist"]) ,
            d["persistence"],
            d["n"],
            _fmt(p["ce_bias"]),
            _fmt(p["ce_rmse"]),
            _fmt(iv["ce_coverage_95"]),
            _fmt(iv["ce_avg_asymptotic_se"]),
            _fmt(iv["ce_valid_vfrac"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function export_tables(; input_name = "mc_grid_results.jld2")
    raw_dir, tables_dir = ensure_result_dirs()

    input_path = joinpath(raw_dir, input_name)
    data = load(input_path)
    summaries = data["summaries"]

    write(joinpath(tables_dir, "mc_point_estimators.tex"), point_table_tex(summaries))
    write(joinpath(tables_dir, "mc_expectile_ci.tex"), expectile_ci_table_tex(summaries))
    write(joinpath(tables_dir, "mc_predicted_expectile.tex"), predicted_expectile_table_tex(summaries))

    println("LaTeX tables written to: $(abspath(tables_dir))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_tables()
end
