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
        if !isfinite(x)
            return isnan(x) ? "NaN" : (x > 0 ? "\$\\infty\$" : "\$-\\infty\$")
        end

        if x == 0 || (abs(x) >= lower_limit && abs(x) < upper_limit)
            return @sprintf("%.4f", x)
        else
            sci_str = @sprintf("%.4e", x)
            mantissa, exp_str = split(sci_str, "e")
            exponent = parse(Int, exp_str)
            return "\$$(mantissa) \\times 10^{$(exponent)}\$"
        end
    else
        return string(x)
    end
end

# Helper to stack the estimate/bias and standard error in one cell
function _fmt_cell(val, se)
    val_str = _fmt(val)
    se_str = _fmt(se)
    return "\\makecell{$(val_str) \\\\ \\scriptsize ($(se_str))}"
end

_row(fields) = join(string.(fields), " & ") * " \\\\"

function format_dist(d_str::String)
    if startswith(d_str, "Normal")
        return "Normal"
    elseif startswith(d_str, "TDist")
        m = match(r"=([0-9\.]+)", d_str)
        if m !== nothing
            df = round(Int, parse(Float64, m.captures[1]))
            return "\$t_{$(df)}\$"
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
    # Reduced number of columns because Bias and RMSE are now in the same cell
    push!(lines, "\\begin{tabular}{lllrrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & \$\\hat\\omega\$ & \$\\hat\\alpha_1\$ & \$\\hat\\beta_1\$ & \$\\hat\\xi_\\tau\$ \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        p = s["point"]
        push!(lines, _row([
            format_dist(d["dist"]),
            d["persistence"],
            d["n"],
            _fmt_cell(p["omega_bias"], p["omega_rmse"]),
            _fmt_cell(p["alpha1_bias"], p["alpha1_rmse"]),
            _fmt_cell(p["beta1_bias"], p["beta1_rmse"]),
            _fmt_cell(p["xi_bias"], p["xi_rmse"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function expectile_ci_table_tex(summaries)
    lines = String[]
    push!(lines, "\\begin{tabular}{lllrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & Cov.(\$\\xi_\\tau\$) & \\makecell{MC SD(\$\\hat\\xi_\\tau\$) \\\\ \\scriptsize (Avg. SE)} & Valid Var. Frac. \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        iv = s["intervals_xi"]
        push!(lines, _row([
            format_dist(d["dist"]),
            d["persistence"],
            d["n"],
            _fmt_cell(iv["xi_coverage_95"], iv["xi_se_coverage"]), # Stacking coverage and its SE
            _fmt_cell(iv["xi_mc_sd"], iv["xi_avg_asymptotic_se"]), # Stacking true SD and estimated SE
            _fmt(iv["xi_valid_vfrac"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function predicted_expectile_table_tex(summaries)
    lines = String[]
    push!(lines, "\\begin{tabular}{lllrrr}")
    push!(lines, "\\hline")
    push!(lines, "Dist & Pers. & \$n\$ & CE Bias (RMSE) & Cov.(CE) & Valid Var. Frac. \\\\")
    push!(lines, "\\hline")

    for s in _sorted_summaries(summaries)
        d = s["design"]
        p = s["point"]
        iv = s["intervals_ce"]

        push!(lines, _row([
            format_dist(d["dist"]),
            d["persistence"],
            d["n"],
            _fmt_cell(p["ce_bias"], p["ce_rmse"]),
            _fmt_cell(iv["ce_coverage_95"], iv["ce_se_coverage"]),
            _fmt(iv["ce_valid_vfrac"]),
        ]))
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function export_tables(; input_name = "mc_grid_results_recalculated.jld2")
    raw_dir, tables_dir = ensure_result_dirs()

    input_path = joinpath(raw_dir, input_name)
    data = load(input_path)
    summaries = data["summaries"]

    # I've appended .
    write(joinpath(tables_dir, "mc_point_estimators.tex"), point_table_tex(summaries))
    write(joinpath(tables_dir, "mc_expectile_ci.tex"), expectile_ci_table_tex(summaries))
    write(joinpath(tables_dir, "mc_predicted_expectile.tex"), predicted_expectile_table_tex(summaries))

    println("LaTeX tables written to: $(abspath(tables_dir))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_tables(input_name = "mc_grid_results.jld2")
end
