using JLD2
using Printf

function csv_escape(x)
    s = string(x)
    escaped = replace(s, '"' => "\"\"")
    return occursin(',', s) || occursin('"', s) ? "\"$(escaped)\"" : s
end

fmt(x) = x isa Number ? (isfinite(x) ? @sprintf("%.6f", x) : string(x)) : string(x)

function design_model(d)
    return get(d, "model_type", get(d, "gjr", false) ? "GJR-GARCH" : "GARCH")
end

function summary_rows(summaries)
    rows = Vector{Vector{Any}}()
    for s in summaries
        d, p, xi, ce = s["design"], s["point"], s["intervals_xi"], s["intervals_ce"]
        base = Any[design_model(d), d["dist"], d["persistence"], d["n"], d["tau"]]
        push!(rows, vcat(base, Any[
            p["xi_bias"], p["xi_rmse"], xi["xi_mc_sd"], xi["xi_avg_asymptotic_se"],
            xi["xi_se_sd_ratio"], xi["xi_coverage_95"], xi["xi_se_coverage"],
            xi["xi_lower_miss_95"], xi["xi_upper_miss_95"], xi["xi_avg_length_95"],
            xi["xi_valid_vfrac"], xi["xi_z_mean"], xi["xi_z_sd"], xi["xi_z_sd_mcse"],
            p["ce_bias"], p["ce_rmse"], ce["ce_mc_sd"], ce["ce_avg_asymptotic_se"],
            ce["ce_se_sd_ratio"], ce["ce_coverage_95"], ce["ce_se_coverage"],
            ce["ce_lower_miss_95"], ce["ce_upper_miss_95"], ce["ce_avg_length_95"],
            ce["ce_valid_vfrac"], ce["ce_z_mean"], ce["ce_z_sd"], ce["ce_z_sd_mcse"],
            p["sigma_error_sd"], p["xi_sigma_error_corr"],
            p["persistence_hat_mean"], p["persistence_hat_p99"], p["near_stationarity_fraction"],
            p["information_condition_median"], p["information_condition_p99"],
            p["max_variance_p99"], p["abs_xi_error_p99"], p["abs_ce_error_p99"]
        ]))
    end
    return sort(rows; by=r -> (r[1], r[2], r[3], r[4]))
end

const SUMMARY_HEADER = [
    "model", "distribution", "persistence", "n", "tau",
    "xi_bias", "xi_rmse", "xi_mc_sd", "xi_avg_se", "xi_se_sd_ratio",
    "xi_coverage", "xi_coverage_mcse", "xi_lower_miss", "xi_upper_miss",
    "xi_avg_length", "xi_valid_fraction", "xi_z_mean", "xi_z_sd", "xi_z_sd_mcse",
    "ce_bias", "ce_rmse", "ce_mc_sd", "ce_avg_se", "ce_se_sd_ratio",
    "ce_coverage", "ce_coverage_mcse", "ce_lower_miss", "ce_upper_miss",
    "ce_avg_length", "ce_valid_fraction", "ce_z_mean", "ce_z_sd", "ce_z_sd_mcse",
    "sigma_error_sd", "xi_sigma_error_correlation", "persistence_hat_mean",
    "persistence_hat_p99", "near_stationarity_fraction", "information_condition_median",
    "information_condition_p99", "max_variance_p99", "abs_xi_error_p99", "abs_ce_error_p99"
]

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csv_escape.(fmt.(row)), ','))
        end
    end
end

function write_inference_table(path, rows)
    open(path, "w") do io
        println(io, "\\begin{tabular}{lllrrrrrr}")
        println(io, "\\toprule")
        println(io, "Model & Dist. & Pers. & \$n\$ & MC SD & Avg. SE & SE/SD & Coverage & Length \\\\")
        println(io, "\\midrule")
        for r in rows
            println(io, join([r[1], r[2], r[3], string(r[4]), fmt(r[22]), fmt(r[23]),
                              fmt(r[24]), fmt(r[25]), fmt(r[29])], " & ") * " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function selected_studentized_rows(summaries)
    rows = Vector{Vector{Any}}()
    for s in summaries
        d = s["design"]
        selected = d["persistence"] == "high" && d["n"] in (500, 5000) && d["dist"] in ("normal", "t4")
        selected || continue
        for (rep, r) in enumerate(s["raw"])
            push!(rows, Any[design_model(d), d["dist"], d["persistence"], d["n"], rep, r.xi_z, r.ce_z])
        end
    end
    return rows
end

function export_paper_results(; input_name="mc_grid_results_48.jld2")
    root = joinpath(@__DIR__, "..", "..", "results", "mc")
    data = load(joinpath(root, "raw", input_name))
    summaries = data["summaries"]
    tables = joinpath(root, "tables")
    figures = joinpath(root, "figures")
    mkpath(tables); mkpath(figures)
    prefix = occursin("pilot", lowercase(input_name)) ? "mc_pilot" : "mc"

    rows = summary_rows(summaries)
    write_csv(joinpath(tables, "$(prefix)_inference_summary.csv"), SUMMARY_HEADER, rows)
    write_inference_table(joinpath(tables, "$(prefix)_conditional_expectile_inference.tex"), rows)
    write_csv(joinpath(figures, "$(prefix)_studentized_selected.csv"),
              ["model", "distribution", "persistence", "n", "rep", "z_xi", "z_ce"],
              selected_studentized_rows(summaries))
    println("Paper tables and plot data written under $(abspath(root)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_paper_results(input_name=isempty(ARGS) ? "mc_grid_results_48.jld2" : ARGS[1])
end
