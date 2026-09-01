using JLD2
using Printf

function design_model(d)
    return get(d, "model_type", get(d, "gjr", false) ? "GJR-GARCH" : "GARCH")
end

fmt(x::Bool) = x ? "1" : "0"
fmt(x::Integer) = string(x)
fmt(x::Real) = isfinite(x) ? @sprintf("%.12g", x) : string(x)
fmt(x) = string(x)

function export_raw_mc_csv(; input_name="mc_grid_results_48_recalculated.jld2")
    root = joinpath(@__DIR__, "..", "..", "results", "mc")
    summaries = load(joinpath(root, "raw", input_name))["summaries"]
    output = joinpath(root, "raw", "mc_raw_replications.csv")
    header = [
        "model", "distribution", "persistence", "n", "tau", "replication", "true_xi",
        "xi_hat", "v_xi", "xi_valid", "xi_cover", "xi_z",
        "ce_true", "ce_hat", "v_ce", "ce_valid", "ce_cover", "ce_z",
        "sigma_true_next", "sigma_hat_next", "fitted_persistence",
        "stationarity_distance", "coefficient_boundary_distance",
        "information_condition", "max_fitted_variance",
    ]

    open(output, "w") do io
        println(io, join(header, ','))
        for summary in summaries
            d = summary["design"]
            base = (design_model(d), d["dist"], d["persistence"], d["n"], d["tau"])
            for (replication, r) in enumerate(summary["raw"])
                row = (
                    base..., replication, d["true_xi"],
                    r.xi_hat, r.v_xi, r.xi_valid, r.xi_cover, r.xi_z,
                    r.ce_true, r.ce_hat, r.v_ce, r.ce_valid, r.ce_cover, r.ce_z,
                    r.sigma_true_next, r.sigma_hat_next, r.fitted_persistence,
                    r.stationarity_distance, r.coefficient_boundary_distance,
                    r.information_condition, r.max_fitted_variance,
                )
                println(io, join(fmt.(row), ','))
            end
        end
    end
    println("Replication-level Monte Carlo CSV written to $(abspath(output)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_raw_mc_csv(input_name=isempty(ARGS) ? "mc_grid_results_48_recalculated.jld2" : ARGS[1])
end
