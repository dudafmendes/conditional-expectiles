using Base.Threads

function run_design(spec::MCSpec)
    println("Running design: label=$(spec.label), n=$(spec.n), dist=$(dist_label(spec)), persistence=$(persistence_label(spec)), τ=$(spec.τ), reps=$(spec.n_sim)")

    true_xi = true_innovation_expectile(spec)
    results = Vector{Any}(undef, spec.n_sim)

    @threads for rep in 1:spec.n_sim
        results[rep] = mc_replication(spec, true_xi, rep)
    end

    summary = Dict{String,Any}()

    summary["design"] = Dict(
        "label" => spec.label,
        "n" => spec.n,
        "n_sim" => spec.n_sim,
        "tau" => spec.τ,
        "risk_alpha" => spec.risk_alpha,
        "dist" => dist_label(spec),
        "persistence" => persistence_label(spec),
        "omega" => spec.model.ω,
        "alpha" => copy(spec.model.α),
        "beta" => copy(spec.model.β),
        "gamma" => spec.model.γ,
        "gjr" => has_gjr(spec),
        "model_type" => has_gjr(spec) ? "GJR-GARCH" : "GARCH",
        "true_xi" => true_xi
    )

    summary["point"] = summarize_point_estimators(results, spec, true_xi)
    summary["intervals_xi"] = summarize_expectile_ci(results, spec)
    summary["intervals_ce"] = summarize_predicted_expectile_ci(results, spec)

    # keep raw only in the saved object
    summary["raw"] = results

    return summary
end
