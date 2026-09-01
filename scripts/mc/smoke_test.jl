include(joinpath(@__DIR__, "mc_runner.jl"))

specs = build_full_grid(n_obs=[500], n_sim=2, truth_mc=1_000)
@assert length(specs) == 12
for spec in specs
    @assert spec.risk_levels == [0.01, 0.05]
    persistence = sum(spec.model.α) + sum(spec.model.β) +
                  (spec.model.γ === nothing ? 0.0 : spec.model.γ / 2)
    target = persistence_label(spec) == "low" ? 0.90 : 0.98
    @assert isapprox(persistence, target; atol=1e-12)
end

header = joint_csv_header(first(specs))
for token in ("0p01", "0p05")
    @assert "xi_tau_$(token)_estimate" in header
    @assert "ce_tau_$(token)_ci_length" in header
    @assert "var_alpha_$(token)_ci_length" in header
    @assert "es_delta_$(token)_ci_length" in header
    @assert "ratio_ce_tau_$(token)_to_var_alpha_$(token)" in header
    @assert "ratio_ce_tau_$(token)_to_es_delta_$(token)" in header
end
@assert !any(occursin("tau_prime"), header)

population = _joint_population_targets(first(specs))
@assert [target.level for target in population.targets] == [0.01, 0.05]

rng = MersenneTwister(1)
@assert isapprox(var(rand(rng, standardized_t(8), 200_000)), 1.0; atol=0.03)

summary = run_design(first(specs))
@assert haskey(summary["intervals_xi"], "xi_se_sd_ratio")
@assert haskey(summary["intervals_ce"], "ce_se_sd_ratio")

include(joinpath(@__DIR__, "export_paper_results.jl"))
rows = summary_rows([summary])
@assert length(rows) == 1
@assert length(rows[1]) == length(SUMMARY_HEADER)
println("Monte Carlo smoke test passed.")
