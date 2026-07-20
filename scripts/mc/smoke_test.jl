include(joinpath(@__DIR__, "mc_runner.jl"))

specs = build_full_grid(n_obs=[500], n_sim=2, truth_mc=1_000)
@assert length(specs) == 12
for spec in specs
    persistence = sum(spec.model.α) + sum(spec.model.β) +
                  (spec.model.γ === nothing ? 0.0 : spec.model.γ / 2)
    target = persistence_label(spec) == "low" ? 0.90 : 0.98
    @assert isapprox(persistence, target; atol=1e-12)
end

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
