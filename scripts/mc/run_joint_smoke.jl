include(joinpath(@__DIR__, "mc_runner.jl"))

reps = parse(Int, get(ENV, "MC_SMOKE_REPS", "2"))
output_name = get(ENV, "MC_SMOKE_OUTPUT", "mc_joint_common_levels_smoke.csv")

specs = build_full_grid(n_obs=[500], n_sim=reps)
run_joint_grid(specs=specs, output_name=output_name)
