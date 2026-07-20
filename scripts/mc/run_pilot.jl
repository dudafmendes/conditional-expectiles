include(joinpath(@__DIR__, "mc_runner.jl"))

reps = parse(Int, get(ENV, "MC_PILOT_REPS", "2000"))
specs = build_pilot_grid(n_sim=reps)
run_grid(save_name="mc_pilot_results_$(reps).jld2", specs=specs)
