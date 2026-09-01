include(joinpath(@__DIR__, "mc_runner.jl"))

reps = parse(Int, get(ENV, "MC_PILOT_REPS", "50"))
output_name = get(ENV, "MC_PILOT_OUTPUT", "mc_joint_common_levels_pilot.csv")
retry_failed = lowercase(get(ENV, "MC_RETRY_FAILED", "false")) in ("1", "true", "yes")

specs = build_pilot_grid(n_sim=reps)
run_joint_grid(
    specs=specs,
    output_name=output_name,
    retry_failed=retry_failed,
)
