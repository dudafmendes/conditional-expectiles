using JLD2
using Distributions

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles

# Include the source files containing the summary functions
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "point_estimators.jl"))
include(joinpath(@__DIR__, "expectile_ci.jl"))
include(joinpath(@__DIR__, "predicted_expectile_ci.jl"))

"""
    parse_dist(d_str::String)

Helper to reconstruct the Julia Distribution object from the string
saved in the results dictionary.
"""
function parse_dist(d_str::String)
    d_lower = lowercase(d_str)
    if d_lower == "normal" || startswith(d_str, "Normal")
        return Normal()
    elseif d_lower == "t8"
        return standardized_t(8)
    elseif d_lower == "t4"
        return standardized_t(4)
    elseif startswith(d_str, "TDist")
        # Extract the degrees of freedom using regex
        m = match(r"=([0-9\.]+)", d_str)
        df = m !== nothing ? parse(Float64, m.captures[1]) : 8.0
        return TDist(df)
    end
    return Normal()
end

"""
    reconstruct_spec(d::Dict)

Recreates the MCSpec object required by the summary functions from the saved design dictionary.
"""
function reconstruct_spec(d::Dict)
    dist = parse_dist(d["dist"])

    # Reconstruct the true Data-Generating Process model
    model = GARCHModel(d["omega"], d["alpha"], d["beta"], d["gamma"], Normal())

    return MCSpec(
        model = model,
        n = d["n"],
        n_sim = d["n_sim"],
        τ = d["tau"],
        label = d["label"],
        innovation_dist = dist
    )
end

"""
    recalculate_statistics()

Loads the raw simulations, reconstructs the MC design, recalculates the summaries,
and saves the output to a new .jld2 file.
"""
function recalculate_statistics(; input_name="mc_grid_results.jld2", output_name="mc_grid_results_recalculated.jld2")
    # Resolve the data paths
    raw_dir = joinpath(@__DIR__, "..", "..", "results", "mc", "raw")
    input_path = joinpath(raw_dir, input_name)
    output_path = joinpath(raw_dir, output_name)

    println("Loading original MC data from: $(abspath(input_path))")
    data = load(input_path)

    summaries = data["summaries"]

    for (i, summary) in enumerate(summaries)
        d = summary["design"]
        raw_results = summary["raw"]

        println("Recalculating summary $i/$(length(summaries)) - Label: $(d["label"]), n: $(d["n"])")

        # 1. Reconstruct parameters required by summary functions
        spec = reconstruct_spec(d)
        true_xi = d["true_xi"]

        # 2. Call the respective functions on the raw results
        summary["point"] = summarize_point_estimators(raw_results, spec, true_xi)
        summary["intervals_xi"] = summarize_expectile_ci(raw_results, spec)
        summary["intervals_ce"] = summarize_predicted_expectile_ci(raw_results, spec)
    end

    # Save the re-computed object with the original metadata
    println("Saving recalculated statistics to: $(abspath(output_path))")
    metadata = data["metadata"]

    # JLD2 @save macro usage
    @save output_path summaries metadata
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    input_name = length(ARGS) >= 1 ? ARGS[1] : "mc_grid_results.jld2"
    output_name = length(ARGS) >= 2 ? ARGS[2] : "mc_grid_results_recalculated.jld2"
    recalculate_statistics(input_name=input_name, output_name=output_name)
end
