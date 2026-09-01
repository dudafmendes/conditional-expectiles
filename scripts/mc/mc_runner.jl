using Dates
using JLD2
using Base.Threads

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "point_estimators.jl"))
include(joinpath(@__DIR__, "expectile_ci.jl"))
include(joinpath(@__DIR__, "predicted_expectile_ci.jl"))
include(joinpath(@__DIR__, "run_parallel.jl"))
include(joinpath(@__DIR__, "joint_risk_mc.jl"))

function ensure_result_dirs()
    raw_dir = joinpath(@__DIR__, "..", "..", "results", "mc", "raw")
    tables_dir = joinpath(@__DIR__, "..", "..", "results", "mc", "tables")
    mkpath(raw_dir)
    mkpath(tables_dir)
    return raw_dir, tables_dir
end


function build_gjr_grid(;
    n_obs = [500, 1000, 2500, 5000],
    n_sim = 10_000,
    risk_levels = [0.01, 0.05],
    α1 = 0.05,
    γ = 0.08,      # Standard leverage effect
    β_low = 0.81,  # Low persistence:  0.05 + 0.81 + (0.08/2) = 0.90
    β_high = 0.89, # High persistence: 0.05 + 0.89 + (0.08/2) = 0.98
    annual_vol = 20.0,
    seed = 24681,
    truth_mc = 200_000,
)
    # Adjust omega for GJR-GARCH unconditional variance: ω = σ² * (1 - α - β - γ/2)
    vol_daily_sq = (annual_vol^2) / 252

    ω_low = vol_daily_sq * (1 - α1 - β_low - γ/2)
    ω_high = vol_daily_sq * (1 - α1 - β_high - γ/2)

    # Dictionary/Array of distributions to iterate over
    dists = [
        ("normal", Normal()),
        ("t8", standardized_t(8)),
        ("t4", standardized_t(4))
    ]

    specs = MCSpec[]

    for n in n_obs
        for (dist_name, dist) in dists

                # 1. Low Persistence Design
                push!(specs, MCSpec(
                    model = GARCHModel(ω_low, [α1], [β_low], γ, Normal()),
                    n = n, n_sim = n_sim, τ = first(risk_levels),
                    seed = seed, truth_mc = truth_mc,
                    label = "gjr-low-$(dist_name)",
                    innovation_dist = dist,
                    risk_levels = collect(Float64, risk_levels),
                ))

                # 2. High Persistence Design
                push!(specs, MCSpec(
                    model = GARCHModel(ω_high, [α1], [β_high], γ, Normal()),
                    n = n, n_sim = n_sim, τ = first(risk_levels),
                    seed = seed, truth_mc = truth_mc,
                    label = "gjr-high-$(dist_name)",
                    innovation_dist = dist,
                    risk_levels = collect(Float64, risk_levels),
                ))

        end
    end

    return specs
end

function build_grid(;
    n_obs = [500, 1000, 2500, 5000],
    n_sim = 10_000,
    risk_levels = [0.01, 0.05],
    α1 = 0.05,
    β_low = 0.85,
    β_high = 0.93,
    annual_vol = 20.0,
    seed = 24681,
    truth_mc = 200_000,
)
    ω_low = annual_vol^2 * (1 - α1 - β_low) / 252
    ω_high = annual_vol^2 * (1 - α1 - β_high) / 252

    specs = MCSpec[]

    for n in n_obs
        push!(specs, MCSpec(
            model = GARCHModel(ω_low, [α1], [β_low], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "normal-low",
            innovation_dist = Normal(),
            risk_levels = collect(Float64, risk_levels),
        ))

        push!(specs, MCSpec(
            model = GARCHModel(ω_low, [α1], [β_low], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "t8-low",
            innovation_dist = standardized_t(8),
            risk_levels = collect(Float64, risk_levels),
        ))

        push!(specs, MCSpec(
            model = GARCHModel(ω_low, [α1], [β_low], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "t4-low",
            innovation_dist = standardized_t(4),
            risk_levels = collect(Float64, risk_levels),
        ))

        push!(specs, MCSpec(
            model = GARCHModel(ω_high, [α1], [β_high], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "normal-high",
            innovation_dist = Normal(),
            risk_levels = collect(Float64, risk_levels),
            ))

        push!(specs, MCSpec(
            model = GARCHModel(ω_high, [α1], [β_high], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "t8-high",
            innovation_dist = standardized_t(8),
            risk_levels = collect(Float64, risk_levels),
        ))

        push!(specs, MCSpec(
            model = GARCHModel(ω_high, [α1], [β_high], nothing, Normal()),
            n = n, n_sim = n_sim, τ = first(risk_levels),
            seed = seed, truth_mc = truth_mc,
            label = "t4-high",
            innovation_dist = standardized_t(4),
            risk_levels = collect(Float64, risk_levels),
        ))
    end

    return specs
end

function run_grid(; save_name = "mc_grid_results.jld2", specs = build_grid(), kwargs...)
    raw_dir, _ = ensure_result_dirs()

    summaries = Vector{Any}(undef, length(specs))

    for (i, spec) in enumerate(specs)
        summaries[i] = run_design(spec)
    end

    metadata = Dict(
        "timestamp" => string(now()),
        "threads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "n_designs" => length(specs),
    )

    save_path = joinpath(raw_dir, save_name)
    @save save_path summaries metadata

    println("Saved Monte Carlo results to: $(abspath(save_path))")
    return summaries
end

"""
    build_full_grid(; kwargs...)

Construct the paper's 48 structural designs: 24 symmetric GARCH designs and 24
correctly specified GJR-GARCH designs. Every fitted replication evaluates the
common 1% and 5% levels for XP, VaR, and ES.
"""
function build_full_grid(; kwargs...)
    return vcat(build_grid(; kwargs...), build_gjr_grid(; kwargs...))
end

"""Construct the targeted 16-design pre-publication pilot grid."""
function build_pilot_grid(; n_obs=[500, 5000], n_sim=2_000, truth_mc=200_000)
    specs = build_full_grid(n_obs=n_obs, n_sim=n_sim, truth_mc=truth_mc)
    return filter(s -> dist_label(s) in ("normal", "t4"), specs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    retry_failed = lowercase(get(ENV, "MC_RETRY_FAILED", "false")) in ("1", "true", "yes")
    run_joint_grid(
        specs=build_full_grid(),
        output_name="mc_joint_common_levels_replications.csv",
        retry_failed=retry_failed,
    )
end
