using Base.Threads
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

const JOINT_MC_SCHEMA_VERSION = "joint-risk-common-levels-v2"
const JOINT_RECORD_SUFFIXES = [
    "truth", "estimate", "rootn_variance", "se", "ci_lower", "ci_upper",
    "ci_length", "wald", "coverage", "valid",
]

_level_token(x::Real) = replace(@sprintf("%.6g", x), "-" => "m", "." => "p")

function _joint_design_id(spec::MCSpec)
    model = has_gjr(spec) ? "gjr" : "garch"
    return join((model, dist_label(spec), persistence_label(spec), "n$(spec.n)"), "-")
end

function _expectile_definitions(spec::MCSpec)
    isempty(spec.risk_levels) && throw(ArgumentError("risk_levels must not be empty"))
    definitions = NamedTuple[]
    seen = Set{String}()
    for tau in spec.risk_levels
        0 < tau < 1 || throw(ArgumentError("every common risk level must lie in (0, 1)"))
        key = "tau_$(_level_token(tau))"
        key in seen && throw(ArgumentError("duplicate expectile level $tau"))
        push!(seen, key)
        push!(definitions, (
            key=key,
            tau=Float64(tau),
            truth=innovation_expectile(spec.innovation_dist, tau),
        ))
    end
    return definitions
end

function _joint_prefixes(spec::MCSpec)
    prefixes = String[]
    for level in spec.risk_levels
        level_token = _level_token(level)
        append!(prefixes, [
            "xi_tau_$level_token", "ce_tau_$level_token",
            "var_alpha_$level_token", "es_delta_$level_token",
        ])
    end
    return prefixes
end

function _ratio_columns(spec::MCSpec)
    columns = String[]
    for level in spec.risk_levels
        token = _level_token(level)
        append!(columns, [
            "ratio_ce_tau_$(token)_to_var_alpha_$(token)",
            "ratio_ce_tau_$(token)_to_es_delta_$(token)",
        ])
    end
    return columns
end

function joint_csv_header(spec::MCSpec)
    header = [
        "design_id", "replication", "status", "error_message", "schema_version",
        "seed", "model", "distribution", "persistence", "n", "burnin",
        "risk_levels",
    ]
    for level in spec.risk_levels
        token = _level_token(level)
        append!(header, ["q_alpha_$token", "innovation_es_delta_$token"])
    end
    append!(header, [
        "true_omega", "true_alpha", "true_beta", "true_gamma",
        "hat_omega", "hat_alpha", "hat_beta", "hat_gamma",
        "fitted_persistence", "stationarity_distance",
        "coefficient_boundary_distance", "information_condition",
        "max_fitted_variance", "sigma_true_next", "sigma_hat_next",
    ])
    append!(header, _ratio_columns(spec))
    push!(header, "runtime_seconds")
    for prefix in _joint_prefixes(spec), suffix in JOINT_RECORD_SUFFIXES
        push!(header, "$(prefix)_$(suffix)")
    end
    return header
end

function _wald_record(estimate::Real, truth::Real, rootn_variance::Real, n::Int)
    tolerance = 1.0e-10 * max(abs(rootn_variance), 1.0)
    valid = isfinite(estimate) && isfinite(truth) && isfinite(rootn_variance) &&
            rootn_variance >= -tolerance
    if !valid
        return (
            truth=Float64(truth), estimate=Float64(estimate),
            rootn_variance=Float64(rootn_variance), se=NaN,
            ci_lower=NaN, ci_upper=NaN, ci_length=NaN, wald=NaN,
            coverage=false, valid=false,
        )
    end
    se = sqrt(max(rootn_variance, 0.0) / n)
    lower = estimate - Z975 * se
    upper = estimate + Z975 * se
    wald = se > 0 ? (estimate - truth) / se : NaN
    return (
        truth=Float64(truth), estimate=Float64(estimate),
        rootn_variance=Float64(rootn_variance), se=se,
        ci_lower=lower, ci_upper=upper, ci_length=upper - lower, wald=wald,
        coverage=lower <= truth <= upper, valid=isfinite(se),
    )
end

function _add_record!(row::Dict{String,Any}, prefix::String, record)
    for suffix in JOINT_RECORD_SUFFIXES
        row["$(prefix)_$(suffix)"] = getproperty(record, Symbol(suffix))
    end
    return row
end

function _joint_population_targets(spec::MCSpec)
    distribution = spec.innovation_dist
    definitions = _expectile_definitions(spec)
    targets = [(
        level=definition.tau,
        token=_level_token(definition.tau),
        xi_truth=definition.truth,
        q_alpha=quantile(distribution, definition.tau),
        innovation_es_delta=population_lower_es(distribution, definition.tau),
    ) for definition in definitions]
    return (targets=targets,)
end

function _base_joint_row(spec::MCSpec, replication::Int, population)
    model = spec.model
    row = Dict{String,Any}(
        "design_id" => _joint_design_id(spec),
        "replication" => replication,
        "status" => "running",
        "error_message" => "",
        "schema_version" => JOINT_MC_SCHEMA_VERSION,
        "seed" => spec.seed + replication,
        "model" => (has_gjr(spec) ? "GJR-GARCH" : "GARCH"),
        "distribution" => dist_label(spec),
        "persistence" => persistence_label(spec),
        "n" => spec.n,
        "burnin" => spec.burnout,
        "risk_levels" => join(spec.risk_levels, ";"),
        "true_omega" => model.ω,
        "true_alpha" => join(model.α, ";"),
        "true_beta" => join(model.β, ";"),
        "true_gamma" => model.γ,
    )
    for target in population.targets
        row["q_alpha_$(target.token)"] = target.q_alpha
        row["innovation_es_delta_$(target.token)"] = target.innovation_es_delta
    end
    return row
end

function joint_mc_replication(spec::MCSpec, population, replication::Int)
    started = time()
    row = _base_joint_row(spec, replication, population)
    rng = MersenneTwister(spec.seed + replication)
    true_model = spec.model
    p, q, gjr = p_order(spec), q_order(spec), has_gjr(spec)

    y, _ = simulate(
        rng, true_model, spec.n;
        burnout=spec.burnout, innovation_dist=spec.innovation_dist,
    )
    fit = GARCHModels.estimate(y, p, q; gjr=gjr, dist=true_model.dist)
    theta_hat = convert(Vector{Float64}, fit)

    residuals, fitted_variance = GARCHModels.residuals_and_variance(y, fit)
    sigma_true_next = sqrt(forecast_variance(y, true_model))
    sigma_hat_next = sqrt(forecast_variance(y, fit))

    fitted_persistence = sum(fit.α) + sum(fit.β) +
                         (fit.γ === nothing ? 0.0 : fit.γ / 2)
    dh, h, _, _ = GARCHModels._garch_variance_gradient(y, fit)
    m = max(p, q, 1)
    D = 0.5 .* dh[(m + 1):end, :] ./ h[(m + 1):end]
    information = Symmetric((D' * D) / size(D, 1))

    row["hat_omega"] = fit.ω
    row["hat_alpha"] = join(fit.α, ";")
    row["hat_beta"] = join(fit.β, ";")
    row["hat_gamma"] = fit.γ
    row["fitted_persistence"] = fitted_persistence
    row["stationarity_distance"] = 0.999 - fitted_persistence
    row["coefficient_boundary_distance"] = minimum(vcat(
        theta_hat[1] - 1e-6, theta_hat[2:end], 1 .- theta_hat[2:end],
    ))
    row["information_condition"] = cond(Matrix(information))
    row["max_fitted_variance"] = maximum(fitted_variance)
    row["sigma_true_next"] = sigma_true_next
    row["sigma_hat_next"] = sigma_hat_next

    for target in population.targets
        xi_hat = expectile(residuals, target.level)
        xi_variance = expectile_var(y, fit, target.level)
        xi_record = _wald_record(xi_hat, target.xi_truth, xi_variance, spec.n)
        _add_record!(row, "xi_tau_$(target.token)", xi_record)

        ce_hat = sigma_hat_next * xi_hat
        ce_truth = sigma_true_next * target.xi_truth
        ce_variance = conditional_expectile_var(y, fit, target.level)
        ce_record = _wald_record(ce_hat, ce_truth, ce_variance, spec.n)
        _add_record!(row, "ce_tau_$(target.token)", ce_record)

        risk_result = gao_song_fhs_intervals(
            y, fit, target.level; ci_level=0.95,
        )
        var_record = _wald_record(
            risk_result.var.estimate,
            sigma_true_next * target.q_alpha,
            risk_result.var.rootn_variance,
            spec.n,
        )
        _add_record!(row, "var_alpha_$(target.token)", var_record)

        es_record = _wald_record(
            risk_result.es.estimate,
            sigma_true_next * target.innovation_es_delta,
            risk_result.es.rootn_variance,
            spec.n,
        )
        _add_record!(row, "es_delta_$(target.token)", es_record)

        row["ratio_ce_tau_$(target.token)_to_var_alpha_$(target.token)"] =
            ce_record.valid && var_record.valid && var_record.ci_length > 0 ?
            ce_record.ci_length / var_record.ci_length : NaN
        row["ratio_ce_tau_$(target.token)_to_es_delta_$(target.token)"] =
            ce_record.valid && es_record.valid && es_record.ci_length > 0 ?
            ce_record.ci_length / es_record.ci_length : NaN
    end
    row["runtime_seconds"] = time() - started
    row["status"] = "success"
    return row
end

_csv_format(::Missing) = ""
_csv_format(::Nothing) = ""
_csv_format(x::Bool) = x ? "1" : "0"
_csv_format(x::Integer) = string(x)
_csv_format(x::Real) = isfinite(x) ? @sprintf("%.17g", x) : string(x)
_csv_format(x) = string(x)

function _csv_escape(value)
    text = _csv_format(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        escaped = replace(text, "\"" => "\"\"")
        return "\"$(escaped)\""
    end
    return text
end

function _write_joint_row(io, header::Vector{String}, row::Dict{String,Any})
    println(io, join((_csv_escape(get(row, column, missing)) for column in header), ','))
    flush(io)
end

function _initialize_joint_csv(path::String, header::Vector{String})
    if !isfile(path) || filesize(path) == 0
        open(path, "w") do io
            println(io, join(header, ','))
        end
        return
    end
    existing_header = open(readline, path)
    existing_header == join(header, ',') || throw(ArgumentError(
        "CSV schema mismatch at $path. Choose a new output_name for this experiment.",
    ))
end

function _completed_joint_replications(path::String; retry_failed::Bool=false)
    completed = Set{Tuple{String,Int}}()
    isfile(path) || return completed
    for (line_number, line) in enumerate(eachline(path))
        line_number == 1 && continue
        fields = split(line, ','; limit=4)
        length(fields) >= 3 || continue
        replication = tryparse(Int, fields[2])
        replication === nothing && continue
        status = fields[3]
        if status == "success" || (status == "failed" && !retry_failed)
            push!(completed, (fields[1], replication))
        end
    end
    return completed
end

function _failure_joint_row(spec::MCSpec, population, replication::Int, err)
    row = _base_joint_row(spec, replication, population)
    row["status"] = "failed"
    row["error_message"] = replace(sprint(showerror, err), r"\s+" => " ")
    return row
end

"""
    run_joint_grid(; specs=build_full_grid(), output_name="mc_joint_common_levels_replications.csv",
                     retry_failed=false)

Run or resume the joint expectile/VaR/ES Monte Carlo experiment. Each replication
is estimated once and immediately written as one complete CSV row. Existing rows
are skipped on restart. Set `retry_failed=true` to rerun rows previously recorded
with `status=failed`.
"""
function run_joint_grid(;
    specs::Vector{MCSpec}=build_full_grid(),
    output_name::String="mc_joint_common_levels_replications.csv",
    retry_failed::Bool=false,
)
    isempty(specs) && throw(ArgumentError("the Monte Carlo grid is empty"))
    reference_header = joint_csv_header(first(specs))
    for spec in specs
        joint_csv_header(spec) == reference_header || throw(ArgumentError(
            "all designs written to one CSV must use the same inferential targets",
        ))
    end

    raw_dir, _ = ensure_result_dirs()
    output_path = joinpath(raw_dir, output_name)
    _initialize_joint_csv(output_path, reference_header)
    completed = _completed_joint_replications(output_path; retry_failed=retry_failed)
    write_lock = ReentrantLock()

    open(output_path, "a") do io
        for (design_index, spec) in enumerate(specs)
            design_id = _joint_design_id(spec)
            population = _joint_population_targets(spec)
            pending = [rep for rep in 1:spec.n_sim if !((design_id, rep) in completed)]
            println(
                "Design $(design_index)/$(length(specs)): $design_id; ",
                "pending=$(length(pending)), completed=$(spec.n_sim - length(pending))",
            )

            @threads for index in eachindex(pending)
                replication = pending[index]
                row = try
                    joint_mc_replication(spec, population, replication)
                catch err
                    _failure_joint_row(spec, population, replication, err)
                end
                lock(write_lock) do
                    _write_joint_row(io, reference_header, row)
                end
            end
        end
    end

    println("Replication-level Monte Carlo CSV written to $(abspath(output_path)).")
    return output_path
end
