using Random
using Statistics
using LinearAlgebra
using Printf
using Distributions

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles

const Z975 = quantile(Normal(), 0.975)

"""
    MCSpec

Monte Carlo specification container used by the MC validation scripts.

# Fields
- `model::GARCHModel`: Data-generating GARCH model.
- `n::Int`: Sample size per replication.
- `n_sim::Int`: Number of Monte Carlo replications.
- `τ::Float64`: Target expectile level.
- `seed::Int`: Base random seed.
- `truth_mc::Int`: Number of draws used to approximate population truth.
- `burnout::Int`: Burn-in observations discarded in simulation.
- `innovation_dist::Distribution`: Innovation distribution used in simulation.
- `label::String`: Optional scenario label used in reporting.
"""
Base.@kwdef struct MCSpec
    model::GARCHModel
    n::Int
    n_sim::Int
    τ::Float64

    seed::Int = 24681
    truth_mc::Int = 200_000
    burnout::Int = 500
    innovation_dist::Distribution = Normal()

    label::String = ""
end

"""
    dist_label(spec::MCSpec) -> String

Return a string label for the model innovation distribution.
"""
function dist_label(spec::MCSpec)
    d = spec.innovation_dist
    if d isa Normal
        return "normal"
    elseif d isa LocationScale && d.ρ isa TDist
        return "t$(round(Int, d.ρ.ν))"
    end
    return string(d)
end

"""Return a variance-one Student-t distribution with `df` degrees of freedom."""
standardized_t(df::Real) = LocationScale(0.0, sqrt((df - 2) / df), TDist(df))

"""
    persistence_label(spec::MCSpec; threshold::Float64 = 0.90)

Classify persistence as `"low"` when `sum(α) + sum(β) <= threshold`, otherwise `"high"`.
"""
function persistence_label(spec::MCSpec; threshold::Float64 = 0.90)
    αsum = sum(spec.model.α)
    βsum = sum(spec.model.β)
    return αsum + βsum <= threshold ? "low" : "high"
end

"""
    p_order(spec::MCSpec) -> Int

Return the ARCH order `p` implied by the model coefficients.
"""
p_order(spec::MCSpec) = length(spec.model.α)

"""
    q_order(spec::MCSpec) -> Int

Return the GARCH order `q` implied by the model coefficients.
"""
q_order(spec::MCSpec) = length(spec.model.β)

"""
    has_gjr(spec::MCSpec) -> Bool

Return `true` when the specification includes a non-null leverage parameter `γ`.
"""
has_gjr(spec::MCSpec) = spec.model.γ !== nothing

"""
    true_innovation_expectile(spec::MCSpec) -> Float64

Estimate the population innovation expectile under the specification's innovation
distribution using Monte Carlo integration.
"""
function true_innovation_expectile(spec::MCSpec)
    d = spec.innovation_dist
    tau = spec.τ

    function balance(e)
        if d isa Normal
            z = (e - d.μ) / d.σ
            upper_first = d.σ * pdf(Normal(), z) + d.μ * (1 - cdf(Normal(), z))
            right = upper_first - e * (1 - cdf(d, e))
            left = e * cdf(d, e) - (d.μ - upper_first)
            return tau * right - (1 - tau) * left
        elseif d isa LocationScale && d.ρ isa TDist
            nu = d.ρ.ν
            z = (e - d.μ) / d.σ
            base = d.ρ
            upper_z_first = ((nu + z^2) / (nu - 1)) * pdf(base, z)
            upper_first = d.μ * (1 - cdf(base, z)) + d.σ * upper_z_first
            right = upper_first - e * (1 - cdf(d, e))
            left = e * cdf(d, e) - (mean(d) - upper_first)
            return tau * right - (1 - tau) * left
        end
        error("No deterministic population-expectile formula for $(typeof(d)).")
    end

    lo, hi = quantile(d, 1e-8), quantile(d, 1 - 1e-8)
    for _ in 1:200
        mid = (lo + hi) / 2
        if balance(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"""
    mc_replication(spec::MCSpec, true_xi::Float64, rep::Int)

Run one Monte Carlo replication for the given specification.

# Returns
A named tuple containing fitted parameters, estimated expectiles, asymptotic
variance estimates, and coverage indicators used by:
1. point estimator validation,
2. innovation expectile interval coverage,
3. predicted conditional expectile interval coverage.
"""
function mc_replication(spec::MCSpec, true_xi::Float64, rep::Int)
    rng = MersenneTwister(spec.seed + rep)

    true_model = spec.model
    p = p_order(spec)
    q = q_order(spec)
    gjr = has_gjr(spec)

    y, _ = simulate(rng, true_model, spec.n; burnout=spec.burnout, innovation_dist=spec.innovation_dist)

    fit = GARCHModels.estimate(y, p, q; gjr=gjr, dist=true_model.dist)

    θ_hat = convert(Vector{Float64}, fit)
    fitted_persistence = sum(fit.α) + sum(fit.β) + (fit.γ === nothing ? 0.0 : fit.γ / 2)

    # residual expectile
    resid, fitted_variance = GARCHModels.residuals_and_variance(y, fit)
    xi_hat = expectile(resid, spec.τ)
    v_xi = expectile_var(y, fit, spec.τ)   # root-n variance object

    # predicted conditional expectile
    s2_true_next = forecast_variance(y, true_model)
    s2_hat_next = forecast_variance(y, fit)

    ce_true = sqrt(s2_true_next) * true_xi
    ce_hat = sqrt(s2_hat_next) * xi_hat
    v_ce = conditional_expectile_var(y, fit, spec.τ)  # root-n variance object

    dh, h, _, _ = GARCHModels._garch_variance_gradient(y, fit)
    m = max(p, q, 1)
    D = 0.5 .* dh[(m + 1):end, :] ./ h[(m + 1):end]
    information = Symmetric((D' * D) / size(D, 1))
    information_condition = cond(Matrix(information))
    coefficient_distance = minimum(vcat(θ_hat[1] - 1e-6, θ_hat[2:end], 1 .- θ_hat[2:end]))
    stationarity_distance = 0.999 - fitted_persistence

    # interval coverage
    xi_se = isfinite(v_xi) && v_xi ≥ 0 ? sqrt(v_xi / spec.n) : NaN
    xi_lo = xi_hat - Z975 * xi_se
    xi_hi = xi_hat + Z975 * xi_se
    xi_valid = isfinite(xi_se)
    xi_cover = xi_valid && (xi_lo <= true_xi <= xi_hi)
    xi_lower_miss = xi_valid && true_xi < xi_lo
    xi_upper_miss = xi_valid && true_xi > xi_hi
    xi_length = xi_valid ? xi_hi - xi_lo : NaN
    xi_z = xi_valid && xi_se > 0 ? (xi_hat - true_xi) / xi_se : NaN

    ce_se = isfinite(v_ce) && v_ce ≥ 0 ? sqrt(v_ce / spec.n) : NaN
    ce_lo = ce_hat - Z975 * ce_se
    ce_hi = ce_hat + Z975 * ce_se
    ce_valid = isfinite(ce_se)
    ce_cover = ce_valid && (ce_lo <= ce_true <= ce_hi)
    ce_lower_miss = ce_valid && ce_true < ce_lo
    ce_upper_miss = ce_valid && ce_true > ce_hi
    ce_length = ce_valid ? ce_hi - ce_lo : NaN
    ce_z = ce_valid && ce_se > 0 ? (ce_hat - ce_true) / ce_se : NaN

    return (
        θ_hat = θ_hat,
        ω_hat = fit.ω,
        α_hat = copy(fit.α),
        β_hat = copy(fit.β),
        γ_hat = fit.γ,
        fitted_persistence = fitted_persistence,
        stationarity_distance = stationarity_distance,
        coefficient_boundary_distance = coefficient_distance,
        information_condition = information_condition,
        max_fitted_variance = maximum(fitted_variance),
        xi_hat = xi_hat,
        v_xi = v_xi,
        xi_valid = xi_valid,
        xi_cover = xi_cover,
        xi_lower_miss = xi_lower_miss,
        xi_upper_miss = xi_upper_miss,
        xi_length = xi_length,
        xi_z = xi_z,
        sigma_true_next = sqrt(s2_true_next),
        sigma_hat_next = sqrt(s2_hat_next),
        ce_hat = ce_hat,
        ce_true = ce_true,
        v_ce = v_ce,
        ce_valid = ce_valid,
        ce_cover = ce_cover,
        ce_lower_miss = ce_lower_miss,
        ce_upper_miss = ce_upper_miss,
        ce_length = ce_length,
        ce_z = ce_z
    )
end
