using Random
using Statistics
using LinearAlgebra
using Printf
using Distributions

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles
using ConditionalExpectiles.GaoSongRisk

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
- `risk_alpha`: Optional lower-tail probability. When provided, each replication
  also computes Gao--Song two-step VaR and ES intervals.
- `risk_levels`: Common lower-tail levels used for XP, VaR, and ES. The joint
  experiment enforces `alpha = delta = tau` at every value in this vector.
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
    risk_alpha::Union{Nothing,Float64} = nothing
    risk_levels::Vector{Float64} = [0.01, 0.05]

    label::String = ""
end

function _upper_first_moment(d::Distribution, point::Real)
    if d isa Normal
        z = (point - d.μ) / d.σ
        return d.σ * pdf(Normal(), z) + d.μ * (1 - cdf(Normal(), z))
    elseif d isa LocationScale && d.ρ isa TDist
        nu = d.ρ.ν
        z = (point - d.μ) / d.σ
        upper_z_first = ((nu + z^2) / (nu - 1)) * pdf(d.ρ, z)
        return d.μ * (1 - cdf(d.ρ, z)) + d.σ * upper_z_first
    end
    error("No deterministic truncated-first-moment formula for $(typeof(d)).")
end

"""Population expectile for the supported innovation distributions."""
function innovation_expectile(d::Distribution, tau::Real)
    0 < tau < 1 || throw(ArgumentError("tau must lie strictly between zero and one"))

    function balance(e)
        upper_first = _upper_first_moment(d, e)
        right = upper_first - e * (1 - cdf(d, e))
        left = e * cdf(d, e) - (mean(d) - upper_first)
        return tau * right - (1 - tau) * left
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

"""Population expectile level whose expectile equals the `alpha` quantile."""
function population_tau_for_quantile(d::Distribution, alpha::Real)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    q = quantile(d, alpha)
    upper_first = _upper_first_moment(d, q)
    below = q * cdf(d, q) - (mean(d) - upper_first)
    above = upper_first - q * (1 - cdf(d, q))
    return below / (below + above)
end

"""Population lower-tail conditional mean at probability `delta`."""
function population_lower_es(d::Distribution, delta::Real)
    0 < delta < 1 || throw(ArgumentError("delta must lie strictly between zero and one"))
    q = quantile(d, delta)
    lower_first = mean(d) - _upper_first_moment(d, q)
    return lower_first / cdf(d, q)
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
    return innovation_expectile(spec.innovation_dist, spec.τ)
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

    # Optional VaR/ES experiment. No tail level is inferred from the expectile
    # level: callers must explicitly choose risk_alpha for the desired design.
    gao_song_risk = spec.risk_alpha === nothing ? nothing :
        gao_song_fhs_intervals(y, fit, spec.risk_alpha; ci_level=0.95)

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
        ce_z = ce_z,
        gao_song_risk = gao_song_risk
    )
end
