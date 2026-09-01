module GaoSongRisk

using Distributions
using LinearAlgebra
using Statistics

using ..GARCHModels: GARCHModel, _risk_variance_path_gradient

export gao_song_fhs_intervals

"""Generalized inverse of the empirical CDF used by the FHS estimator."""
function _empirical_quantile(x::AbstractVector{<:Real}, p::Real)
    0 < p < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    xs = sort!(collect(Float64, x))
    isempty(xs) && throw(ArgumentError("the empirical distribution must be nonempty"))
    return xs[ceil(Int, length(xs) * p)]
end

"""Gaussian kernel density estimate with Scott's normal-reference bandwidth."""
function _kernel_density(x::AbstractVector{<:Real}, point::Real)
    n = length(x)
    n > 1 || return NaN
    s = std(x)
    (!isfinite(s) || s <= 0) && return NaN
    bandwidth = 1.06 * s * n^(-1 / 5)
    kernel = Normal()
    return mean(pdf(kernel, (point - value) / bandwidth) / bandwidth for value in x)
end

"""Compatibility alias for the shared risk-measure variance path."""
_variance_path_gradient(data::AbstractVector{<:Real}, model::GARCHModel) =
    _risk_variance_path_gradient(data, model)

function _safe_interval(point, rootn_variance, n, critical_value)
    tolerance = 1.0e-10 * max(abs(rootn_variance), 1.0)
    if !isfinite(rootn_variance) || rootn_variance < -tolerance
        return (se=NaN, lower=NaN, upper=NaN, length=NaN, valid=false)
    end
    se = sqrt(max(rootn_variance, 0.0) / n)
    return (
        se=se,
        lower=point - critical_value * se,
        upper=point + critical_value * se,
        length=2 * critical_value * se,
        valid=isfinite(se),
    )
end

"""
    gao_song_fhs_intervals(data, model, alpha; threshold=nothing, ci_level=0.95)

Compute feasible two-step filtered-historical-simulation confidence intervals for
one-step-ahead conditional VaR and ES. The VaR variance implements Gao and Song
(2008), Theorems 3.1--3.2; the ES variance implements Theorems 3.3--3.4.

`threshold=nothing` uses the generalized-inverse empirical `alpha` quantile of the
fitted standardized residuals. Passing a threshold supports the matched gain-loss
comparison, in which `alpha` is the empirical probability at or below that
threshold. The implementation uses the paper's uncentered-residual formulas and
Scott's normal-reference bandwidth for the Gaussian kernel density estimate.
Returned `rootn_variance` objects estimate the variance of the corresponding
root-n statistic; the reported standard errors divide these quantities by the
sample size.
"""
function gao_song_fhs_intervals(
    data::AbstractVector{<:Real},
    model::GARCHModel,
    alpha::Real;
    threshold::Union{Nothing,Real}=nothing,
    ci_level::Real=0.95,
)
    n = length(data)
    n > 1 || throw(ArgumentError("at least two observations are required"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    0 < ci_level < 1 || throw(ArgumentError("ci_level must lie strictly between zero and one"))

    h, dh, h_next, u_next = _variance_path_gradient(data, model)
    residuals = collect(Float64, data) ./ sqrt.(h)
    sigma_next = sqrt(h_next)
    q = threshold === nothing ? _empirical_quantile(residuals, alpha) : Float64(threshold)
    density = _kernel_density(residuals, q)

    # Feasible versions of e and M in Gao--Song Theorems 3.2 and 3.4.
    scaled_gradient = dh ./ h
    ehat = vec(mean(scaled_gradient; dims=1))
    mhat = Symmetric((scaled_gradient' * scaled_gradient) / n)
    minv = try
        inv(mhat)
    catch
        fill(NaN, size(mhat))
    end

    mu4 = mean(residuals .^ 4)
    at_or_below = residuals .<= q
    below = residuals .< q
    tail_count = count(below)
    tail_count > 0 || throw(ArgumentError("the selected threshold contains no tail observations"))
    mu = mean(residuals[below])

    quadratic_form_e = dot(ehat, minv * ehat)
    quadratic_form_u = dot(u_next, minv * u_next)
    cross_form = dot(u_next, minv * ehat)

    # D1 estimates n Var(sigma_hat_{n+1}) in both feasible theorems.
    d1 = (mu4 - 1) * quadratic_form_u / (4 * h_next)

    # VaR: Theorem 3.2.
    mu2_alpha = mean((residuals .^ 2 .- 1) .* at_or_below)
    d2_var = (mu4 - 1) * q^2 * quadratic_form_e / 4 +
             alpha * (1 - alpha) / density^2 +
             q * mu2_alpha * quadratic_form_e / density
    d12_var = -cross_form / (2 * sigma_next) *
              (q * (mu4 - 1) / 2 + mu2_alpha / density)
    rootn_var_variance = d1 * q^2 + d2_var * h_next +
                         2 * sigma_next * q * d12_var
    var_point = sigma_next * q

    # ES: Theorem 3.4. These formulas avoid density estimation.
    mu3_alpha = mean((residuals[below] .- q) .* (residuals[below] .^ 2 .- 1))
    capped = ifelse.(below, residuals, q)
    capped_second = ifelse.(below, residuals .^ 2, q^2)
    d2_es = ((mu4 - 1) * mu^2 / 4 - mu * mu3_alpha) * quadratic_form_e +
            n * sum(capped_second) / tail_count^2 -
            sum(capped)^2 / tail_count^2
    d12_es = -cross_form / (2 * sigma_next) *
             (mu * (mu4 - 1) / 2 - mu3_alpha)
    rootn_es_variance = d1 * mu^2 + d2_es * h_next +
                        2 * sigma_next * mu * d12_es
    es_point = sigma_next * mu

    critical_value = quantile(Normal(), 0.5 + ci_level / 2)
    var_interval = _safe_interval(var_point, rootn_var_variance, n, critical_value)
    es_interval = _safe_interval(es_point, rootn_es_variance, n, critical_value)

    return (
        alpha=Float64(alpha),
        threshold=q,
        density=density,
        residuals=residuals,
        sigma_next=sigma_next,
        var=(
            estimate=var_point,
            rootn_variance=rootn_var_variance,
            interval=var_interval,
            components=(D1=d1, D2=d2_var, D12=d12_var),
        ),
        es=(
            estimate=es_point,
            innovation_mean=mu,
            rootn_variance=rootn_es_variance,
            interval=es_interval,
            components=(D1=d1, D2=d2_es, D12=d12_es),
        ),
        diagnostics=(mu4=mu4, tail_count=tail_count, information=mhat),
    )
end

end
