module Expectiles

using Distributions
using LinearAlgebra
using Statistics
using ForwardDiff

import ..GARCHModels

export expectile,
       aux_expectile_var,
       residual_expectile,
       expectile_var,
       conditional_expectile,
       conditional_expectile_var

# ============================================================
# Core expectile routines
# ============================================================

"""
    expectile(X, τ; t0=nothing, tol=1e-10, maxiter=200)

Compute the empirical τ-expectile of `X` by solving the first-order condition
for asymmetric least squares.

Works for any τ in (0,1).
"""
function expectile(X::AbstractVector{<:Real}, τ::Real; t0=nothing, tol::Real=1e-10, maxiter::Int=200)
    @assert 0.0 < τ < 1.0 "τ must lie in (0,1)"
    @assert !isempty(X) "X must be nonempty"

    t = isnothing(t0) ? mean(X) : t0

    function psi_and_deriv(tval)
        ψ  = 0.0
        dψ = 0.0
        for x in X
            if x >= tval
                ψ  += τ * (x - tval)
                dψ -= τ
            else
                ψ  -= (1 - τ) * (tval - x)
                dψ -= (1 - τ)
            end
        end
        return ψ, dψ
    end

    # Newton
    for _ in 1:maxiter
        ψ, dψ = psi_and_deriv(t)
        if abs(ψ) < tol
            return t
        end
        if abs(dψ) < eps()
            break
        end
        tnew = t - ψ / dψ
        if !isfinite(tnew)
            break
        end
        t = tnew
    end

    # Safe fallback: bisection
    a = minimum(X) - 1.0
    b = maximum(X) + 1.0
    ψa, _ = psi_and_deriv(a)
    ψb, _ = psi_and_deriv(b)

    @assert ψa * ψb <= 0 "Failed to bracket the expectile root"

    for _ in 1:250
        c = (a + b) / 2
        ψc, _ = psi_and_deriv(c)
        if abs(ψc) < tol
            return c
        end
        if ψa * ψc <= 0
            b = c
            ψb = ψc
        else
            a = c
            ψa = ψc
        end
    end

    return (a + b) / 2
end

"""
    aux_expectile_var(ε, eτ, τ)

Return the two sample quantities used in the asymptotic variance of the
empirical expectile:

- `s2 = n^{-1} Σ ψ_i^2`
- `dψ = n^{-1} Σ |1(ε_i < eτ) - τ|`

where `ψ_i = (ε_i - eτ) * |1(ε_i < eτ) - τ|`.
"""
function aux_expectile_var(ε::AbstractVector{<:Real}, eτ::Real, τ::Real)
    n = length(ε)
    s2 = 0.0
    dψ = 0.0
    for i in 1:n
        w = abs((ε[i] < eτ) - τ)
        s2 += ((ε[i] - eτ)^2) * (w^2) / n
        dψ += w / n
    end
    return s2, dψ
end

# ============================================================
# Standardized residual helpers
# ============================================================

"""
    residual_expectile(y, σ2, τ)

Empirical expectile of standardized residuals `y ./ sqrt.(σ2)`.
"""
function residual_expectile(y::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, τ::Real)
    @assert length(y) == length(σ2) "y and σ2 must have the same length"
    @assert all(σ2 .> 0) "All conditional variances must be positive"
    ε = GARCHModels.residuals(y, σ2)
    return expectile(ε, τ)
end

"""
    residual_expectile(y, model, τ)

Empirical expectile of standardized residuals from the fitted GARCH model.
"""
function residual_expectile(y::AbstractVector{<:Real}, model, τ::Real)
    ε = GARCHModels.residuals(y, model)
    return expectile(ε, τ)
end




# ============================================================
# GARCH bridge for asymptotic variance
# ============================================================

function _garch_rootn_vcov(y::AbstractVector{<:Real}, model)
    V = GARCHModels.garch_parameter_variance(
            collect(Float64, y),
            model;
            rootn = true
        )

    return V
end

function _expectile_garch_components(y::AbstractVector{<:Real}, model, tau::Real)
    @assert 0.0 < tau < 1.0 "tau must lie in (0,1)"

    yv = collect(Float64, y)
    n = length(yv)
    # Use one shared equation-(3.19) path for the residuals, historical
    # derivatives, and one-step-ahead variance and derivative.
    variance, dh, h_next, dh_next =
        GARCHModels._risk_variance_path_gradient(yv, model)
    residuals = yv ./ sqrt.(variance)
    xi = expectile(residuals, tau)
    score_variance, score_derivative = aux_expectile_var(residuals, xi, tau)

    D = similar(dh)
    for t in 1:n
        @views D[t, :] .= 0.5 .* dh[t, :] ./ variance[t]
    end
    J = vec(sum(D, dims=1)) / n
    Jx = 0.5 .* dh_next ./ h_next

    # Match the effective sample used by the QMLE covariance calculation.
    m = max(length(model.α), length(model.β), 1)
    idx = (m + 1):n
    Deff = D[idx, :]
    residuals_eff = residuals[idx]
    g1 = zeros(Float64, length(idx))
    g2 = zeros(Float64, length(idx))
    GARCHModels._kernel_derivatives!(g1, g2, residuals_eff, model.dist)

    psi = similar(residuals_eff)
    for i in eachindex(residuals_eff)
        psi[i] = (residuals_eff[i] - xi) * abs((residuals_eff[i] < xi) - tau)
    end
    psi .-= mean(psi)

    information = Symmetric((Deff' * Deff) / length(idx))
    information_inv = pinv(information)
    Eg12 = mean(abs2, g1)
    Eg2 = mean(g2)
    Epsi_g1 = mean(g1 .* psi)

    # The estimator expansion uses -E[g2]^{-1} because g2 is the second
    # derivative of the log likelihood (negative in expectation).  Hence the
    # cross covariance is -E[psi*g1]/E[g2] * I^{-1}J. Multiplying J by the full
    # QMLE sandwich covariance would introduce an extra score-variance factor.
    Sigma_theta = Symmetric((Eg12 / Eg2^2) .* information_inv)
    Sigma_xp_theta = (-Epsi_g1 / Eg2) .* (information_inv * J)

    return (
        ξ = xi,
        s2 = score_variance,
        dψ = score_derivative,
        J = J,
        Jx = Jx,
        Σθ = Sigma_theta,
        ΣXPθ = Sigma_xp_theta,
        σ2 = variance,
        σ2_next = h_next,
        information = information,
        Eg2 = Eg2,
        Epsi_g1 = Epsi_g1
    )
end

# ============================================================
# Asymptotic variances
# ============================================================

"""
    expectile_var(y, σ2, τ)

Asymptotic variance of the empirical expectile based on known conditional scales.
This is the pure expectile term only.
"""
function expectile_var(y::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, τ::Real)
    @assert 0.0 < τ < 1.0 "τ must lie in (0,1)"
    @assert length(y) == length(σ2) "y and σ2 must have the same length"
    @assert all(σ2 .> 0) "All conditional variances must be positive"

    ε = GARCHModels.residuals(y, σ2)
    ξ = expectile(ε, τ)
    s2, dψ = aux_expectile_var(ε, ξ, τ)

    return s2 / dψ^2
end

"""
    expectile_var(y, model, τ; verbose=false)

Plug-in asymptotic variance of the two-step expectile estimator, including
the first-step QML estimation effect.
"""
function expectile_var(y::AbstractVector{<:Real}, model, τ::Real; verbose::Bool=false)
    comp = _expectile_garch_components(y, model, τ)

    pure_term  = comp.s2 / comp.dψ^2
    garch_term = comp.ξ^2 * dot(comp.J, comp.Σθ * comp.J)
    cross_term = 2 * (1 / comp.dψ) * comp.ξ * dot(comp.J, comp.ΣXPθ)

    Vξ = pure_term + garch_term - cross_term

    if verbose
        println("pure_term  = ", pure_term)
        println("garch_term = ", garch_term)
        println("cross_term = ", cross_term)
        println("Vξ         = ", Vξ)
    end

    return Vξ
end

# ============================================================
# Conditional expectile and predictive variance
# ============================================================

"""
    conditional_expectile(y, model, τ)

One-step-ahead conditional expectile forecast:
    σ_{n+1} * ξ̂_τ
"""
function conditional_expectile(y::AbstractVector{<:Real}, model, τ::Real)
    ξ = residual_expectile(y, model, τ)
    σ2_next = GARCHModels.forecast_variance(collect(Float64, y), model)
    return sqrt(σ2_next) * ξ
end

"""
    conditional_expectile_var(y, model, τ; verbose=false)

Plug-in asymptotic variance for the one-step-ahead conditional expectile forecast.

This implements the conditional version using Δ = J_x - J.
"""
function conditional_expectile_var(y::AbstractVector{<:Real}, model, τ::Real; verbose::Bool=false)
    comp = _expectile_garch_components(y, model, τ)

    Δ = comp.Jx - comp.J

    pure_term  = comp.s2 / comp.dψ^2
    garch_term = comp.ξ^2 * dot(Δ, comp.Σθ * Δ)
    cross_term = 2 * (1 / comp.dψ) * comp.ξ * dot(Δ, comp.ΣXPθ)

    # Conditional influence function:
    # dPsi^{-1}*psi + xi*(Jx-J)'*theta_influence.
    # The covariance term therefore enters with a positive sign.
    Vcond = comp.σ2_next * (pure_term + garch_term + cross_term)

    if verbose
        println("σ2_next    = ", comp.σ2_next)
        println("pure_term  = ", pure_term)
        println("garch_term = ", garch_term)
        println("cross_term = ", cross_term)
        println("Vcond      = ", Vcond)
    end

    return Vcond
end

end # module
