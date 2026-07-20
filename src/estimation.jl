"""
extract parameters
"""

function _unpack_params(params, p, q, gjr, dist)
    ω = params[1]
    α = p > 0 ? collect(params[2:1+p]) : Float64[]
    β = q > 0 ? collect(params[2+p:1+p+q]) : Float64[]
    γ = gjr ? params[2+p+q] : nothing
    return GARCHModel(ω, α, β, γ, dist)
end

function _persistence(params, p, q, gjr)
    alpha_sum = p > 0 ? sum(params[2:1+p]) : zero(eltype(params))
    beta_sum = q > 0 ? sum(params[2+p:1+p+q]) : zero(eltype(params))
    gamma_term = gjr ? params[2+p+q] / 2 : zero(eltype(params))
    return alpha_sum + beta_sum + gamma_term
end

function _stationarity_penalty(params, p, q, gjr; cap=0.999)
    excess = max(_persistence(params, p, q, gjr) - cap, zero(eltype(params)))
    return 1.0e10 * excess^2
end

"""
    estimate_fixed(data::Vector{Float64}, X::Matrix{Float64}, p::Int, q::Int; gjr::Bool=false, dist::Distribution=Normal())

Estimate GARCH parameters using precomputed design matrices to minimize garch_likelihood_fixed.

# Arguments
- `data::Vector{Float64}`: Time series data
- `X::Matrix{Float64}`: Precomputed lagged values matrix (from garch_design_matrix)
- `p::Int`: ARCH order
- `q::Int`: GARCH order
- `gjr::Bool=false`: If true, estimate GJR-GARCH model with asymmetric term
- `dist::Distribution=Normal()`: Error distribution

# Returns
- `GARCHModel`: Estimated model parameters

# Notes
- Uses same initialization and bounds as estimate()
- More efficient than estimate() since X matrix is precomputed
"""
function estimate_fixed(data::Vector{Float64}, X::Matrix{Float64}, p::Int, q::Int; gjr::Bool=false, dist::Distribution=Normal(), hot_start::Bool=false)
    # Initial parameter values
    ω0 = var(data) * 0.1
    α0 = p > 0 ? fill(0.1/p, p) : Float64[]
    β0 = q > 0 ? fill(0.8/q, q) : Float64[]
    γ0 = gjr ? 0.05 : Float64[]
    x0 = vcat([ω0], α0, β0, γ0)


    # hot_start = true means we only evaluate the likelihood on the data points where σ2 is fully determined by the model (i.e., after the initial m points)
    data_eff = data
    X_eff = X
    if hot_start
        m = max(p, 1)
        data_eff = data[m+1:end]
        X_eff = X[m+1:end, :]
    end

    # Parameter bounds
    lower = vcat([1e-6], fill(0.0, p+q+(gjr ? 1 : 0)))
    upper = vcat([10.0], fill(1.0, p+q+(gjr ? 1 : 0)))

    # Optimization function that allocates σ2 with proper type for BackwardDiff
    function optf(params)
        T = promote_type(eltype(params), eltype(data_eff), eltype(X_eff))
        σ2_temp = zeros(T, length(data_eff))  # Allocate with proper type
        value = garch_likelihood_fixed!(σ2_temp, params, data_eff, X_eff, gjr, dist; hot_start=hot_start)
        return value + length(data_eff) * _stationarity_penalty(params, p, q, gjr)
    end

    # Optimize using Fminbox with BFGS
    sol = optimize(optf, lower, upper, x0, Fminbox(BFGS()); autodiff = AutoForwardDiff()) # testar BHHH

    if !Optim.converged(sol)
        error("GARCH parameter estimation failed to converge.")
    end

    # Extract parameters
    params = Optim.minimizer(sol)
    _persistence(params, p, q, gjr) <= 0.99901 || error("GARCH estimate violates the stationarity constraint.")
    return _unpack_params(params, p, q, gjr, dist)
end

function estimate_fixed(data::Vector{Float64}, p::Int, q::Int; gjr::Bool=false, dist::Distribution=Normal(), hot_start::Bool=false)
    # obtain fixed design matrices
    X = garch_design_matrix(data, p)
    return estimate_fixed(data, X, p, q; gjr=gjr, dist=dist, hot_start=hot_start)
end



"""
    estimate(data::Vector{Float64}, p::Int, q::Int; gjr::Bool=false, dist::Distribution=Normal(), hot_start::Bool=false)

Estimate the parameters of a GARCH(p, q) model using Maximum Likelihood Estimation (MLE).

This function fits a GARCH model to the provided time series data by optimizing the negative log-likelihood function. It supports standard GARCH and GJR-GARCH variants, with customizable error distributions.

# Arguments
- `data::Vector{Float64}`: The time series data (e.g., returns or residuals) to model.
- `p::Int`: The order of the ARCH component (number of lagged squared residuals).
- `q::Int`: The order of the GARCH component (number of lagged variances).
- `gjr::Bool=false`: If `true`, fits a GJR-GARCH model which includes an asymmetric term for negative shocks.
- `dist::Distribution=Normal()`: The distribution of the error terms (e.g., `Normal()`, `StudentizedNormal()`, etc.). Defaults to standard normal.
- `hot_start::Bool=false`: If `true`, uses a hot start initialization for the optimization.

# Returns
- `GARCHModel`: A struct containing the estimated parameters:
  - `ω`: The constant term in the variance equation.
  - `α`: Vector of ARCH coefficients (length `p`).
  - `β`: Vector of GARCH coefficients (length `q`).
  - `γ`: The asymmetric coefficient for GJR-GARCH (if `gjr=true`).
  - `dist`: The error distribution used.

# Notes
- Initial parameter values are set heuristically: `ω=0.1`, `α=0.1` for each lag, `β=0.8/q` for each lag, and `γ=0.0` for GJR.
- Bounds are applied during optimization: `ω ∈ [1e-6, 10.0]`, coefficients ∈ [0.0, 1.0].
- Uses the `Optim.jl` package with `Fminbox(BFGS())` for constrained optimization.
- Assumes the data is stationary and suitable for GARCH modeling.

# Examples
# Create a GARCH(1,1) model with normal distribution
model = GARCHModel(0.1, [0.2], [0.7], nothing, Normal())
# Simulate 1000 observations
data, variances = simulate_garch(model, 1000)
# Estimate GARCH(1,1) parameters from the simulated data
estimated_model = estimate_garch(data, 1, 1)
println("Estimated ω: ", estimated_model.ω)
println("Estimated α: ", estimated_model.α)
println("Estimated β: ", estimated_model.β)
"""
function estimate(data::Vector{Float64}, p::Int, q::Int; gjr::Bool=false, dist::Distribution=Normal())
    ω0 = 0.1
    α0 = p > 0 ? fill(0.1/p, p) : Float64[]
    β0 = q > 0 ? fill(0.8/q, q) : Float64[]
    γ0 = gjr ? 0.05 : Float64[]
    x0 = vcat([ω0], α0, β0, γ0)
    lower = vcat([1e-6], fill(0.0, p+q+(gjr ? 1 : 0)))
    upper = vcat([10.0], fill(1.0, p+q+(gjr ? 1 : 0)))

    function optf(x)
        return garch_negloglik(x, data, p, q, gjr, dist) +
               length(data) * _stationarity_penalty(x, p, q, gjr)
    end
    sol = optimize(optf, lower, upper, x0, Fminbox(BFGS()); autodiff = AutoForwardDiff())
    if !Optim.converged(sol)
        error("GARCH parameter estimation failed to converge.")
    end
    params = Optim.minimizer(sol)
    _persistence(params, p, q, gjr) <= 0.99901 || error("GARCH estimate violates the stationarity constraint.")
    return _unpack_params(params, p, q, gjr, dist)
end
