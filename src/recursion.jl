const VARIANCE_FLOOR = 1.0e-8

"""
    _risk_variance_path_gradient(data, model)

Compute the common finite-history conditional-variance path used by the risk
measure estimators and its analytical parameter gradients. For symmetric GARCH,
this is Gao and Song's (2008) equation (3.19): unavailable returns are truncated,
while unavailable conditional variances equal
`omega / (1 - sum(beta))`. The same initialization is applied to the GJR
extension, with unavailable leverage terms set to zero.

The parameter order is `[omega, alpha..., beta..., gamma]`. Returns are
`(h, dh, h_next, dh_next)`.
"""
function _risk_variance_path_gradient(
    data::AbstractVector{<:Real}, model::GARCHModel,
)
    y = collect(Float64, data)
    n = length(y)
    p, q = length(model.α), length(model.β)
    gjr = model.γ !== nothing
    k = 1 + p + q + (gjr ? 1 : 0)
    h = zeros(n)
    dh = zeros(n, k)

    beta_sum = sum(model.β)
    intercept_denominator = 1.0 - beta_sum
    intercept_denominator > 0 || throw(ArgumentError(
        "the finite-history risk recursion requires sum(beta) < 1",
    ))
    h_initial = model.ω / intercept_denominator
    dh_initial = zeros(k)
    dh_initial[1] = 1.0 / intercept_denominator
    for j in 1:q
        dh_initial[1 + p + j] = model.ω / intercept_denominator^2
    end

    for t in 1:n
        value = model.ω
        gradient = zeros(k)
        gradient[1] = 1.0

        for i in 1:p
            if t - i > 0
                lagged_square = y[t - i]^2
                value += model.α[i] * lagged_square
                gradient[1 + i] += lagged_square
            end
        end
        for j in 1:q
            observed_lag = t - j > 0
            lagged_variance = observed_lag ? h[t - j] : h_initial
            lagged_gradient = observed_lag ? view(dh, t - j, :) : dh_initial
            value += model.β[j] * lagged_variance
            gradient[1 + p + j] += lagged_variance
            gradient .+= model.β[j] .* lagged_gradient
        end
        if gjr && t > 1 && y[t - 1] < 0
            lagged_square = y[t - 1]^2
            value += model.γ * lagged_square
            gradient[end] += lagged_square
        end

        if value <= VARIANCE_FLOOR
            h[t] = VARIANCE_FLOOR
            dh[t, :] .= 0.0
        else
            h[t] = value
            dh[t, :] .= gradient
        end
    end

    value_next = model.ω
    gradient_next = zeros(k)
    gradient_next[1] = 1.0
    for i in 1:p
        if n + 1 - i > 0
            lagged_square = y[n + 1 - i]^2
            value_next += model.α[i] * lagged_square
            gradient_next[1 + i] += lagged_square
        end
    end
    for j in 1:q
        observed_lag = n + 1 - j > 0
        lagged_variance = observed_lag ? h[n + 1 - j] : h_initial
        lagged_gradient = observed_lag ? view(dh, n + 1 - j, :) : dh_initial
        value_next += model.β[j] * lagged_variance
        gradient_next[1 + p + j] += lagged_variance
        gradient_next .+= model.β[j] .* lagged_gradient
    end
    if gjr && n > 0 && y[n] < 0
        lagged_square = y[n]^2
        value_next += model.γ * lagged_square
        gradient_next[end] += lagged_square
    end
    if value_next <= VARIANCE_FLOOR
        value_next = VARIANCE_FLOOR
        gradient_next .= 0.0
    end

    return h, dh, value_next, gradient_next
end

"""
    forecast_variance(data::Vector{Float64}, model::GARCHModel)

Forecast the conditional variance one step ahead using the common finite-history
risk-measure recursion based on Gao and Song's equation (3.19).

# Arguments
- `data::Vector{Float64}`: Observed time series data (e.g., returns or residuals).
- `model::GARCHModel`: Fitted GARCH model.

# Returns
- `Float64`: Forecasted conditional variance for time T+1.

# Example
σ2_next = forecast_variance(data, model)
"""
function forecast_variance(data::Vector{Float64}, model::GARCHModel)
    _, _, σ2_next, _ = _risk_variance_path_gradient(data, model)
    return σ2_next
end

"""
    residuals(data::Vector{Float64}, model::GARCHModel)

Return standardized residuals and conditional variances for a `GARCHModel`.
# Arguments
- `data::Vector{Float64}`: Observed time series data (e.g., returns or residuals).
- `model::GARCHModel`: Fitted GARCH model.
# Returns
- A tuple `(standardized_residuals, σ2)` where:
  - `standardized_residuals`: A vector of standardized residuals (data divided by conditional standard deviations).
  - `σ2`: A vector of conditional variances.
# Example
standardized_residuals, σ2 = residuals(data, model)
"""

function residuals(data::Vector{Float64}, model::GARCHModel)
    σ2 = garch_variance(data, model)
    return data ./ sqrt.(σ2)
end

residuals(y::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}) = y ./ sqrt.(σ2)

function residuals_and_variance(data::Vector{Float64}, model::GARCHModel)
    σ2 = garch_variance(data, model)
    return data ./ sqrt.(σ2), σ2
end

"""
Populate `σ2` with conditional variances using precomputed lagged values and parameter vector.
"""
function garch_variance_fixed!(σ2::AbstractVector, X::AbstractMatrix, params::AbstractVector, gjr::Bool = false; hot_start::Bool=false)
    # element type that will be used for arithmetic (supports Dual numbers from ForwardDiff)
    T = promote_type(eltype(σ2), eltype(X), eltype(params))
    n, p = size(X)
    # number of β terms inferred from params (remaining after ω and α₁:α_p)
    q = gjr ? max(0, length(params) - 1 - p - 1) : max(0, length(params) - 1 - p)

    m = 0
    if hot_start
        m = max(p, q, 1) # number of initial values needed for recursion
    end
    h_init = hot_start ? var(X[:,1]) : zero(T) # initial variance for recursion if hot start is enabled

    tiny = convert(T, 1e-8)
    zeroT = zero(T)
    ω = convert(T, params[1])

    @inbounds for t in 1:n
        # ARCH part: α_i * (X_i)^2, with precomputed lags in X
        arch_sum = zeroT
        if p > 0
            for i in 1:p
                ai = convert(T, params[1 + i])          # α_i
                xi = convert(T, X[t, i])               # residual_{t-i} (0 if out-of-sample)
                arch_sum += ai * xi^2
            end
        end

        # GARCH part: β_j * σ2_{t-j}
        garch_sum = zeroT
        if q > 0
            for j in 1:q
                bj = convert(T, params[1 + p + j])    # β_j
                # Convert σ2[t-j] to type T for compatibility
                prev_var = t - j > 0 ? convert(T, σ2[t - j]) : h_init
                garch_sum += bj * prev_var
            end
        end

        # GJR part: γ * I(X_1 < 0) * (X_1)^2
        gjr_sum = zeroT
        if gjr && length(params) == 2 + p + q
            γ  = convert(T, params[end])               # γ
            x1 = convert(T, X[t, 1])
            gjr_sum = γ * (x1 < 0 ? x1^2 : zeroT)
        end

        σ2t = ω + arch_sum + garch_sum + gjr_sum

        # ensure positivity, keep type T (works with Dual)
        # Convert back to the element type of σ2 for assignment
        σ2[t] = convert(eltype(σ2), ifelse(σ2t < tiny, tiny, σ2t))
    end

    return σ2
end

# Convenience allocation wrapper that works when params may contain Dual numbers.
function garch_variance_fixed(X::AbstractMatrix, params::AbstractVector, gjr::Bool = false; hot_start::Bool=false)
    T = promote_type(eltype(X), eltype(params))
    n = size(X,1)
    σ2 = zeros(T, n)
    garch_variance_fixed!(σ2, X, params, gjr; hot_start=hot_start)
    return σ2
end

"""
Populate `σ2` with conditional variances using lagged matrix `X` and a `GARCHModel`.
"""
function garch_variance_fixed!(σ2, X::AbstractMatrix, model::GARCHModel; hot_start::Bool=false)
    gjr = model.γ !== nothing
    garch_variance_fixed!(σ2, X, convert(Vector{Float64}, model), gjr; hot_start=hot_start)
end

"""
Return conditional variances for lagged matrix `X` given a `GARCHModel`.
"""
function garch_variance_fixed(X::AbstractMatrix, model::GARCHModel; hot_start::Bool=false)
    n = size(X,1)
    σ2 = zeros(Float64, n)
    garch_variance_fixed!(σ2, X, model; hot_start=hot_start)
    return σ2
end

"""
Return conditional variances given a `GARCHModel`. By default, the path uses the
common finite-history risk-measure initialization based on Gao and Song's equation
(3.19). Set `hot_start=true` to retain the legacy sample-variance initialization.
"""
function garch_variance(data::AbstractVector, model::GARCHModel; hot_start::Bool=false)
    if !hot_start
        σ2, _, _, _ = _risk_variance_path_gradient(data, model)
        return σ2
    end
    p = model.α === nothing ? 0 : length(model.α)
    X = garch_design_matrix(data, p)
    σ2 =  garch_variance_fixed(X, model; hot_start=hot_start)
    return σ2
end
