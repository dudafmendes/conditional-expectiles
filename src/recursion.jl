"""
    forecast_variance(data::Vector{Float64}, model::GARCHModel)

Forecast the conditional variance one step ahead given observed data and a fitted GARCHModel.

# Arguments
- `data::Vector{Float64}`: Observed time series data (e.g., returns or residuals).
- `model::GARCHModel`: Fitted GARCH model.

# Returns
- `Float64`: Forecasted conditional variance for time T+1.

# Example
σ2_next = forecast_variance(data, model)
"""
function forecast_variance(data::Vector{Float64}, model::GARCHModel)
    n = length(data)
    p = model.α === nothing ? 0 : length(model.α)
    q = model.β === nothing ? 0 : length(model.β)
    γ = model.γ
    # Compute last p residuals and last q variances
    arch_sum = p == 0 ? 0.0 : sum(model.α[i] * (n-i+1 > 0 ? data[n-i+1]^2 : 0.0) for i in 1:p)
    # Compute conditional variances up to n
    σ2 = garch_variance(data, model)
    garch_sum = q == 0 ? 0.0 : sum(model.β[j] * (n-j+1 > 0 ? σ2[n-j+1] : 0.0) for j in 1:q)
    gjr_sum = γ === nothing ? 0.0 : γ * (n > 0 && data[n] < 0 ? data[n]^2 : 0.0)
    σ2_next = model.ω + arch_sum + garch_sum + gjr_sum
    return max(σ2_next, 1e-8)
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
Return conditional variances given a `GARCHModel`.
"""
function garch_variance(data::AbstractVector, model::GARCHModel; hot_start::Bool=false)
    p = model.α === nothing ? 0 : length(model.α)
    X = garch_design_matrix(data, p)
    σ2 =  garch_variance_fixed(X, model; hot_start=hot_start)
    return σ2
end
