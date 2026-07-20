
"""
    garch_negloglik(params, data, p, q, gjr=false, dist=Normal())

Compute the negative log-likelihood for a GARCH(p,q) or GJR-GARCH model.

# Arguments
- `params::Vector{Real}`: A vector of model parameters. The first element is ω (constant term).
  If `p > 0`, the next `p` elements are α coefficients for ARCH terms.
  If `q > 0`, the following `q` elements are β coefficients for GARCH terms.
  If `gjr=true`, the remaining element is the γ coefficient for the GJR asymmetry term.
- `data::Vector{Real}`: The time series data (returns or residuals).
- `p::Int`: The order of the ARCH component (number of α parameters).
- `q::Int`: The order of the GARCH component (number of β parameters).
- `gjr::Bool=false`: If `true`, includes GJR asymmetry term with γ parameter.
- `dist::Distribution=Normal()`: The error distribution, either `Normal()` or `TDist(ν)` for Student's t-distribution.

# Returns
- `Float64`: The negative log-likelihood value of the model given the data and parameters.

# Description
This function computes the conditional variances σ²_t recursively using the GARCH specification:
- σ²_t = ω + ∑_{i=1}^p α_i ε_{t-i}^2 + ∑_{j=1}^q β_j σ²_{t-j} + γ I(ε_{t-1} < 0) ε_{t-1}^2 (if `gjr=true`)
where ε_t are the data points, and I is the indicator function.

The negative log-likelihood is then calculated based on the specified distribution:
- For `Normal()`, uses the standard Gaussian log-likelihood.
- For `TDist(ν)`, uses the Student's t-distribution log-likelihood.

The variance is floored at 1e-8 to prevent numerical issues.

# Example

data = randn(100)  # Simulated data
params = [0.1, 0.2, 0.7]  # ω, α1, β1
nll = garch_negloglik(params, data, 1, 1)
println("Negative log-likelihood: ", nll)
"""
function garch_negloglik(params, data::AbstractVector{<:Real}, p::Int, q::Int, gjr::Bool=false, dist::Distribution=Normal())
    T = promote_type(eltype(params), eltype(data))
    ω = params[1]
    α = p > 0 ? params[2:1+p] : nothing
    β = q > 0 ? params[2+p:1+p+q] : nothing
    γ = gjr ? params[2+p+q] : nothing
    n = length(data)
    σ2 = zeros(T, n)

    m = max(p, 1)
    σ2[1:m] .= var(data)  # Initialize the first m variances with the sample variance
    for t in (m+1):n
        arch_sum = α === nothing ? zero(T) : sum(α[i] * (t-i > 0 ? T(data[t-i]^2) : zero(T)) for i in 1:p)
        garch_sum = β === nothing ? zero(T) : sum(β[j] * (t-j > 0 ? σ2[t-j] : zero(T)) for j in 1:q)
        gjr_sum = gjr && γ !== nothing ? γ * (t-1 > 0 && data[t-1] < 0 ? T(data[t-1]^2) : zero(T)) : zero(T)
        σ2[t] = ω + arch_sum + garch_sum + gjr_sum
        σ2[t] = max(σ2[t], T(1e-8))
    end

    if dist isa Normal
        return 0.5 * sum(log.(σ2[m+1:end]) .+ (T.(data[m+1:end]).^2) ./ σ2[m+1:end])
    elseif dist isa TDist
        ν = dist.ν
        # Negative log-likelihood for Student-t innovations with scale sqrt(σ2)
        return ((ν+1)/2) * sum(log.(1 .+ (T.(data[m+1:end]).^2) ./ (ν .* σ2[m+1:end]))) + 0.5 * sum(log.(σ2[m+1:end]))
    else
        error("Unsupported distribution")
    end
end


"""
Evaluate the negative log-likelihood using precomputed lagged values.
The initial values come from the design matrix X, which should be constructed to include the necessary lags for the (G)ARCH terms.
This function is more efficient than `garch_negloglik` since it avoids redundant lag computations
# Arguments
- `σ2::Vector{T}`: Preallocated vector to store conditional variances.
- `params::Vector{T}`: Model parameters in the same format as `garch_negloglik`.
- `data::Vector{T}`: Time series data.
- `X::Matrix{T}`: Precomputed lagged values matrix (from `garch_design_matrix`).
- `gjr::Bool=false`: If true, includes GJR asymmetry term.
- `dist::Distribution=Normal()`: Error distribution.

# Returns
- `T`: The negative log-likelihood value.
"""
function garch_likelihood_fixed!(σ2, params, data, X, gjr::Bool=false, dist::Distribution=Normal(); hot_start::Bool=false)
    # Use promote_type to ensure compatibility with ForwardDiff.Dual
    T = promote_type(eltype(params), eltype(data), eltype(X))
    garch_variance_fixed!(σ2, X, params, gjr; hot_start=hot_start)

    data_eff = data
    σ2_eff = σ2
    if hot_start
        m = max(size(X, 2), 1) # number of initial values needed for recursion
        data_eff = data[m+1:end]
        σ2_eff = σ2[m+1:end]
    end

    if dist isa Normal
        return T(0.5) * sum(log.(σ2_eff) .+ (T.(data_eff).^2) ./ σ2_eff)
    elseif dist isa TDist
        ν = T(dist.ν)
        # The following line is not compatible with ForwardDiff if dist.ν is not a number.
        # To fix: ensure dist.ν is a real number or a Dual number.
        return ((ν+1)/T(2)) * sum(log.(1 .+ (T.(data_eff).^2) ./ (ν .* σ2_eff))) + T(0.5) * sum(log.(σ2_eff))
    else
        error("Unsupported distribution")
    end
end
