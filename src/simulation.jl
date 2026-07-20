"""
    simulate_garch(model::GARCHModel, n::Int; burnout::Int=500)

Simulate `n` observations from a GARCH model, including optional burnout period to ensure stationarity.

# Arguments
- `model::GARCHModel`: The GARCH model specification containing parameters like ω, α, β, γ, and distribution.
- `n::Int`: The number of observations to simulate.
- `burnout::Int=500`: The number of initial observations to discard for stationarity (default is 500).

# Returns
- A tuple `(ε, σ²)` where:
  - `ε`: A vector of simulated residuals (innovations) of length `n`.
  - `σ²`: A vector of simulated conditional variances of length `n`.

# Details
The function generates a total of `n + burnout` observations, computes the conditional variances using the GARCH(p,q) or GJR-GARCH specification, and returns only the last `n` values after the burnout period. The variance is floored at 1e-8 to prevent numerical issues. The residuals are drawn from the specified distribution in the model.

# Example
model = GARCHModel(0.1, [0.2], [0.7], nothing, Normal())
ε, σ² = simulate_garch(model, 1000)
"""
function simulate(rng::AbstractRNG,model::GARCHModel, n::Int; burnout::Int=500, innovation_dist::Distribution = Distributions.Normal())
    p = model.α === nothing ? 0 : length(model.α)
    q = model.β === nothing ? 0 : length(model.β)
    γ = model.γ
    total = n + burnout

    ε = zeros(Float64, total)
    σ2 = zeros(Float64, total)

    for t in 1:total
        arch_sum = p == 0 ? 0.0 : sum(model.α[i] * (t-i > 0 ? ε[t-i]^2 : 0.0) for i in 1:p)
        garch_sum = q == 0 ? 0.0 : sum(model.β[j] * (t-j > 0 ? σ2[t-j] : 0.0) for j in 1:q)
        gjr_sum = γ === nothing ? 0.0 : γ * (t-1 > 0 && ε[t-1] < 0 ? ε[t-1]^2 : 0.0)

        σ2[t] = max(model.ω + arch_sum + garch_sum + gjr_sum, 1e-8)
        ε[t] = rand(rng, innovation_dist)  * sqrt(σ2[t])
    end
    return ε[burnout+1:end], σ2[burnout+1:end]
end

function simulate(model::GARCHModel, n::Int; burnout::Int=500, innovation_dist::Distribution = Distributions.Normal())
    return simulate(Random.default_rng(), model, n; burnout=burnout, innovation_dist=innovation_dist)
end

"""
    simulate_next_observation(data::Vector{Float64}, model::GARCHModel)

Simulate the next observation from a GARCH model given observed data.
# Arguments
- `data::Vector{Float64}`: Observed time series data (e.g., returns or residuals).
- `model::GARCHModel`: Fitted GARCH model.
# Returns
- A tuple `(ε_next, σ2_next)` where:
  - `ε_next`: The simulated next observation (residual).
  - `σ2_next`: The forecasted conditional variance for the next observation.
# Example
ε_next, σ2_next = simulate_next_observation(data, model)

"""
function simulate_next_observation(data::Vector{Float64}, model::GARCHModel; innovation_dist::Distribution = Distributions.Normal())
    σ2_next = forecast_variance(data, model)

    ε_next = rand(innovation_dist) * sqrt(σ2_next)
    return ε_next, σ2_next
end
