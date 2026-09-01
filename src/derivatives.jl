

"""
Fill the provided matrix with lagged values of `data` for ARCH order `p`.
"""
function garch_design_matrix!(X::Matrix{Float64}, data::Vector{Float64}, p::Int)
    n = length(data)

    for i in 1:n
        for j in 1:p
            X[i,j] = i-j > 0 ? data[i-j] : 0.0
        end
    end
    return X
end

"""
Return a matrix of lagged values for `data` with ARCH order `p`.
"""
function garch_design_matrix(data::Vector{Float64}, p::Int)
    n = length(data)
    X = zeros(Float64, n, p)
    garch_design_matrix!(X, data, p)
    return X
end

"""
Return data reconstructed from residuals `ε` and conditional variances `σ2`.
"""
function data_from_residuals_variance(ε::Vector{Float64}, σ2::Vector{Float64})
    return ε .* sqrt.(σ2), σ2
end

# --------------------------------------------
# helper: kernel derivatives g1, g2 at scale s=1
# --------------------------------------------
function _kernel_derivatives!(g1::AbstractVector, g2::AbstractVector,
                              residuals::AbstractVector, dist::Distribution)
    n = length(residuals)

    if dist isa Normal
        g = (x, s) -> -log(s) - 0.5 * (x / s)^2
        for i in 1:n
            x = residuals[i]
            g1[i] = ForwardDiff.derivative(t -> g(x, t), 1.0)
            g2[i] = ForwardDiff.derivative(t -> ForwardDiff.derivative(z -> g(x, z), t), 1.0)
        end

    elseif dist isa TDist
        ν = dist.ν
        g = (x, s) -> -log(s) - 0.5 * (ν + 1) * log(1 + (x / s)^2 / ν)
        for i in 1:n
            x = residuals[i]
            g1[i] = ForwardDiff.derivative(t -> g(x, t), 1.0)
            g2[i] = ForwardDiff.derivative(t -> ForwardDiff.derivative(z -> g(x, z), t), 1.0)
        end

    else
        error("Unsupported distribution")
    end

    return g1, g2
end

"""Compatibility wrapper returning the common risk path in the legacy order."""
function _garch_variance_gradient(data::Vector{Float64}, model::GARCHModel)
    h, dh, h_next, dh_next = _risk_variance_path_gradient(data, model)
    return dh, h, dh_next, h_next
end

# --------------------------------------------
# main function
# returns covariance matrix of θ̂ by default
# if rootn=true, returns covariance of √n(θ̂-θ0)
# --------------------------------------------
function garch_parameter_variance(data::Vector{Float64}, model::GARCHModel; rootn::Bool=false)
    n = length(data)
    p,q = length(model.α), length(model.β)
    m = max(p,q,1) # for indexing lags
    n_eff = n - m

    # recursive derivatives and conditional variances
    dh, h, _, _ = _garch_variance_gradient(data, model)

    # standardized residuals
    η = data ./ sqrt.(h)

    # paper uses ∂σ2_t / σ_t = (1/2) ∂h_t / h_t
    D = similar(dh)
    for t in 1:n
        @views D[t, :] .= 0.5 .* dh[t, :] ./ h[t]
    end

    # truncate to effective sample size for estimation
    D = D[(m+1):end, :]
    η = η[(m+1):end]

    # I_hat = average outer product
    I_hat = (D' * D) / n_eff

    # kernel derivative terms
    g1 = zeros(Float64, n_eff)
    g2 = zeros(Float64, n_eff)
    _kernel_derivatives!(g1, g2, η, model.dist)

    Eg12 = dot(g1,g1)/n_eff # mean(g1 .^ 2)
    Eg2  = mean(g2)

    # τ_h^2 from the paper
    tau_h2 = Eg12 / (Eg2^2)

    # asymptotic covariance of sqrt(n)(θ̂-θ0)
    Σ_rootn = Symmetric(tau_h2 * pinv(Symmetric(I_hat)))

    # covariance of θ̂ itself
    return rootn ? Σ_rootn : Σ_rootn / n
    # return (
    #     vcov = rootn ? Σ_rootn : Σ_rootn / n,
    #     rootn_vcov = Σ_rootn,
    #     I_hat = I_hat,
    #     tau_h2 = tau_h2,
    #     h = h,
    #     D = D,
    #     residuals = η
    # )
end
