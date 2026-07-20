using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

Random.seed!(246241)

model = GARCHModel(0.05, [0.05], [0.9], nothing, Normal()) # garch with gaussian QML
τ = 0.90

n_sim = 5000
n_obs = 5000
burnout = 1000
true_mc = 1_000_000
innovation_dist = Distributions.TDist(4)

z975 = quantile(Normal(), 0.975) # 95% CI critical value


# ------------------------------------------------------------
# True innovation expectile via large MC
# ------------------------------------------------------------

println("Computing population innovation expectile...")
rng_truth = MersenneTwister(9999)
y_mc, s2_mc = simulate(rng_truth, model, true_mc; burnout=burnout, innovation_dist=innovation_dist)
true_expectile = expectile(y_mc ./ sqrt.(s2_mc), τ)

println("τ = $τ")
println("Population innovation expectile ≈ $(round(true_expectile, digits=6))")

# ------------------------------------------------------------
# Check current asymptotic variance formulas on one long sample
# ------------------------------------------------------------

println("\nSingle-sample diagnostic:")
y0, s20 = simulate(MersenneTwister(111), model, 10_000; burnout=burnout, innovation_dist=innovation_dist)

v_known = expectile_var(y0, s20, τ)
println("Known-scale asymptotic sd      = $(v_known >= 0 ? sqrt(v_known) : NaN)")

fit0 = GARCHModels.estimate(y0, 1, 1; gjr=false, dist=model.dist)
v_est = expectile_var(y0, fit0, τ; verbose=true)
println("Estimated-GARCH asymptotic sd  = $(v_est >= 0 ? sqrt(v_est) : NaN)")

v_cond = conditional_expectile_var(y0, fit0, τ; verbose=true)
println("Conditional expectile asymptotic sd = $(v_cond >= 0 ? sqrt(v_cond) : NaN)")

# ------------------------------------------------------------
# 1. MC validation with known conditional variance
# ------------------------------------------------------------

println("\n[1] Validating residual expectile variance with known scale...")

expectile_estimates_1 = zeros(n_sim)
variance_estimates_1 = zeros(n_sim)

for i in 1:n_sim
    rng = MersenneTwister(10_000 + i)
    y_sim, s2_sim = simulate(rng, model, n_obs; burnout=burnout, innovation_dist=innovation_dist)

    η = residuals(y_sim, s2_sim)
    ξhat = expectile(η, τ)
    vhat = expectile_var(y_sim, s2_sim, τ)

    expectile_estimates_1[i] = ξhat - true_expectile
    variance_estimates_1[i] = vhat
end

mc_sd_1 = sqrt(n_obs * mean(expectile_estimates_1 .^ 2))
asy_sd_1 = mean(sqrt.(variance_estimates_1))

cover_1 = mean(abs.(sqrt(n_obs) .* expectile_estimates_1 ./ sqrt.(variance_estimates_1)) .<= z975)

println("MC root-n sd                    = $(round(mc_sd_1, digits=6))")
println("Mean asymptotic root-n sd       = $(round(asy_sd_1, digits=6))")
println("95% asymptotic CI coverage      = $(round(cover_1, digits=4))")

# ------------------------------------------------------------
# 2. MC validation with estimated GARCH model
# ------------------------------------------------------------

println("\n[2] Validating residual expectile variance with estimated GARCH...")

expectile_estimates_2 = zeros(n_sim)
variance_estimates_2 = zeros(n_sim)

for i in 1:n_sim
    rng = MersenneTwister(10_000 + i)
    y_sim, _ = simulate(rng, model, n_obs; burnout=burnout, innovation_dist=innovation_dist)

    fit = GARCHModels.estimate(y_sim, 1, 1; gjr=false, dist=model.dist)
    resid = residuals(y_sim, fit)

    ξhat = expectile(resid, τ)
    vhat = expectile_var(y_sim, fit, τ)

    expectile_estimates_2[i] = ξhat - true_expectile
    variance_estimates_2[i] = vhat
end

valid_2 = isfinite.(variance_estimates_2) .& (variance_estimates_2 .> 0)

mc_sd_2 = sqrt(n_obs * mean(expectile_estimates_2 .^ 2))
asy_sd_2 = mean(sqrt.(variance_estimates_2[valid_2]))

cover_2 = mean(abs.(sqrt(n_obs) .* expectile_estimates_2[valid_2] ./ sqrt.(variance_estimates_2[valid_2])) .<= z975)

println("Valid variance fraction         = $(round(mean(valid_2), digits=4))")
println("MC root-n sd                    = $(round(mc_sd_2, digits=6))")
println("Mean asymptotic root-n sd       = $(round(asy_sd_2, digits=6))")
println("95% asymptotic CI coverage      = $(round(cover_2, digits=4))")

# ------------------------------------------------------------
# 3. MC validation of one-step-ahead conditional expectile variance
# ------------------------------------------------------------

println("\n[3] Validating conditional expectile variance...")

cond_expectile_errors = zeros(n_sim)
cond_variance_estimates = zeros(n_sim)

for i in 1:n_sim
    rng = MersenneTwister(10_000 + i)
    y_sim, _ = GARCHModels.simulate(rng, model, n_obs; burnout=burnout, innovation_dist=innovation_dist)

    fit = GARCHModels.estimate(y_sim, 1, 1; gjr=false, dist=model.dist)

    s2_true_pred = forecast_variance(y_sim, model)
    s2_hat_pred = forecast_variance(y_sim, fit)

    resid = GARCHModels.residuals(y_sim, fit)

    ce_true = sqrt(s2_true_pred) * true_expectile
    ce_hat = expectile(resid, τ) * sqrt(s2_hat_pred)
    vhat = conditional_expectile_var(y_sim, fit, τ)

    cond_expectile_errors[i] = ce_hat - ce_true
    cond_variance_estimates[i] = vhat
end

valid_3 = isfinite.(cond_variance_estimates) .& (cond_variance_estimates .> 0)

mc_sd_3 = sqrt(n_obs * mean(cond_expectile_errors .^ 2))
asy_sd_3 = mean(sqrt.(cond_variance_estimates[valid_3]))

cover_3 = mean(abs.(sqrt(n_obs) .* cond_expectile_errors[valid_3] ./ sqrt.(cond_variance_estimates[valid_3])) .<= z975)

println("Valid variance fraction         = $(round(mean(valid_3), digits=4))")
println("MC root-n sd                    = $(round(mc_sd_3, digits=6))")
println("Mean asymptotic root-n sd       = $(round(asy_sd_3, digits=6))")
println("95% asymptotic CI coverage      = $(round(cover_3, digits=4))")

# ------------------------------------------------------------
# 4. MC validation of GARCH parameter variance
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4. MC validation of GARCH parameter variance
# ------------------------------------------------------------

println("\n[4] Validating GARCH parameter variance...")

θ_true = convert(Vector{Float64}, model)
k = length(θ_true)

θ_errors = zeros(n_sim, k)
θ_variance_estimates = Vector{Matrix{Float64}}(undef, n_sim)

for i in 1:n_sim
    rng = MersenneTwister(10_000 + i)
    y_sim, _ = simulate(rng, model, n_obs; burnout=burnout, innovation_dist=innovation_dist)

    fit = GARCHModels.estimate(y_sim, 1, 1; gjr=false, dist=model.dist)

    θ_hat = convert(Vector{Float64}, fit)
    Vhat = GARCHModels.garch_parameter_variance(y_sim, fit; rootn=true)

    θ_errors[i, :] = θ_hat .- θ_true
    θ_variance_estimates[i] = Vhat
end

# validity of estimated covariance matrices
valid_4 = falses(n_sim)
for i in 1:n_sim
    Vhat = θ_variance_estimates[i]
    valid_4[i] =
        all(isfinite, Vhat) &&
        size(Vhat,1) == k &&
        size(Vhat,2) == k &&
        all(diag(Vhat) .> 0)
end

println("Valid variance matrix fraction  = $(round(mean(valid_4), digits=4))")

param_names = ["ω", "α₁", "β₁"]

for j in 1:k
    mc_sd_j = sqrt(n_obs * mean(θ_errors[:, j].^2))

    asy_sd_j = mean([
        sqrt(θ_variance_estimates[i][j, j])
        for i in 1:n_sim if valid_4[i]
    ])

    cover_j = mean([
        abs(sqrt(n_obs) * θ_errors[i, j] / sqrt(θ_variance_estimates[i][j, j])) <= z975
        for i in 1:n_sim if valid_4[i]
    ])

    println("Parameter $(param_names[j]):")
    println("  MC root-n sd               = $(round(mc_sd_j, digits=6))")
    println("  Mean asymptotic root-n sd  = $(round(asy_sd_j, digits=6))")
    println("  95% asymptotic CI coverage = $(round(cover_j, digits=4))")
end

# alpha + beta
ab_true = θ_true[2] + θ_true[3]
ab_errors = θ_errors[:, 2] .+ θ_errors[:, 3]

ab_valid = falses(n_sim)
ab_var = fill(NaN, n_sim)

for i in 1:n_sim
    if valid_4[i]
        Vhat = θ_variance_estimates[i]
        ab_var[i] = Vhat[2,2] + Vhat[3,3] + 2*Vhat[2,3]
        ab_valid[i] = isfinite(ab_var[i]) && ab_var[i] > 0
    end
end

mc_sd_ab = sqrt(n_obs * mean(ab_errors.^2))
asy_sd_ab = mean(sqrt.(ab_var[ab_valid]))
cover_ab = mean(abs.(sqrt(n_obs) .* ab_errors[ab_valid] ./ sqrt.(ab_var[ab_valid])) .<= z975)

println("Parameter α₁+β₁:")
println("  True value                 = $(round(ab_true, digits=6))")
println("  MC root-n sd               = $(round(mc_sd_ab, digits=6))")
println("  Mean asymptotic root-n sd  = $(round(asy_sd_ab, digits=6))")
println("  95% asymptotic CI coverage = $(round(cover_ab, digits=4))")
println("  Valid variance fraction    = $(round(mean(ab_valid), digits=4))")

# ------------------------------------------------------------
# 5. Simple summary
# ------------------------------------------------------------

println("\nSummary")
println("----------------------------------------------------")
println("Known scale:      MC sd = $(round(mc_sd_1, digits=4)) | Asy sd = $(round(asy_sd_1, digits=4)) | Cov = $(round(cover_1, digits=4))")
println("Estimated GARCH:  MC sd = $(round(mc_sd_2, digits=4)) | Asy sd = $(round(asy_sd_2, digits=4)) | Cov = $(round(cover_2, digits=4)) | valid = $(round(mean(valid_2), digits=4))")
println("Conditional pred: MC sd = $(round(mc_sd_3, digits=4)) | Asy sd = $(round(asy_sd_3, digits=4)) | Cov = $(round(cover_3, digits=4)) | valid = $(round(mean(valid_3), digits=4))")
println("----------------------------------------------------")
println("\nGARCH parameter coverage summary")
println("----------------------------------------------------")

for j in 1:k
    mc_sd_j = sqrt(n_obs * mean(θ_errors[:, j].^2))

    asy_sd_j = mean([
        sqrt(θ_variance_estimates[i][j, j])
        for i in 1:n_sim if valid_4[i]
    ])

    cover_j = mean([
        abs(sqrt(n_obs) * θ_errors[i, j] / sqrt(θ_variance_estimates[i][j, j])) <= z975
        for i in 1:n_sim if valid_4[i]
    ])

    println("$(param_names[j]): MC sd = $(round(mc_sd_j, digits=4)) | Asy sd = $(round(asy_sd_j, digits=4)) | Cov = $(round(cover_j, digits=4))")
end
ab_true = θ_true[2] + θ_true[3]
ab_errors = θ_errors[:, 2] .+ θ_errors[:, 3]

ab_valid = falses(n_sim)
ab_var = fill(NaN, n_sim)

for i in 1:n_sim
    if valid_4[i]
        Vhat = θ_variance_estimates[i]
        ab_var[i] = Vhat[2,2] + Vhat[3,3] + 2*Vhat[2,3]
        ab_valid[i] = isfinite(ab_var[i]) && ab_var[i] > 0
    end
end

mc_sd_ab = sqrt(n_obs * mean(ab_errors.^2))
asy_sd_ab = mean(sqrt.(ab_var[ab_valid]))
cover_ab = mean(abs.(sqrt(n_obs) .* ab_errors[ab_valid] ./ sqrt.(ab_var[ab_valid])) .<= z975)

println("α₁+β₁: MC sd = $(round(mc_sd_ab, digits=4)) | Asy sd = $(round(asy_sd_ab, digits=4)) | Cov = $(round(cover_ab, digits=4)) | valid = $(round(mean(ab_valid), digits=4))")
println("----------------------------------------------------")
