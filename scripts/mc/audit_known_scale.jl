using Random
using Statistics
using Distributions
using ConditionalExpectiles.Expectiles

include(joinpath(@__DIR__, "common.jl"))

function audit_known_scale(; n=5000, reps=5000, tau=0.95, dist=Normal())
    spec = MCSpec(model=GARCHModel(0.1, [0.05], [0.90], nothing, Normal()),
                  n=n, n_sim=reps, τ=tau, innovation_dist=dist)
    truth = true_innovation_expectile(spec)
    z = Vector{Float64}(undef, reps)
    estimates = Vector{Float64}(undef, reps)
    estimated_se = Vector{Float64}(undef, reps)
    for rep in 1:reps
        innovations = rand(MersenneTwister(spec.seed + rep), dist, n)
        estimate = expectile(innovations, tau)
        s2, derivative = aux_expectile_var(innovations, estimate, tau)
        se = sqrt(s2 / derivative^2 / n)
        estimates[rep] = estimate
        estimated_se[rep] = se
        z[rep] = (estimate - truth) / se
    end
    println((distribution=dist_label(spec), n=n, reps=reps, truth=truth,
             bias=mean(estimates)-truth, mc_sd=std(estimates), avg_se=mean(estimated_se),
             se_sd_ratio=mean(estimated_se)/std(estimates), z_mean=mean(z), z_sd=std(z),
             coverage=mean(abs.(z) .<= Z975)))
end

audit_known_scale()
audit_known_scale(dist=standardized_t(4))
