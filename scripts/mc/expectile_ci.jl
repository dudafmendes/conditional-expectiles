using Statistics

function summarize_expectile_ci(results::Vector, spec::MCSpec)
    xi = [r.xi_hat for r in results]
    vx = [r.v_xi for r in results]
    valid = [isfinite(v) && v > 0 for v in vx]
    vr = results[valid]
    ses = [sqrt(vx[i] / spec.n) for i in eachindex(vx) if valid[i]]
    out = Dict{String,Any}()
    out["xi_coverage_95"] = isempty(vr) ? NaN : mean(r.xi_cover for r in vr)
    out["xi_lower_miss_95"] = isempty(vr) ? NaN : mean(r.xi_lower_miss for r in vr)
    out["xi_upper_miss_95"] = isempty(vr) ? NaN : mean(r.xi_upper_miss for r in vr)
    out["xi_avg_length_95"] = isempty(vr) ? NaN : mean(r.xi_length for r in vr)
    out["xi_avg_asymptotic_se"] = isempty(ses) ? NaN : mean(ses)
    out["xi_mc_sd"] = std(xi)
    out["xi_se_sd_ratio"] = out["xi_avg_asymptotic_se"] / out["xi_mc_sd"]
    out["xi_valid_vfrac"] = mean(valid)
    out["xi_z_mean"] = isempty(vr) ? NaN : mean(r.xi_z for r in vr)
    out["xi_z_sd"] = isempty(vr) ? NaN : std(r.xi_z for r in vr)
    nv = length(vr)
    p = out["xi_coverage_95"]
    out["xi_se_coverage"] = nv == 0 ? NaN : sqrt(p * (1 - p) / nv)
    out["xi_se_avg_asymptotic_se"] = nv <= 1 ? NaN : std(ses) / sqrt(nv)
    if nv <= 3
        out["xi_z_sd_mcse"] = NaN
    else
        z = [r.xi_z for r in vr]
        zbar = mean(z)
        zsd = out["xi_z_sd"]
        mu4 = mean((z .- zbar).^4)
        var_s2 = (mu4 - ((nv - 3) / (nv - 1)) * zsd^4) / nv
        out["xi_z_sd_mcse"] = sqrt(max(var_s2, 0.0)) / (2 * zsd)
    end
    return out
end
