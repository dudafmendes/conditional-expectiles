using Statistics

function summarize_predicted_expectile_ci(results::Vector, spec::MCSpec)
    ce = [r.ce_hat for r in results]
    ce_true = [r.ce_true for r in results]
    vc = [r.v_ce for r in results]
    ce_err = ce .- ce_true
    valid = [isfinite(v) && v > 0 for v in vc]
    vr = results[valid]
    ses = [sqrt(vc[i] / spec.n) for i in eachindex(vc) if valid[i]]
    out = Dict{String,Any}()
    out["ce_coverage_95"] = isempty(vr) ? NaN : mean(r.ce_cover for r in vr)
    out["ce_lower_miss_95"] = isempty(vr) ? NaN : mean(r.ce_lower_miss for r in vr)
    out["ce_upper_miss_95"] = isempty(vr) ? NaN : mean(r.ce_upper_miss for r in vr)
    out["ce_avg_length_95"] = isempty(vr) ? NaN : mean(r.ce_length for r in vr)
    out["ce_avg_asymptotic_se"] = isempty(ses) ? NaN : mean(ses)
    out["ce_mc_sd"] = std(ce_err)
    out["ce_se_sd_ratio"] = out["ce_avg_asymptotic_se"] / out["ce_mc_sd"]
    out["ce_valid_vfrac"] = mean(valid)
    out["ce_z_mean"] = isempty(vr) ? NaN : mean(r.ce_z for r in vr)
    out["ce_z_sd"] = isempty(vr) ? NaN : std(r.ce_z for r in vr)
    nv = length(vr)
    p = out["ce_coverage_95"]
    out["ce_se_coverage"] = nv == 0 ? NaN : sqrt(p * (1 - p) / nv)
    out["ce_se_avg_asymptotic_se"] = nv <= 1 ? NaN : std(ses) / sqrt(nv)
    if nv <= 3
        out["ce_z_sd_mcse"] = NaN
    else
        z = [r.ce_z for r in vr]
        zbar = mean(z)
        zsd = out["ce_z_sd"]
        mu4 = mean((z .- zbar).^4)
        var_s2 = (mu4 - ((nv - 3) / (nv - 1)) * zsd^4) / nv
        out["ce_z_sd_mcse"] = sqrt(max(var_s2, 0.0)) / (2 * zsd)
    end
    return out
end
