using DelimitedFiles
using Distributions
using Optim
using Printf
using Statistics

# Fixed-index application backtests at alpha = delta = tau = 1%.
# Run after applications_crypto.jl:
#   julia --project=. scripts/application_backtests.jl

const ROOT = abspath(joinpath(@__DIR__, ".."))
const APP_DIR = joinpath(ROOT, "output", "applications")
const FORECAST_FILE = joinpath(APP_DIR, "rolling_risk_forecasts.csv")
const BACKTEST_DIR = joinpath(APP_DIR, "backtests")
const ALPHA = 0.01
const DELTA = 0.01
const TAU = 0.01
const LOSS_TAU = 1 - TAU
const ASSETS = ["BNB", "BTC", "ETH", "EUR"]

struct ForecastRow
    ticker::String
    loss::Float64
    var_loss::Float64
    es_loss::Float64
    xp_loss::Float64
end

fmt(x) = isfinite(x) ? @sprintf("%.10g", x) : ""

function read_forecasts(path)
    raw, header = readdlm(path, ',', String; header=true)
    index = Dict(String(h) => i for (i, h) in enumerate(vec(header)))
    required = ("ticker", "realized_return", "var_alpha", "var", "es_01_alpha", "es_01", "xp_fixed_tau", "xp_fixed")
    all(haskey(index, field) for field in required) || error("$(path) is stale. Re-run applications_crypto.jl.")

    levels = (
        alpha = parse(Float64, raw[1, index["var_alpha"]]),
        delta = parse(Float64, raw[1, index["es_01_alpha"]]),
        tau = parse(Float64, raw[1, index["xp_fixed_tau"]]),
    )
    isapprox(levels.alpha, ALPHA; atol=1e-12) || error("Backtests require var_alpha = $(ALPHA), found $(levels.alpha).")
    isapprox(levels.delta, DELTA; atol=1e-12) || error("Backtests require es_01_alpha = $(DELTA), found $(levels.delta).")
    isapprox(levels.tau, TAU; atol=1e-12) || error("Backtests require xp_fixed_tau = $(TAU), found $(levels.tau).")

    rows = ForecastRow[]
    for r in axes(raw, 1)
        ticker = raw[r, index["ticker"]]
        ticker in ASSETS || continue
        push!(rows, ForecastRow(
            ticker,
            -parse(Float64, raw[r, index["realized_return"]]),
            -parse(Float64, raw[r, index["var"]]),
            -parse(Float64, raw[r, index["es_01"]]),
            -parse(Float64, raw[r, index["xp_fixed"]]),
        ))
    end
    isempty(rows) && error("No fixed-level forecasts found for $(join(ASSETS, ", ")).")
    return rows
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        foreach(row -> println(io, join(row, ",")), rows)
    end
end

function bernoulli_loglik(k, n, probability)
    p = clamp(probability, eps(), 1 - eps())
    k * log(p) + (n - k) * log1p(-p)
end

function kupiec_test(hits, probability)
    n, observed = length(hits), sum(hits)
    rate = observed / n
    lr = max(0.0, -2 * (bernoulli_loglik(observed, n, probability) - bernoulli_loglik(observed, n, rate)))
    return (expected=n * probability, observed=observed, rate=rate, lr=lr, pvalue=ccdf(Chisq(1), lr))
end

function christoffersen_independence_test(hits)
    n00 = n01 = n10 = n11 = 0
    for t in 2:length(hits)
        if !hits[t - 1] && !hits[t]
            n00 += 1
        elseif !hits[t - 1] && hits[t]
            n01 += 1
        elseif hits[t - 1] && !hits[t]
            n10 += 1
        else
            n11 += 1
        end
    end
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    pooled = total == 0 ? 0.0 : (n01 + n11) / total
    p01, p11 = n0 == 0 ? 0.0 : n01 / n0, n1 == 0 ? 0.0 : n11 / n1
    lr = max(0.0, -2 * (bernoulli_loglik(n01 + n11, total, pooled) - bernoulli_loglik(n01, n0, p01) - bernoulli_loglik(n11, n1, p11)))
    return (lr=lr, pvalue=ccdf(Chisq(1), lr))
end

function duration_test(hits)
    hit_dates = findall(identity, hits)
    length(hit_dates) < 3 && return (count=max(length(hit_dates) - 1, 0), shape=NaN, lr=NaN, pvalue=NaN)
    durations = Float64.(diff(hit_dates))
    mean_duration = mean(durations)
    weibull_ll(scale, shape) = scale <= 0 || shape <= 0 ? -Inf : sum(log(shape) - shape * log(scale) + (shape - 1) * log(d) - (d / scale)^shape for d in durations)
    objective(z) = -weibull_ll(exp(z[1]), exp(z[2]))
    fit = optimize(objective, log.([mean_duration, 1.0]), BFGS())
    scale, shape = exp.(Optim.minimizer(fit))
    lr = max(0.0, -2 * (weibull_ll(mean_duration, 1.0) + Optim.minimum(fit)))
    return (count=length(durations), shape=shape, lr=lr, pvalue=ccdf(Chisq(1), lr))
end

function normal_mean_test(values)
    x = filter(isfinite, values)
    n = length(x)
    n < 2 && return (n=n, mean=NaN, statistic=NaN, pvalue=NaN)
    estimate = mean(x)
    standard_error = std(x) / sqrt(n)
    statistic = standard_error > 0 ? estimate / standard_error : NaN
    pvalue = isfinite(statistic) ? 2 * ccdf(Normal(), abs(statistic)) : NaN
    return (n=n, mean=estimate, statistic=statistic, pvalue=pvalue)
end

# First-order identification moments for ES and expectiles, respectively. The
# return-side lower expectile at tau becomes an upper expectile at 1 - tau once
# returns are converted to losses.
es_residual(loss, var, es) = loss > var ? loss - es : NaN
xp_moment(loss, xp) = (loss - xp) * abs((loss < xp ? 1.0 : 0.0) - LOSS_TAU)

function grouped(rows)
    Dict(asset => [row for row in rows if row.ticker == asset] for asset in ASSETS)
end

function var_calibration_rows(rows)
    output = Vector{Vector{String}}()
    for asset in ASSETS
        rs = grouped(rows)[asset]
        hits = [row.loss > row.var_loss for row in rs]
        uc = kupiec_test(hits, ALPHA)
        ind = christoffersen_independence_test(hits)
        duration = duration_test(hits)
        conditional_lr = uc.lr + ind.lr
        push!(output, [asset, string(length(rs)), fmt(uc.expected), string(uc.observed), fmt(uc.rate), fmt(uc.lr), fmt(uc.pvalue), fmt(ind.lr), fmt(ind.pvalue), fmt(conditional_lr), fmt(ccdf(Chisq(2), conditional_lr)), string(duration.count), fmt(duration.shape), fmt(duration.lr), fmt(duration.pvalue)])
    end
    output
end

function fixed_level_rows(rows)
    output = Vector{Vector{String}}()
    for asset in ASSETS
        rs = grouped(rows)[asset]
        var_hits = [row.loss > row.var_loss for row in rs]
        xp_hits = [row.loss > row.xp_loss for row in rs]
        uc = kupiec_test(var_hits, ALPHA)
        ind = christoffersen_independence_test(var_hits)
        duration = duration_test(var_hits)
        xp = normal_mean_test([xp_moment(row.loss, row.xp_loss) for row in rs])
        n = length(rs)
        push!(output, [asset, string(n), fmt(uc.expected), string(uc.observed), fmt(uc.rate), fmt(mean(row.var_loss - row.loss for row in rs)), fmt(n * TAU), string(sum(xp_hits)), fmt(mean(xp_hits)), fmt(mean(row.xp_loss - row.loss for row in rs)), fmt(uc.pvalue), fmt(ind.pvalue), fmt(ccdf(Chisq(2), uc.lr + ind.lr)), fmt(duration.pvalue), fmt(xp.pvalue)])
    end
    output
end

function residual_identification_rows(rows)
    output = Vector{Vector{String}}()
    for asset in ASSETS
        rs = grouped(rows)[asset]
        es = normal_mean_test([es_residual(row.loss, row.var_loss, row.es_loss) for row in rs])
        xp = normal_mean_test([xp_moment(row.loss, row.xp_loss) for row in rs])
        push!(output, [asset, string(es.n), fmt(es.mean), fmt(es.statistic), fmt(es.pvalue), string(xp.n), fmt(xp.mean), fmt(xp.statistic), fmt(xp.pvalue)])
    end
    output
end

function main()
    mkpath(BACKTEST_DIR)
    rows = read_forecasts(FORECAST_FILE)
    write_csv(joinpath(BACKTEST_DIR, "var_calibration_tests.csv"), ["ticker", "n", "expected_exceptions", "observed_exceptions", "exception_rate", "kupiec_lr", "kupiec_pvalue", "independence_lr", "independence_pvalue", "conditional_coverage_lr", "conditional_coverage_pvalue", "duration_count", "weibull_shape", "duration_lr", "duration_pvalue"], var_calibration_rows(rows))
    write_csv(joinpath(BACKTEST_DIR, "fixed_level_var_xp_backtests.csv"), ["ticker", "n", "var_expected_exceptions", "var_observed_exceptions", "var_exception_rate", "var_avg_buffer", "xp_expected_exceptions", "xp_observed_exceptions", "xp_exception_rate", "xp_avg_buffer", "kupiec_pvalue", "christoffersen_independence_pvalue", "christoffersen_conditional_coverage_pvalue", "duration_pvalue", "xp_identification_pvalue"], fixed_level_rows(rows))
    write_csv(joinpath(BACKTEST_DIR, "residual_identification_tests.csv"), ["ticker", "es_1pct_n", "es_1pct_mean", "es_1pct_z", "es_1pct_pvalue", "xp_1pct_n", "xp_1pct_identification_mean", "xp_1pct_identification_z", "xp_1pct_identification_pvalue"], residual_identification_rows(rows))
    println("Fixed-level backtests written to ", BACKTEST_DIR)
    println("Levels: alpha = delta = tau = ", ALPHA, "; loss convention: L_t = -r_t.")
end

main()
