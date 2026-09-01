using Dates
using DelimitedFiles
using Distributions
using Printf
using Statistics

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles
using ConditionalExpectiles.GaoSongRisk

# Replicates the empirical crypto application with the current asymptotic
# inference code. Run with: julia -t auto --project=. scripts/applications_crypto.jl
# Optional quick run: APP_MAX_WINDOWS=20 julia -t auto --project=. scripts/applications_crypto.jl

const ROOT = abspath(joinpath(@__DIR__, ".."))
const DATA_FILE = joinpath(ROOT, "data", "crypto_data.csv")
const OUT_DIR = joinpath(ROOT, "output", "applications")

const TICKER_LABELS = Dict(
    "^GSPC" => "SPX",
    "EURUSD=X" => "EUR",
    "BTC-USD" => "BTC",
    "ETH-USD" => "ETH",
    "BNB-USD" => "BNB",
    "ADA-USD" => "ADA",
)

const PLOT_ORDER = ["BNB", "BTC", "ETH", "EUR"]
const WINDOW = parse(Int, get(ENV, "APP_WINDOW", "1000"))
const VAR_ALPHA = parse(Float64, get(ENV, "APP_VAR_ALPHA", "0.01"))
const ES_ALPHA = parse(Float64, get(ENV, "APP_ES_ALPHA", "0.025"))
const ES_ALPHA_1PCT = parse(Float64, get(ENV, "APP_ES_ALPHA_1PCT", "0.01"))
const FIXED_XP_TAU = parse(Float64, get(ENV, "APP_FIXED_XP_TAU", "0.01"))
const CI_LEVEL = parse(Float64, get(ENV, "APP_CI_LEVEL", "0.95"))
const MAX_WINDOWS = parse(Int, get(ENV, "APP_MAX_WINDOWS", "0"))
const Z_CI = quantile(Normal(), 0.5 + CI_LEVEL / 2)

struct AssetSeries
    ticker::String
    label::String
    dates::Vector{Date}
    prices::Vector{Float64}
    returns::Vector{Float64}
end

function parse_float_cell(x)
    s = strip(String(x))
    isempty(s) && return missing
    return parse(Float64, s)
end

function read_assets(path::AbstractString)
    raw, header = readdlm(path, ',', String; header=true)
    names = vec(header)
    idx = Dict(name => i for (i, name) in enumerate(names))
    by_ticker = Dict{String,Vector{Tuple{Date,Float64}}}()

    for r in axes(raw, 1)
        ticker = raw[r, idx["ticker"]]
        haskey(TICKER_LABELS, ticker) || continue
        price = parse_float_cell(raw[r, idx["price_adjusted"]])
        ismissing(price) && continue
        date = Date(raw[r, idx["ref_date"]])
        push!(get!(by_ticker, ticker, Tuple{Date,Float64}[]), (date, price))
    end

    assets = AssetSeries[]
    selected_tickers = [t for t in keys(by_ticker) if TICKER_LABELS[t] in PLOT_ORDER]
    for ticker in sort(selected_tickers; by=t -> findfirst(==(TICKER_LABELS[t]), PLOT_ORDER))
        rows = sort(by_ticker[ticker]; by=first)
        dates = Date[]
        prices = Float64[]
        returns = Float64[]
        for i in 2:length(rows)
            p0 = rows[i - 1][2]
            p1 = rows[i][2]
            if p0 > 0 && p1 > 0
                push!(dates, rows[i][1])
                push!(prices, p1)
                push!(returns, 100 * log(p1 / p0))
            end
        end
        push!(assets, AssetSeries(ticker, TICKER_LABELS[ticker], dates, prices, returns))
    end

    return sort(assets; by=a -> findfirst(==(a.label), PLOT_ORDER))
end

function empirical_quantile(x::AbstractVector{<:Real}, p::Real)
    xs = sort(collect(Float64, x))
    n = length(xs)
    n == 0 && return NaN
    p <= 0 && return xs[1]
    p >= 1 && return xs[end]
    h = 1 + (n - 1) * p
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo == hi && return xs[lo]
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])
end

function skewness(x)
    s = std(x)
    s == 0 && return 0.0
    z = (x .- mean(x)) ./ s
    return mean(z .^ 3)
end

function kurtosis_raw(x)
    s = std(x)
    s == 0 && return 0.0
    z = (x .- mean(x)) ./ s
    return mean(z .^ 4)
end

function tau_for_quantile(x::AbstractVector{<:Real}, q::Real)
    below = sum(max(q - v, 0.0) for v in x)
    above = sum(max(v - q, 0.0) for v in x)
    denom = below + above
    return denom <= 0 ? NaN : below / denom
end

alpha_for_expectile(x::AbstractVector{<:Real}, xi::Real) = sum(v <= xi ? 1 : 0 for v in x) / length(x)

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

fmt(x) = isfinite(x) ? @sprintf("%.10g", x) : ""

function descriptive_rows(assets)
    rows = Vector{Vector{String}}()
    for a in assets
        y = a.returns
        push!(rows, [
            a.label,
            string(length(y)),
            string(first(a.dates)),
            string(last(a.dates)),
            fmt(minimum(y)),
            fmt(empirical_quantile(y, 0.25)),
            fmt(empirical_quantile(y, 0.50)),
            fmt(empirical_quantile(y, 0.75)),
            fmt(maximum(y)),
            fmt(mean(y)),
            fmt(std(y)),
            fmt(skewness(y)),
            fmt(kurtosis_raw(y)),
        ])
    end
    return rows
end

function rolling_forecast_at(asset::AssetSeries, i::Int)
    y = asset.returns
    ywin = y[(i - WINDOW):(i - 1)]

    fit = try
        GARCHModels.estimate(ywin, 1, 1; gjr=true, dist=Normal())
    catch err
        println("Skipping ", asset.label, " window ending ", asset.dates[i - 1], ": ", err)
        return nothing
    end

    resid = GARCHModels.residuals(ywin, fit)
    sigma_next = sqrt(GARCHModels.forecast_variance(ywin, fit))

    var_risk = gao_song_fhs_intervals(ywin, fit, VAR_ALPHA; ci_level=CI_LEVEL)
    q_var = var_risk.threshold
    tau_alpha = tau_for_quantile(resid, q_var)
    xi = Expectiles.expectile(resid, tau_alpha)
    v_xi = try
        Expectiles.expectile_var(ywin, fit, tau_alpha)
    catch err
        println("Residual expectile variance failed for ", asset.label, " on ", asset.dates[i], ": ", err)
        NaN
    end
    xi_se = isfinite(v_xi) && v_xi >= 0 ? sqrt(v_xi / WINDOW) : NaN

    ce = sigma_next * xi
    v_ce = try
        Expectiles.conditional_expectile_var(ywin, fit, tau_alpha)
    catch err
        println("Conditional expectile variance failed for ", asset.label, " on ", asset.dates[i], ": ", err)
        NaN
    end
    ce_se = isfinite(v_ce) && v_ce >= 0 ? sqrt(v_ce / WINDOW) : NaN

    # Fixed-index expectile retained for the VaR/XP backtest.  This is distinct
    # from `ce`, whose rolling level is chosen to match VaR at VAR_ALPHA.
    xi_fixed = Expectiles.expectile(resid, FIXED_XP_TAU)
    xp_fixed = sigma_next * xi_fixed
    xp_fixed_alpha = alpha_for_expectile(resid, xi_fixed)
    v_xp_fixed = try
        Expectiles.conditional_expectile_var(ywin, fit, FIXED_XP_TAU)
    catch err
        println("Conditional fixed-1% expectile variance failed for ", asset.label, " on ", asset.dates[i], ": ", err)
        NaN
    end
    xp_fixed_se = isfinite(v_xp_fixed) && v_xp_fixed >= 0 ? sqrt(v_xp_fixed / WINDOW) : NaN

    es_risk_1pct = gao_song_fhs_intervals(ywin, fit, ES_ALPHA_1PCT; ci_level=CI_LEVEL)
    es_risk = gao_song_fhs_intervals(ywin, fit, ES_ALPHA; ci_level=CI_LEVEL)
    var_alpha = var_risk.var.estimate
    es_alpha = es_risk.es.estimate
    omega_alpha = isfinite(tau_alpha) && tau_alpha > 0 ? 1 / tau_alpha - 1 : NaN
    alpha_tau = alpha_for_expectile(resid, xi)
    garch_parameters = convert(Vector{Float64}, fit)

    return (
        ticker = asset.label,
        date = asset.dates[i],
        window_start = asset.dates[i - WINDOW],
        window_end = asset.dates[i - 1],
        window_size = WINDOW,
        garch_omega = garch_parameters[1],
        garch_alpha1 = garch_parameters[2],
        garch_beta1 = garch_parameters[3],
        garch_gamma = length(garch_parameters) >= 4 ? garch_parameters[4] : NaN,
        realized_return = y[i],
        sigma = sigma_next,
        xi = xi,
        xi_lo = xi - Z_CI * xi_se,
        xi_hi = xi + Z_CI * xi_se,
        ce = ce,
        ce_se = ce_se,
        ce_lo = ce - Z_CI * ce_se,
        ce_hi = ce + Z_CI * ce_se,
        xp_fixed_tau = FIXED_XP_TAU,
        xp_fixed = xp_fixed,
        xp_fixed_se = xp_fixed_se,
        xp_fixed_lo = xp_fixed - Z_CI * xp_fixed_se,
        xp_fixed_hi = xp_fixed + Z_CI * xp_fixed_se,
        xp_fixed_alpha = xp_fixed_alpha,
        es_01 = es_risk_1pct.es.estimate,
        es_01_se = es_risk_1pct.es.interval.se,
        es_01_lo = es_risk_1pct.es.interval.lower,
        es_01_hi = es_risk_1pct.es.interval.upper,
        es_01_var = es_risk_1pct.var.estimate,
        es_01_var_se = es_risk_1pct.var.interval.se,
        var = var_alpha,
        var_se = var_risk.var.interval.se,
        var_lo = var_risk.var.interval.lower,
        var_hi = var_risk.var.interval.upper,
        es = es_alpha,
        es_se = es_risk.es.interval.se,
        es_lo = es_risk.es.interval.lower,
        es_hi = es_risk.es.interval.upper,
        es_var = es_risk.var.estimate,
        es_var_se = es_risk.var.interval.se,
        tau_alpha = tau_alpha,
        omega_alpha = omega_alpha,
        alpha_tau = alpha_tau,
        exception_var = y[i] < var_alpha,
        exception_ce = y[i] < ce,
        exception_xp_fixed = y[i] < xp_fixed,
    )
end

function rolling_forecasts(asset::AssetSeries)
    n = length(asset.returns)
    if n <= WINDOW
        return Any[]
    end

    first_i = WINDOW + 1
    last_i = MAX_WINDOWS > 0 ? min(n, WINDOW + MAX_WINDOWS) : n
    indices = collect(first_i:last_i)
    results = Vector{Any}(undef, length(indices))
    fill!(results, nothing)

    Threads.@threads for k in eachindex(indices)
        results[k] = rolling_forecast_at(asset, indices[k])
    end

    return [r for r in results if r !== nothing]
end

function forecast_rows(forecasts)
    rows = Vector{Vector{String}}()
    for r in forecasts
        push!(rows, [
            r.ticker, string(r.date), string(r.window_start), string(r.window_end), string(r.window_size),
            fmt(r.garch_omega), fmt(r.garch_alpha1), fmt(r.garch_beta1), fmt(r.garch_gamma),
            fmt(r.realized_return), fmt(r.sigma),
            fmt(r.xi), fmt(r.xi_lo), fmt(r.xi_hi),
            fmt(r.ce), fmt(r.ce_se), fmt(r.ce_lo), fmt(r.ce_hi), fmt(CI_LEVEL),
            fmt(r.xp_fixed_tau), fmt(r.xp_fixed), fmt(r.xp_fixed_se), fmt(r.xp_fixed_lo), fmt(r.xp_fixed_hi), fmt(r.xp_fixed_alpha),
            fmt(ES_ALPHA_1PCT), fmt(r.es_01), fmt(r.es_01_se), fmt(r.es_01_lo), fmt(r.es_01_hi), fmt(r.es_01_var), fmt(r.es_01_var_se),
            fmt(VAR_ALPHA), fmt(r.var), fmt(r.var_se), fmt(r.var_lo), fmt(r.var_hi),
            fmt(ES_ALPHA), fmt(r.es_var), fmt(r.es_var_se), fmt(r.es), fmt(r.es_se), fmt(r.es_lo), fmt(r.es_hi),
            fmt(r.tau_alpha), fmt(r.omega_alpha), fmt(r.alpha_tau),
            string(r.exception_var), string(r.exception_ce), string(r.exception_xp_fixed),
        ])
    end
    return rows
end

function main()
    mkpath(OUT_DIR)
    assets = read_assets(DATA_FILE)

    write_csv(
        joinpath(OUT_DIR, "descriptive_statistics.csv"),
        ["ticker", "T", "start", "end", "min", "q25", "median", "q75", "max", "mean", "std_dev", "skew", "kurtosis"],
        descriptive_rows(assets),
    )

    forecasts = Any[]
    for label in PLOT_ORDER
        asset = only([a for a in assets if a.label == label])
        println("Rolling GJR-GARCH forecasts for ", label, " on ", Threads.nthreads(), " Julia thread(s)...")
        append!(forecasts, rolling_forecasts(asset))
    end

    write_csv(
        joinpath(OUT_DIR, "rolling_risk_forecasts.csv"),
        ["ticker", "date", "window_start", "window_end", "window_size", "garch_omega", "garch_alpha1", "garch_beta1", "garch_gamma", "realized_return", "sigma", "xi", "xi_lo", "xi_hi", "ce", "ce_se", "ce_lo", "ce_hi", "ci_level", "xp_fixed_tau", "xp_fixed", "xp_fixed_se", "xp_fixed_lo", "xp_fixed_hi", "xp_fixed_alpha", "es_01_alpha", "es_01", "es_01_se", "es_01_lo", "es_01_hi", "es_01_var", "es_01_var_se", "var_alpha", "var", "var_se", "var_lo", "var_hi", "es_alpha", "es_var", "es_var_se", "es", "es_se", "es_lo", "es_hi", "tau_alpha", "omega_alpha", "alpha_tau", "exception_var", "exception_ce", "exception_xp_fixed"],
        forecast_rows(forecasts),
    )

    println("Application outputs written to ", OUT_DIR)
    println("CLT interval level: ", CI_LEVEL, "; window: ", WINDOW, "; VaR alpha: ", VAR_ALPHA,
            "; ES levels: ", ES_ALPHA_1PCT, " and ", ES_ALPHA)
end

main()
