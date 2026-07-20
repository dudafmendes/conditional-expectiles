using Dates
using DelimitedFiles
using Distributions
using Printf
using Statistics

using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles

# Replicates the empirical crypto application with the current asymptotic
# inference code. Run with: julia -t auto --project=. scripts/applications_crypto.jl
# Optional quick run: APP_MAX_WINDOWS=20 julia -t auto --project=. scripts/applications_crypto.jl

const ROOT = abspath(joinpath(@__DIR__, ".."))
const DATA_FILE = joinpath(ROOT, "data", "crypto_data.csv")
const OUT_DIR = joinpath(ROOT, "output", "applications")
const PLOT_DIR = joinpath(OUT_DIR, "plots")

const TICKER_LABELS = Dict(
    "^GSPC" => "SPX",
    "EURUSD=X" => "EUR",
    "BTC-USD" => "BTC",
    "ETH-USD" => "ETH",
    "BNB-USD" => "BNB",
    "ADA-USD" => "ADA",
)

const PLOT_ORDER = ["SPX", "EUR", "BTC", "ETH", "BNB", "ADA"]
const CRYPTO_ORDER = ["BTC", "ETH", "BNB", "ADA"]
const FOCUS_ORDER = ["BTC", "SPX"]
const WINDOW = parse(Int, get(ENV, "APP_WINDOW", "1000"))
const TAU = parse(Float64, get(ENV, "APP_TAU", "0.01"))
const ALPHA = parse(Float64, get(ENV, "APP_ALPHA", "0.01"))
const CI_LEVEL = parse(Float64, get(ENV, "APP_CI_LEVEL", "0.90"))
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
    for ticker in sort(collect(keys(by_ticker)); by=t -> TICKER_LABELS[t])
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

function downside_es(x::AbstractVector{<:Real}, alpha::Real)
    q = empirical_quantile(x, alpha)
    tail = [v for v in x if v <= q]
    return isempty(tail) ? q : mean(tail)
end

function downside_es_at_threshold(x::AbstractVector{<:Real}, threshold::Real)
    tail = [v for v in x if v <= threshold]
    return isempty(tail) ? threshold : mean(tail)
end

function tau_for_quantile(x::AbstractVector{<:Real}, q::Real)
    below = sum(max(q - v, 0.0) for v in x)
    above = sum(max(v - q, 0.0) for v in x)
    denom = below + above
    return denom <= 0 ? NaN : below / denom
end

alpha_for_expectile(x::AbstractVector{<:Real}, xi::Real) = sum(v <= xi ? 1 : 0 for v in x) / length(x)

function drawdown(prices::Vector{Float64})
    out = similar(prices)
    peak = prices[1]
    for i in eachindex(prices)
        peak = max(peak, prices[i])
        out[i] = 100 * (peak - prices[i]) / peak
    end
    return out
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

fmt(x) = isfinite(x) ? @sprintf("%.10g", x) : ""
fmt2(x) = isfinite(x) ? @sprintf("%.2f", x) : ""

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

function svg_escape(s)
    replace(String(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
end

function line_svg(path, panels; width=1200, panel_h=210, title="")
    left, right, top, bottom = 58, 20, 34, 36
    height = max(1, length(panels)) * panel_h + 28
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="#ffffff"/>""")
        !isempty(title) && println(io, """<text x="$(width/2)" y="22" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">$(svg_escape(title))</text>""")
        for (pi, p) in enumerate(panels)
            x = p[:x]
            ys = p[:ys]
            labels = p[:labels]
            yvals = [v for v in reduce(vcat, ys) if isfinite(v)]
            isempty(yvals) && continue
            ymin, ymax = minimum(yvals), maximum(yvals)
            pad = max((ymax - ymin) * 0.08, 1e-8)
            ymin -= pad
            ymax += pad
            x0, y0 = left, 28 + (pi - 1) * panel_h + top
            pw, ph = width - left - right, panel_h - top - bottom
            sx(i) = x0 + (i - 1) / max(length(x) - 1, 1) * pw
            sy(v) = y0 + ph - (v - ymin) / (ymax - ymin) * ph
            println(io, """<text x="$left" y="$(y0 - 14)" font-family="Arial" font-size="15" font-weight="700">$(svg_escape(p[:title]))</text>""")
            println(io, """<line x1="$x0" y1="$(y0 + ph)" x2="$(x0 + pw)" y2="$(y0 + ph)" stroke="#555"/>""")
            println(io, """<line x1="$x0" y1="$y0" x2="$x0" y2="$(y0 + ph)" stroke="#555"/>""")
            println(io, """<text x="$(x0 - 8)" y="$(y0 + 4)" text-anchor="end" font-family="Arial" font-size="11">$(fmt(ymax))</text>""")
            println(io, """<text x="$(x0 - 8)" y="$(y0 + ph)" text-anchor="end" font-family="Arial" font-size="11">$(fmt(ymin))</text>""")
            println(io, """<text x="$x0" y="$(y0 + ph + 20)" font-family="Arial" font-size="11">$(first(x))</text>""")
            println(io, """<text x="$(x0 + pw)" y="$(y0 + ph + 20)" text-anchor="end" font-family="Arial" font-size="11">$(last(x))</text>""")
            for (j, y) in enumerate(ys)
                pts = join(["$(sx(i)),$(sy(y[i]))" for i in eachindex(y) if isfinite(y[i])], " ")
                isempty(pts) && continue
                println(io, """<polyline points="$pts" fill="none" stroke="$(colors[mod1(j, length(colors))])" stroke-width="$(j == 1 ? 1.6 : 1.2)" opacity="$(j == 1 ? 0.95 : 0.75)"/>""")
                println(io, """<text x="$(x0 + pw - 92)" y="$(y0 + 14 * j)" font-family="Arial" font-size="11" fill="$(colors[mod1(j, length(colors))])">$(svg_escape(labels[j]))</text>""")
            end
        end
        println(io, "</svg>")
    end
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

    xi = Expectiles.expectile(resid, TAU)
    v_xi = try
        Expectiles.expectile_var(ywin, fit, TAU)
    catch err
        println("Residual expectile variance failed for ", asset.label, " on ", asset.dates[i], ": ", err)
        NaN
    end
    xi_se = isfinite(v_xi) && v_xi >= 0 ? sqrt(v_xi / WINDOW) : NaN

    ce = sigma_next * xi
    v_ce = try
        Expectiles.conditional_expectile_var(ywin, fit, TAU)
    catch err
        println("Conditional expectile variance failed for ", asset.label, " on ", asset.dates[i], ": ", err)
        NaN
    end
    ce_se = isfinite(v_ce) && v_ce >= 0 ? sqrt(v_ce / WINDOW) : NaN

    q_alpha = empirical_quantile(resid, ALPHA)
    var_alpha = sigma_next * q_alpha
    es_alpha = sigma_next * downside_es(resid, ALPHA)

    tau_alpha = tau_for_quantile(resid, q_alpha)
    omega_alpha = isfinite(tau_alpha) && tau_alpha > 0 ? 1 / tau_alpha - 1 : NaN
    alpha_tau = alpha_for_expectile(resid, xi)
    var_matched = ce
    es_matched = sigma_next * downside_es_at_threshold(resid, xi)

    return (
        ticker = asset.label,
        date = asset.dates[i],
        realized_return = y[i],
        sigma = sigma_next,
        xi = xi,
        xi_lo = xi - Z_CI * xi_se,
        xi_hi = xi + Z_CI * xi_se,
        ce = ce,
        ce_lo = ce - Z_CI * ce_se,
        ce_hi = ce + Z_CI * ce_se,
        var = var_alpha,
        es = es_alpha,
        tau_alpha = tau_alpha,
        omega_alpha = omega_alpha,
        alpha_tau = alpha_tau,
        var_matched = var_matched,
        es_matched = es_matched,
        exception_var = y[i] < var_alpha,
        exception_ce = y[i] < ce,
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
            r.ticker, string(r.date),
            fmt(r.realized_return), fmt(r.sigma),
            fmt(r.xi), fmt(r.xi_lo), fmt(r.xi_hi),
            fmt(r.ce), fmt(r.ce_lo), fmt(r.ce_hi),
            fmt(r.var), fmt(r.es),
            fmt(r.tau_alpha), fmt(r.omega_alpha), fmt(r.alpha_tau),
            fmt(r.var_matched), fmt(r.es_matched),
            string(r.exception_var), string(r.exception_ce),
        ])
    end
    return rows
end

function backtest_rows(forecasts)
    rows = Vector{Vector{String}}()
    for ticker in PLOT_ORDER
        fs = [r for r in forecasts if r.ticker == ticker]
        isempty(fs) && continue
        n = length(fs)
        observed_var = sum(r.exception_var ? 1 : 0 for r in fs)
        observed_ce = sum(r.exception_ce ? 1 : 0 for r in fs)
        rows_for_ticker = [
            [ticker, "VaR", fmt(n * ALPHA), string(observed_var), fmt(observed_var / n), fmt(mean([r.var - r.realized_return for r in fs]))],
            [ticker, "XP", fmt(n * TAU), string(observed_ce), fmt(observed_ce / n), fmt(mean([r.ce - r.realized_return for r in fs]))],
        ]
        append!(rows, rows_for_ticker)
    end
    return rows
end

function matched_gain_loss_rows(forecasts)
    rows = Vector{Vector{String}}()
    for ticker in PLOT_ORDER
        fs = [r for r in forecasts if r.ticker == ticker]
        isempty(fs) && continue
        n = length(fs)
        push!(rows, [
            ticker,
            string(n),
            fmt2(mean([100 * r.alpha_tau for r in fs])),
            fmt2(mean([r.ce - r.realized_return for r in fs])),
            fmt2(mean([r.var_matched - r.realized_return for r in fs])),
            fmt2(mean([r.es_matched - r.realized_return for r in fs])),
        ])
    end
    return rows
end

function write_matched_gain_loss_tex(path, rows)
    open(path, "w") do io
        println(io, "\\begin{table}[t]")
        println(io, "\\centering")
        println(io, "\\caption{Gain-loss-matched capital buffers}")
        println(io, "\\label{tab:app_matched_buffers}")
        println(io, "\\begin{tabular}{lrrrrr}")
        println(io, "\\toprule")
        println(io, "Asset & Windows & Avg. \\(\\alpha(\\tau)\\) & XP & Matched VaR & Matched ES \\\\")
        println(io, "\\midrule")
        for row in rows
            println(io, join(row, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\begin{flushleft}")
        println(io, "\\footnotesize")
        println(io, "Notes: The table fixes the expectile level at \\(\\tau=0.01\\), corresponding to ")
        println(io, "a gain-loss ratio \\(\\Omega=(1-\\tau)/\\tau=99\\). The matched VaR uses ")
        println(io, "the rolling empirical quantile level \\(\\alpha(\\tau)\\) such that ")
        println(io, "\\(\\widehat{VaR}_{\\alpha(\\tau)}=\\widehat{XP}_{\\tau}\\). The matched ES ")
        println(io, "is the empirical tail mean below that matched quantile. Buffer columns report ")
        println(io, "the average forecast minus realized return.")
        println(io, "\\end{flushleft}")
        println(io, "\\end{table}")
    end
end

function plot_prices(assets)
    panels = []
    for a in assets
        normalized = 100 .* a.prices ./ first(a.prices)
        push!(panels, Dict(:title => a.label, :x => a.dates, :ys => [normalized], :labels => ["price index"]))
    end
    line_svg(joinpath(PLOT_DIR, "price_paths.svg"), panels; title="Normalized adjusted prices")
end

function plot_drawdowns(assets)
    panels = []
    for a in assets
        keep = findall(d -> d >= Date(2018, 1, 1), a.dates)
        isempty(keep) && continue
        dd = drawdown(a.prices)
        push!(panels, Dict(:title => a.label, :x => a.dates[keep], :ys => [dd[keep]], :labels => ["drawdown (%)"]))
    end
    line_svg(joinpath(PLOT_DIR, "drawdowns.svg"), panels; title="Historical drawdowns since 2018")
end

function plot_forecast_panels(forecasts)
    for ticker in FOCUS_ORDER
        fs = [r for r in forecasts if r.ticker == ticker && r.date >= Date(2020, 1, 1)]
        isempty(fs) && continue
        panels = [
            Dict(
                :title => "$ticker risk forecasts",
                :x => [r.date for r in fs],
                :ys => [
                    [r.realized_return for r in fs],
                    [r.var for r in fs],
                    [r.es for r in fs],
                    [r.ce for r in fs],
                    [r.ce_lo for r in fs],
                    [r.ce_hi for r in fs],
                ],
                :labels => ["return", "VaR", "ES", "XP", "XP lower", "XP upper"],
            ),
        ]
        line_svg(joinpath(PLOT_DIR, "risk_forecasts_$(ticker).svg"), panels; title="$ticker one-day risk forecasts")
    end
end

function plot_mappings(forecasts)
    panels_omega = []
    panels_alpha = []
    for ticker in FOCUS_ORDER
        fs = [r for r in forecasts if r.ticker == ticker]
        isempty(fs) && continue
        push!(panels_omega, Dict(
            :title => ticker,
            :x => [r.date for r in fs],
            :ys => [[r.omega_alpha for r in fs]],
            :labels => ["omega at VaR"],
        ))
        push!(panels_alpha, Dict(
            :title => ticker,
            :x => [r.date for r in fs],
            :ys => [[100 * r.alpha_tau for r in fs]],
            :labels => ["alpha(tau) (%)"],
        ))
    end
    !isempty(panels_omega) && line_svg(joinpath(PLOT_DIR, "omega_at_var.svg"), panels_omega; title="Gain-loss ratio implied by VaR")
    !isempty(panels_alpha) && line_svg(joinpath(PLOT_DIR, "alpha_at_expectile.svg"), panels_alpha; title="Quantile level implied by fixed expectile")
end

function main()
    mkpath(PLOT_DIR)
    assets = read_assets(DATA_FILE)

    write_csv(
        joinpath(OUT_DIR, "descriptive_statistics.csv"),
        ["ticker", "T", "start", "end", "min", "q25", "median", "q75", "max", "mean", "std_dev", "skew", "kurtosis"],
        descriptive_rows(assets),
    )

    plot_prices(assets)
    plot_drawdowns(assets)

    forecasts = Any[]
    for label in vcat(CRYPTO_ORDER, ["SPX", "EUR"])
        asset = only([a for a in assets if a.label == label])
        println("Rolling GJR-GARCH forecasts for ", label, " on ", Threads.nthreads(), " Julia thread(s)...")
        append!(forecasts, rolling_forecasts(asset))
    end

    write_csv(
        joinpath(OUT_DIR, "rolling_risk_forecasts.csv"),
        ["ticker", "date", "realized_return", "sigma", "xi", "xi_lo", "xi_hi", "ce", "ce_lo", "ce_hi", "var", "es", "tau_alpha", "omega_alpha", "alpha_tau", "var_matched", "es_matched", "exception_var", "exception_ce"],
        forecast_rows(forecasts),
    )

    write_csv(
        joinpath(OUT_DIR, "risk_backtest_summary.csv"),
        ["ticker", "risk_measure", "expected_exceptions", "observed_exceptions", "observed_exception_rate", "avg_capital_buffer"],
        backtest_rows(forecasts),
    )

    matched_rows = matched_gain_loss_rows(forecasts)
    write_csv(
        joinpath(OUT_DIR, "matched_gain_loss_summary.csv"),
        ["ticker", "windows", "avg_alpha_tau_pct", "avg_xp_buffer", "avg_var_matched_buffer", "avg_es_matched_buffer"],
        matched_rows,
    )
    write_matched_gain_loss_tex(joinpath(OUT_DIR, "matched_gain_loss_summary.tex"), matched_rows)

    plot_forecast_panels(forecasts)
    plot_mappings(forecasts)

    println("Application outputs written to ", OUT_DIR)
    println("CLT interval level: ", CI_LEVEL, "; window: ", WINDOW, "; tau: ", TAU, "; alpha: ", ALPHA)
end

main()
