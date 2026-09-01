using DelimitedFiles
using Printf
using Statistics

# Recreates the three paper tables in the application prompt.
# Run after applications_crypto.jl and application_backtests.jl.

const ROOT = abspath(joinpath(@__DIR__, ".."))
const APP = joinpath(ROOT, "output", "applications")
const OUT = joinpath(APP, "tables")
const ASSETS = ["BNB", "BTC", "ETH", "EUR"]

fmt2(x) = isfinite(x) ? @sprintf("%.2f", x) : "--"
fmtint(x) = isfinite(x) ? string(round(Int, x)) : "--"
fmtpct(x) = isfinite(x) ? @sprintf("%.2f\\%%", 100x) : "--"

function readcsv(path)
    raw, header = readdlm(path, ',', String; header=true)
    raw, Dict(String(h) => i for (i, h) in enumerate(vec(header)))
end
function num(raw, index, row, field)
    text = strip(raw[row, index[field]])
    isempty(text) ? NaN : parse(Float64, text)
end
function rows(raw, index, asset)
    found = [r for r in axes(raw, 1) if raw[r, index["ticker"]] == asset]
    isempty(found) && error("No observations for $(asset).")
    found
end
function finite_stat(f, values)
    usable = filter(isfinite, collect(values))
    isempty(usable) ? NaN : f(usable)
end
latexrow(io, cells) = println(io, join(cells, " & "), " ", '\\', '\\')

function table1()
    raw, i = readcsv(joinpath(APP, "descriptive_statistics.csv"))
    open(joinpath(OUT, "table_1_descriptive_statistics.tex"), "w") do io
        println(io, "\\begin{table}[!ht]")
        println(io, raw"\caption{Descriptive statistics for the daily log-changes in exchange rates\\ {\footnotesize We report the number of observations (\#obs) for each time series, as well as their main summary statistics. In particular, we display not only their minimum (min), average (mean) and maximum (max) values, but also their standard deviation (std dev), skewness (skew), kurtosis (kurt), and quartiles ($q_{0.25}$, median and $q_{0.75}$). The sample ranges from January 2016 to December 2023 for EUR and BTC, and from November 2017 to December 2023 for BNB and ETH.}}")
        println(io, "\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{p{1.85cm}crrrrrrrrr}\n\\hline")
        println(io, raw"currency & \#obs & min & $q_{0.25}$ & median & $q_{0.75}$ & max & mean & std dev & skew & kurt\\")
        println(io, "\\hline")
        for asset in ASSETS
            r = only(rows(raw, i, asset)); count = replace(raw[r, i["T"]], r"(?<=\d)(?=(\d{3})+$)" => ",")
            cells = [asset, count, fmt2(num(raw,i,r,"min")), fmt2(num(raw,i,r,"q25")), fmt2(num(raw,i,r,"median")), fmt2(num(raw,i,r,"q75")), fmt2(num(raw,i,r,"max")), fmt2(num(raw,i,r,"mean")), fmt2(num(raw,i,r,"std_dev")), fmt2(num(raw,i,r,"skew")), fmt2(num(raw,i,r,"kurtosis"))]
            latexrow(io, cells)
        end
        println(io, "\\hline\n\\end{tabular}\\end{adjustbox}\n\\label{tab:summary}\n\\end{table}")
    end
end

function table2()
    raw, i = readcsv(joinpath(APP, "rolling_risk_forecasts.csv"))
    required = ("tau_alpha", "xp_fixed", "xp_fixed_lo", "xp_fixed_hi", "es_01", "es_01_lo", "es_01_hi", "ci_level")
    all(haskey(i, field) for field in required) || error("Forecast file is stale. Re-run applications_crypto.jl.")
    levels = unique(num(raw, i, r, "ci_level") for r in axes(raw,1))
    length(levels) == 1 && isapprox(only(levels), .95; atol=1e-10) || error("Table 2 requires 95% CLT intervals.")
    open(joinpath(OUT, "table_2_prediction_intervals_capital_buffers.tex"), "w") do io
        println(io, "\\begin{table}[!htb]\n\\centering")
        println(io, raw"\caption{Prediction intervals and capital buffers of the tail risk measures at $\alpha=\tau=0.01$\\ {\footnotesize We report the number of rolling windows (\#windows) we deploy to produce out-of-sample forecasts of the tail risk measures with $\tau=1\%$ and $\alpha=0.01$, as well as the average $\bar\tau(\alpha)$ for which daily value at risk and expectile coincide. We also display not only the average difference between capital requirements and realized losses (capital buffer), but also the median width of the 95\% prediction intervals of the one-step-ahead forecasts of each tail risk measure (interval width).}}")
        println(io, "\\label{tab:app_clt_mapping}\n\\begin{tabular}{lccrcccrccc}\n\\toprule")
        println(io, " & & && \\multicolumn{3}{c}{capital buffer} && \\multicolumn{3}{c}{interval width}", '\\', '\\')
        println(io, "\\cline{5-7}\\cline{9-11}")
        println(io, raw"currency & \#windows & $\bar\tau(\alpha)=\tau(0.01)$ && VaR & ES & XP && VaR & ES & XP", '\\', '\\')
        println(io, "\\midrule")
        for asset in ASSETS
            rs = rows(raw,i,asset)
            meanfield(f) = finite_stat(mean, (num(raw,i,r,"realized_return") - num(raw,i,r,f) for r in rs))
            width(lo,hi) = finite_stat(median, (num(raw,i,r,hi) - num(raw,i,r,lo) for r in rs))
            cells = [asset, fmtint(length(rs)), fmtpct(finite_stat(mean,(num(raw,i,r,"tau_alpha") for r in rs))), "", fmt2(meanfield("var")), fmt2(meanfield("es_01")), fmt2(meanfield("xp_fixed")), "", fmt2(width("var_lo","var_hi")), fmt2(width("es_01_lo","es_01_hi")), fmt2(width("xp_fixed_lo","xp_fixed_hi"))]
            latexrow(io, cells)
        end
        println(io, "\\bottomrule\n\\end{tabular}\\end{table}")
    end
end

function table3()
    raw, i = readcsv(joinpath(APP, "backtests", "fixed_level_var_xp_backtests.csv"))
    need = ("var_expected_exceptions","var_observed_exceptions","var_exception_rate","xp_expected_exceptions","xp_observed_exceptions","xp_exception_rate","kupiec_pvalue","christoffersen_conditional_coverage_pvalue","duration_pvalue","xp_identification_pvalue")
    all(haskey(i, field) for field in need) || error("Backtests are stale. Re-run application_backtests.jl.")
    r = Dict(raw[row,i["ticker"]] => row for row in axes(raw,1))
    all(haskey(r,a) for a in ASSETS) || error("Backtests must contain BNB, BTC, ETH, and EUR.")
    function paired(label, a, b, form)
        cells = String[label]
        for (n, asset) in enumerate(ASSETS)
            append!(cells, [form(num(raw,i,r[asset],a)), form(num(raw,i,r[asset],b))])
            n < length(ASSETS) && push!(cells, "")
        end
        cells
    end
    function varonly(label, field)
        cells = String[label]
        for (n, asset) in enumerate(ASSETS)
            append!(cells, [fmt2(num(raw,i,r[asset],field)), ""])
            n < length(ASSETS) && push!(cells, "")
        end
        cells
    end
    function xponly(label, field)
        cells = String[label]
        for (n, asset) in enumerate(ASSETS)
            append!(cells, ["", fmt2(num(raw,i,r[asset],field))])
            n < length(ASSETS) && push!(cells, "")
        end
        cells
    end
    open(joinpath(OUT, "table_3_fixed_level_backtests.tex"), "w") do io
        println(io, "\\begin{table}[!ht]")
        println(io, raw"\vspace*{1em}\centering\caption{Frequency of value at risk and expectile forecast exceedances at $\alpha=\tau=0.01$\\ {\footnotesize We report the expected and realized number of exceedances for the one-step-ahead forecasts of the value at risk and expectile based on rolling windows of 1,000 observations. Exceedances occur if realized loss exceeds the tail risk measure. Apart from the corresponding relative frequency (i.e., number of realized exceedances over number of out-of-sample forecasts), we also document the p-values of the coverage tests for the value at risk and the normal-approximation p-value of the residual test for the expectile.}}")
        println(io, "\\label{tab:app_backtest}\n\\begin{tabular}{lccrccrccrcc}\n\\toprule")
        println(io, " & \\multicolumn{2}{c}{BNB} && \\multicolumn{2}{c}{BTC} && \\multicolumn{2}{c}{ETH} && \\multicolumn{2}{c}{EUR}", '\\', '\\')
        println(io, "\\cline{2-3}\\cline{5-6}\\cline{8-9}\\cline{11-12}")
        println(io, "daily exceedances & VaR & XP && VaR & XP && VaR & XP && VaR & XP", '\\', '\\')
        println(io, "\\midrule")
        latexrow(io, paired("expected","var_expected_exceptions","xp_expected_exceptions",fmt2))
        latexrow(io, paired("realized","var_observed_exceptions","xp_observed_exceptions",fmtint))
        latexrow(io, paired("relative frequency","var_exception_rate","xp_exception_rate",fmtpct))
        println(io, "coverage tests\\\\")
        latexrow(io, varonly("~~~~unconditional","kupiec_pvalue"))
        latexrow(io, varonly("~~~~conditional","christoffersen_conditional_coverage_pvalue"))
        latexrow(io, varonly("~~~~duration","duration_pvalue"))
        latexrow(io, xponly("residual test","xp_identification_pvalue"))
        println(io, "\\bottomrule\n\\end{tabular}\\end{table}")
    end
end

function main()
    mkpath(OUT); table1(); table2(); table3()
    println("Recreated Tables 1--3 in ", OUT)
end
main()
