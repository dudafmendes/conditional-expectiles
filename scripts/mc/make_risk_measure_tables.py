"""Build publication-format tables for unconditional and conditional expectiles."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "mc" / "tables"
RAW = ROOT / "results" / "mc" / "raw" / "mc_raw_replications.csv"

MODELS = ["GARCH", "GJR-GARCH"]
DISTRIBUTIONS = ["normal", "t8", "t4"]
DIST_TEX = {"normal": "Normal", "t8": r"$t_8$", "t4": r"$t_4$"}
PERSISTENCE = ["low", "high"]
SAMPLE_SIZES = [500, 1000, 2500, 5000]


def empirical_moments(values):
    center = statistics.fmean(values)
    deviations = [value - center for value in values]
    m2 = statistics.fmean(value**2 for value in deviations)
    skewness = statistics.fmean(value**3 for value in deviations) / m2**1.5
    excess_kurtosis = statistics.fmean(value**4 for value in deviations) / m2**2 - 3.0
    return statistics.stdev(values), skewness, excess_kurtosis


def aggregate_raw_csv():
    groups = defaultdict(lambda: {"xi_error": [], "xi_z": [], "ce_error": [], "ce_z": []})
    with RAW.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["distribution"], row["persistence"], int(row["n"]))
            xi_error = float(row["xi_hat"]) - float(row["true_xi"])
            ce_error = float(row["ce_hat"]) - float(row["ce_true"])
            groups[key]["xi_error"].append(xi_error)
            groups[key]["ce_error"].append(ce_error)
            for target in ("xi", "ce"):
                variance = float(row[f"v_{target}"])
                z_value = float(row[f"{target}_z"])
                if variance > 0 and math.isfinite(variance) and math.isfinite(z_value):
                    groups[key][f"{target}_z"].append(z_value)

    results = {}
    for key, values in groups.items():
        result = {}
        for target in ("xi", "ce"):
            errors = values[f"{target}_error"]
            z_values = values[f"{target}_z"]
            z_sd, z_skew, z_kurt = empirical_moments(z_values)
            result[target] = {
                "bias": statistics.fmean(errors),
                "rmse": math.sqrt(statistics.fmean(error**2 for error in errors)),
                "z_sd": z_sd,
                "z_skew": z_skew,
                "z_excess_kurtosis": z_kurt,
                "coverage": statistics.fmean(abs(value) <= 1.96 for value in z_values),
                "valid": len(z_values),
                "total": len(errors),
            }
        results[key] = result
    return results


def make_table(target, results):
    is_xi = target == "xi"
    output = TABLE_DIR / (
        "mc_unconditional_expectile_wald.tex" if is_xi
        else "mc_conditional_expectile_wald.tex"
    )
    caption = (
        "Point accuracy and Wald-statistic diagnostics for the unconditional innovation expectile."
        if is_xi else
        "Point accuracy and Wald-statistic diagnostics for the one-step-ahead conditional expectile."
    )
    label = "tab:mc-xi-wald-diagnostics" if is_xi else "tab:mc-ce-wald-diagnostics"
    z_symbol = r"Z_{\xi,n}" if is_xi else r"Z_{c,n+1}"

    lines = [
        r"\begin{sidewaystable}[!p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{tabular}{llr*{12}{r}}",
        r"\toprule",
        r"&&&\multicolumn{6}{c}{GARCH}&\multicolumn{6}{c}{GJR-GARCH}\\",
        r"\cmidrule(lr){4-9}\cmidrule(lr){10-15}",
        r"&&&\multicolumn{2}{c}{Point accuracy}&\multicolumn{4}{c}{Wald statistic}"
        r"&\multicolumn{2}{c}{Point accuracy}&\multicolumn{4}{c}{Wald statistic}\\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}\cmidrule(lr){12-15}",
        rf"Innov. & Pers. & $n$ & Bias & RMSE & SD & Skew. & Ex. kurt. & Cov."
        rf" & Bias & RMSE & SD & Skew. & Ex. kurt. & Cov.\\",
        r"\midrule",
    ]

    for dist_index, distribution in enumerate(DISTRIBUTIONS):
        first_distribution = True
        for pers_index, persistence in enumerate(PERSISTENCE):
            first_persistence = True
            for n in SAMPLE_SIZES:
                ids = [
                    DIST_TEX[distribution] if first_distribution else "",
                    persistence.capitalize() if first_persistence else "",
                    f"{n:,}",
                ]
                values = []
                for model in MODELS:
                    key = (model, distribution, persistence, n)
                    values_for_model = results[key][target]
                    values.extend([
                        f"{values_for_model['bias']:.4f}",
                        f"{values_for_model['rmse']:.4f}",
                        f"{values_for_model['z_sd']:.3f}",
                        f"{values_for_model['z_skew']:.3f}",
                        f"{values_for_model['z_excess_kurtosis']:.3f}",
                        f"{values_for_model['coverage']:.3f}",
                    ])
                lines.append(" & ".join(ids + values) + r"\\")
                first_distribution = False
                first_persistence = False
            if pers_index == 0:
                lines.append(r"\addlinespace[1pt]")
        if dist_index < len(DISTRIBUTIONS) - 1:
            lines.append(r"\addlinespace[3pt]")

    if is_xi:
        definition = (
            r"The target is the innovation expectile $\xi_0\equiv\xp^\eta$ and the estimator is "
            r"$\widehat\xi_n\equiv\xpb{\widehat{\bs\theta}_n}$. For replication $r$, define "
            r"$e_r=\widehat\xi_{n,r}-\xi_0$. The corresponding studentized statistic is "
            r"$Z_{\xi,n,r}=\sqrt n\,e_r/\sqrt{\widehat V_{\xi,r}}$."
        )
    else:
        definition = (
            r"The target is $c_{n+1}=\sigma_{n+1}(\bs\theta_0)\xi_0$ and its estimator is "
            r"$\widehat c_{n+1}=\widetilde\sigma_{n+1}(\widehat{\bs\theta}_n)\widehat\xi_n$. "
            r"For replication $r$, define $e_r=\widehat c_{n+1,r}-c_{n+1,r}$. The corresponding "
            r"studentized statistic is $Z_{c,n+1,r}=\sqrt n\,e_r/\sqrt{\widehat V_{n+1,r}}$."
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\textheight}\small\vspace{4pt}",
        r"\textit{Notes:} " + definition + " "
        r"The number of Monte Carlo replications is $R=10{,}000$. Bias is "
        r"$R^{-1}\sum_{r=1}^{R}e_r$, and RMSE is "
        r"$\{R^{-1}\sum_{r=1}^{R}e_r^2\}^{1/2}$. SD, Skew., and Ex. kurt. are, "
        rf"respectively, the sample standard deviation and the empirical standardized third "
        rf"and fourth central moments of ${z_symbol}$; 3 is subtracted from the fourth moment. "
        r"The standard Normal reference values are therefore 1, 0, and 0. Cov. is the fraction "
        rf"of replications satisfying $|{z_symbol}|\leq 1.96$, equivalently the empirical coverage "
        r"of the nominal 95\% two-sided Wald interval. Wald-statistic moments and coverage are "
        r"computed over replications with a finite, strictly positive estimated asymptotic variance; "
        r"all 10,000 replications satisfy this requirement in every reported design. The expectile "
        r"level is $\tau=0.95$. Low and high persistence correspond to $p_0=0.90$ and $p_0=0.98$, "
        r"respectively. Student-$t$ innovations are standardized to unit variance. Bias and RMSE "
        r"are reported in the units of the corresponding risk measure.",
        r"\end{minipage}",
        r"\end{sidewaystable}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {output}")


def make_portrait_table(target, results):
    """Create the compact portrait counterpart without higher-moment diagnostics."""
    is_xi = target == "xi"
    output = TABLE_DIR / (
        "mc_unconditional_expectile_wald_portrait.tex" if is_xi
        else "mc_conditional_expectile_wald_portrait.tex"
    )
    caption = (
        "Point accuracy and Wald inference for the unconditional innovation expectile."
        if is_xi else
        "Point accuracy and Wald inference for the one-step-ahead conditional expectile."
    )
    label = "tab:mc-xi-wald-portrait" if is_xi else "tab:mc-ce-wald-portrait"
    z_symbol = r"Z_{\xi,n}" if is_xi else r"Z_{c,n+1}"

    lines = [
        r"\begin{table}[!p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabular}{llr*{8}{r}}",
        r"\toprule",
        r"&&&\multicolumn{4}{c}{GARCH}&\multicolumn{4}{c}{GJR-GARCH}\\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        r"Innov. & Pers. & $n$ & Bias & RMSE & Wald SD & Cov."
        r" & Bias & RMSE & Wald SD & Cov.\\",
        r"\midrule",
    ]

    for dist_index, distribution in enumerate(DISTRIBUTIONS):
        first_distribution = True
        for pers_index, persistence in enumerate(PERSISTENCE):
            first_persistence = True
            for n in SAMPLE_SIZES:
                identifiers = [
                    DIST_TEX[distribution] if first_distribution else "",
                    persistence.capitalize() if first_persistence else "",
                    f"{n:,}",
                ]
                values = []
                for model in MODELS:
                    result = results[(model, distribution, persistence, n)][target]
                    values.extend([
                        f"{result['bias']:.4f}",
                        f"{result['rmse']:.4f}",
                        f"{result['z_sd']:.3f}",
                        f"{result['coverage']:.3f}",
                    ])
                lines.append(" & ".join(identifiers + values) + r"\\")
                first_distribution = False
                first_persistence = False
            if pers_index == 0:
                lines.append(r"\addlinespace[1pt]")
        if dist_index < len(DISTRIBUTIONS) - 1:
            lines.append(r"\addlinespace[3pt]")

    if is_xi:
        definition = (
            r"The target is the innovation expectile $\xi_0\equiv\xp^\eta$, estimated by "
            r"$\widehat\xi_n\equiv\xpb{\widehat{\bs\theta}_n}$. For replication $r$, "
            r"$e_r=\widehat\xi_{n,r}-\xi_0$ and "
            r"$Z_{\xi,n,r}=\sqrt n\,e_r/\sqrt{\widehat V_{\xi,r}}$."
        )
    else:
        definition = (
            r"The target is $c_{n+1}=\sigma_{n+1}(\bs\theta_0)\xi_0$, estimated by "
            r"$\widehat c_{n+1}=\widetilde\sigma_{n+1}(\widehat{\bs\theta}_n)\widehat\xi_n$. "
            r"For replication $r$, $e_r=\widehat c_{n+1,r}-c_{n+1,r}$ and "
            r"$Z_{c,n+1,r}=\sqrt n\,e_r/\sqrt{\widehat V_{n+1,r}}$."
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\textwidth}\footnotesize\vspace{4pt}",
        r"\textit{Notes:} " + definition + " "
        r"The number of Monte Carlo replications is $R=10{,}000$. Bias is "
        r"$R^{-1}\sum_{r=1}^{R}e_r$, and RMSE is "
        r"$\{R^{-1}\sum_{r=1}^{R}e_r^2\}^{1/2}$. Wald SD is the sample standard deviation "
        rf"of ${z_symbol}$; its standard Normal benchmark is one. Cov. is the fraction of "
        rf"replications satisfying $|{z_symbol}|\leq 1.96$, equivalently the empirical coverage "
        r"of the nominal 95\% two-sided Wald interval. Wald SD and coverage are computed over "
        r"replications with a finite, strictly positive estimated asymptotic variance; all 10,000 "
        r"replications satisfy this requirement in every reported design. The expectile level is "
        r"$\tau=0.95$. Low and high persistence correspond to $p_0=0.90$ and $p_0=0.98$, "
        r"respectively. Student-$t$ innovations are standardized to unit variance. Bias and RMSE "
        r"are reported in the units of the corresponding risk measure. All entries are computed "
        r"directly from the replication-level Monte Carlo CSV.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {output}")


raw_results = aggregate_raw_csv()
if len(raw_results) != 48:
    raise ValueError(f"Expected 48 Monte Carlo designs, found {len(raw_results)}")
if any(v[t]["valid"] != v[t]["total"] for v in raw_results.values() for t in ("xi", "ce")):
    raise ValueError("At least one design has an invalid Wald statistic; revise the table note.")
for table_target in ("xi", "ce"):
    make_table(table_target, raw_results)
    make_portrait_table(table_target, raw_results)
