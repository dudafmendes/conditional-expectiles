"""Generate a portrait innovation-expectile table from the full Monte Carlo CSV."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "mc" / "tables" / "mc_inference_summary.csv"
OUTPUT = ROOT / "results" / "mc" / "tables" / "mc_innovation_expectile_portrait.tex"

MODELS = ["GARCH", "GJR-GARCH"]
DISTRIBUTIONS = ["normal", "t8", "t4"]
DIST_TEX = {"normal": "Normal", "t8": r"$t_8$", "t4": r"$t_4$"}
PERSISTENCE = ["low", "high"]
SAMPLE_SIZES = [500, 1000, 2500, 5000]
STATISTICS = ["xi_bias", "xi_rmse", "xi_mc_sd", "xi_avg_se", "xi_se_sd_ratio"]


with INPUT.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

lookup = {
    (r["model"], r["distribution"], r["persistence"], int(float(r["n"]))): r
    for r in rows
}

lines = [
    r"\begin{table}[!p]",
    r"\centering",
    r"\caption{Detailed innovation-expectile results by volatility specification.}",
    r"\label{tab:mc-xi-detailed}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.1pt}",
    r"\renewcommand{\arraystretch}{1.02}",
    r"\begin{tabular}{llr*{10}{r}}",
    r"\toprule",
    r"&&&\multicolumn{5}{c}{GARCH}&\multicolumn{5}{c}{GJR-GARCH}\\",
    r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}",
    r"&&&\multicolumn{3}{c}{Sampling performance}&\multicolumn{2}{c}{SE calibration}"
    r"&\multicolumn{3}{c}{Sampling performance}&\multicolumn{2}{c}{SE calibration}\\",
    r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}\cmidrule(lr){12-13}",
    r"Innov. & Pers. & $n$ & Bias & RMSE & MC SD & Avg. SE & SE/SD"
    r" & Bias & RMSE & MC SD & Avg. SE & SE/SD\\",
    r"\midrule",
]

for distribution_index, distribution in enumerate(DISTRIBUTIONS):
    first_distribution = True
    for persistence_index, persistence in enumerate(PERSISTENCE):
        first_persistence = True
        for n in SAMPLE_SIZES:
            identifiers = [
                DIST_TEX[distribution] if first_distribution else "",
                persistence.capitalize() if first_persistence else "",
                f"{n:,}",
            ]
            values = []
            for model in MODELS:
                row = lookup[(model, distribution, persistence, n)]
                values.extend(f"{float(row[column]):.4f}" for column in STATISTICS)
            lines.append(" & ".join(identifiers + values) + r"\\")
            first_distribution = False
            first_persistence = False
        if persistence_index == 0:
            lines.append(r"\addlinespace[1pt]")
    if distribution_index < len(DISTRIBUTIONS) - 1:
        lines.append(r"\addlinespace[3pt]")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{minipage}{0.98\textwidth}\footnotesize\vspace{4pt}",
    r"\textit{Notes:} Results concern the innovation-expectile estimator $\widehat\xi_n$ "
    r"at $\tau=0.95$. For replication $r$, define $e_r=\widehat\xi_{n,r}-\xi_0$, with "
    r"$R=10{,}000$ replications. Bias is $R^{-1}\sum_{r=1}^R e_r$; RMSE is "
    r"$\{R^{-1}\sum_{r=1}^R e_r^2\}^{1/2}$; and MC SD is the sample standard deviation "
    r"of $\widehat\xi_{n,r}$. Avg. SE is the average plug-in Wald standard error, "
    r"$\{\widehat V_{\xi,r}/n\}^{1/2}$, among replications for which "
    r"$\widehat V_{\xi,r}$ is finite and strictly positive. SE/SD is Avg. SE divided by "
    r"MC SD; values near one indicate agreement between the estimated standard errors and "
    r"the Monte Carlo sampling dispersion. Low and high persistence correspond to "
    r"$p_0=0.90$ and $p_0=0.98$, respectively. Student-$t$ innovations are standardized "
    r"to unit variance. All statistics are reported on the innovation scale.",
    r"\end{minipage}",
    r"\end{table}",
]

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Generated {OUTPUT}")
