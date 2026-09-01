from pathlib import Path
import os
import runpy
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "results" / "mc" / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "mc" / "tables" / "mc_inference_summary.csv"
TABLE_DIR = ROOT / "results" / "mc" / "tables"
FIG_DIR = ROOT / "results" / "mc" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(DATA)
d["n"] = d["n"].astype(int)
models = ["GARCH", "GJR-GARCH"]
dists = ["normal", "t8", "t4"]
dist_tex = {"normal": "Normal", "t8": r"$t_8$", "t4": r"$t_4$"}
colors = {"low": "#0072B2", "high": "#D55E00"}
markers = {"low": "o", "high": "s"}


def coverage_figure(target, filename, ylabel):
    coverage = f"{target}_coverage"
    zsd = f"{target}_z_sd"
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True, sharey="row")
    for i, model in enumerate(models):
        for j, dist in enumerate(dists):
            ax = axes[i, j]
            sub = d[(d.model == model) & (d.distribution == dist)]
            for persistence in ["low", "high"]:
                s = sub[sub.persistence == persistence].sort_values("n")
                ax.plot(range(len(s)), s[coverage], color=colors[persistence], marker=markers[persistence],
                        linewidth=1.5, markersize=4.5, label=persistence.capitalize())
            ax.axhline(0.95, color="0.25", linestyle="--", linewidth=0.9)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["500", "1,000", "2,500", "5,000"])
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.set_title(f"{model}; {dist_tex[dist]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("Sample size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / filename, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / filename.replace(".pdf", ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True, sharey="row")
    for i, model in enumerate(models):
        for j, dist in enumerate(dists):
            ax = axes[i, j]
            sub = d[(d.model == model) & (d.distribution == dist)]
            for persistence in ["low", "high"]:
                s = sub[sub.persistence == persistence].sort_values("n")
                ax.plot(range(len(s)), s[zsd], color=colors[persistence], marker=markers[persistence],
                        linewidth=1.5, markersize=4.5, label=persistence.capitalize())
            ax.axhline(1.0, color="0.25", linestyle="--", linewidth=0.9)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["500", "1,000", "2,500", "5,000"])
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.set_title(f"{model}; {dist_tex[dist]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Studentized standard deviation")
            if i == 1:
                ax.set_xlabel("Sample size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    sd_name = filename.replace("coverage", "studentized_sd")
    fig.savefig(FIG_DIR / sd_name, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / sd_name.replace(".pdf", ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main_table(target, filename, caption, label):
    coverage = f"{target}_coverage"
    coverage_se = f"{target}_coverage_mcse"
    zsd = f"{target}_z_sd"
    zsd_se = f"{target}_z_sd_mcse"
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lll" + "cc" * 4 + "}",
        r"\toprule",
        r"&&&\multicolumn{2}{c}{$n=500$}&\multicolumn{2}{c}{$n=1{,}000$}&\multicolumn{2}{c}{$n=2{,}500$}&\multicolumn{2}{c}{$n=5{,}000$}\\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"Model&Innov.&Pers.&Cov.&SD&Cov.&SD&Cov.&SD&Cov.&SD\\",
        r"\midrule",
    ]
    for model in models:
        first_model = True
        for dist in dists:
            for persistence in ["low", "high"]:
                s = d[(d.model == model) & (d.distribution == dist) &
                      (d.persistence == persistence)].sort_values("n")
                model_cell = model if first_model else ""
                dist_cell = dist_tex[dist] if persistence == "low" else ""
                vals = []
                for _, row in s.iterrows():
                    vals.extend([
                        rf"\shortstack{{{row[coverage]:.3f}\\[-1pt]\scriptsize({row[coverage_se]:.3f})}}",
                        rf"\shortstack{{{row[zsd]:.3f}\\[-1pt]\scriptsize({row[zsd_se]:.3f})}}",
                    ])
                lines.append(" & ".join([model_cell, dist_cell, persistence.capitalize(), *vals]) + r"\\")
                first_model = False
            lines.append(r"\addlinespace[1pt]")
        lines.append(r"\midrule" if model != models[-1] else r"\bottomrule")
    lines += [
        r"\end{tabular}",
        r"\begin{minipage}{0.98\textwidth}\footnotesize\vspace{2pt}",
        r"\textit{Notes:} Cov. is empirical coverage of the nominal 95\% Wald interval. "
        r"SD is the Monte Carlo standard deviation of the corresponding studentized statistic. "
        r"Monte Carlo standard errors are reported in parentheses. Coverage standard errors use "
        r"the binomial formula; SD standard errors use the delta method based on the empirical fourth "
        r"central moment. Each design uses 10,000 replications.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    (TABLE_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def detailed_xi_table(filename):
    """Export the full-grid innovation-expectile results as an appendix longtable."""
    header = [
        r"\toprule",
        r"Model & Innov. & Pers. & $n$ & Bias & RMSE & MC SD & Avg. SE & SE/SD \\",
        r"\midrule",
    ]
    lines = [
        r"\begin{longtable}{lllrrrrrr}",
        r"\caption{Detailed inference results for the innovation expectile $\xi_0$.}",
        r"\label{tab:mc-xi-detailed}\\",
        *header,
        r"\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ -- continued from previous page}\\",
        *header,
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for model_index, model in enumerate(models):
        first_model = True
        for dist in dists:
            first_dist = True
            for persistence in ["low", "high"]:
                s = d[(d.model == model) & (d.distribution == dist) &
                      (d.persistence == persistence)].sort_values("n")
                for _, row in s.iterrows():
                    lines.append(" & ".join([
                        model if first_model else "",
                        dist_tex[dist] if first_dist else "",
                        persistence.capitalize(),
                        f"{int(row['n']):,}",
                        f"{row['xi_bias']:.4f}",
                        f"{row['xi_rmse']:.4f}",
                        f"{row['xi_mc_sd']:.4f}",
                        f"{row['xi_avg_se']:.4f}",
                        f"{row['xi_se_sd_ratio']:.4f}",
                    ]) + r"\\")
                    first_model = False
                    first_dist = False
            lines.append(r"\addlinespace[1pt]")
        if model_index < len(models) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\addlinespace[2pt]",
        r"\multicolumn{9}{p{0.96\textwidth}}{\footnotesize \textit{Notes:} "
        r"Bias, RMSE, and MC SD are computed across all retained replications. Avg. SE is "
        r"the average estimated asymptotic standard error among replications with a finite, "
        r"strictly positive variance estimate. SE/SD is Avg. SE divided by MC SD. All quantities "
        r"are reported on the innovation scale. Each design uses 10,000 replications.}\\",
        r"\end{longtable}",
    ]
    (TABLE_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


coverage_figure("xi", "mc_xi_coverage.pdf", r"Coverage for $\xi_0$")
coverage_figure("ce", "mc_ce_coverage.pdf", r"Coverage for $c_{n+1}$")
main_table("xi", "mc_xi_main.tex", r"Inference for the innovation expectile $\xi_0$.", "tab:mc-xi")
main_table("ce", "mc_ce_main.tex", r"Inference for the one-step-ahead conditional expectile $c_{n+1}$.", "tab:mc-ce")
runpy.run_path(str(Path(__file__).with_name("make_detailed_xi_table.py")))
print("Generated Monte Carlo figures and LaTeX tables.")
