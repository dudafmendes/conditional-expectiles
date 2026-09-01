"""Create the joint Monte Carlo expectile tables and interval-length figure.

All reported quantities are recomputed from the replication-level joint CSV.
The script never reads the older summary tables or JLD2 result files.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "mc" / "raw" / "mc_joint_common_levels_replications.csv"
TABLE_DIR = ROOT / "results" / "mc" / "tables"
FIGURE_DIR = ROOT / "results" / "mc" / "figures"
SUMMARY = TABLE_DIR / "mc_common_levels_xp_summary.csv"

MODELS = ["GARCH", "GJR-GARCH"]
DISTRIBUTIONS = ["normal", "t8", "t4"]
PERSISTENCE = ["low", "high"]
SAMPLE_SIZES = [500, 1000, 2500, 5000]
DIST_TEX = {"normal": "normal", "t8": r"$t_8$", "t4": r"$t_4$"}

TARGETS = {
    "tau_0p01": {
        "tau_text": r"$\tau=0.01$",
        "tau_note": r"The expectile level is $\tau=0.01$.",
    },
    "tau_0p05": {
        "tau_text": r"$\tau=0.05$",
        "tau_note": r"The expectile level is $\tau=0.05$.",
    },
}

LEVELS = [("0p01", 0.01), ("0p05", 0.05)]


def required_columns() -> list[str]:
    columns = ["design_id", "replication", "status", "model", "distribution", "persistence", "n"]
    for key in TARGETS:
        for scale in ("xi", "ce"):
            prefix = f"{scale}_{key}"
            columns.extend(
                [
                    f"{prefix}_truth",
                    f"{prefix}_estimate",
                    f"{prefix}_wald",
                    f"{prefix}_valid",
                ]
            )
    for token, _ in LEVELS:
        columns.extend(
            [
                f"ratio_ce_tau_{token}_to_var_alpha_{token}",
                f"ratio_ce_tau_{token}_to_es_delta_{token}",
            ]
        )
    return columns


def read_and_audit() -> pd.DataFrame:
    data = pd.read_csv(RAW, usecols=required_columns())
    if len(data) != 480_000:
        raise ValueError(f"Expected 480,000 rows, found {len(data):,}.")
    if set(data["status"]) != {"success"}:
        raise ValueError(f"The joint CSV contains non-success rows: {data['status'].value_counts().to_dict()}")
    if data.duplicated(["design_id", "replication"]).any():
        raise ValueError("Duplicate (design_id, replication) rows were found.")

    counts = data.groupby(["model", "distribution", "persistence", "n"], observed=True).size()
    if len(counts) != 48 or not (counts == 10_000).all():
        raise ValueError("The CSV does not contain 48 complete designs with 10,000 rows each.")
    return data


def monte_carlo_statistics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    error = (frame[f"{prefix}_estimate"] - frame[f"{prefix}_truth"]).to_numpy(dtype=float)
    if not np.isfinite(error).all():
        raise ValueError(f"Non-finite point-estimation error in {prefix}.")

    replications = len(error)
    squared_error = error**2
    bias = float(error.mean())
    rmse = float(np.sqrt(squared_error.mean()))
    bias_mcse = float(error.std(ddof=1) / np.sqrt(replications))
    rmse_mcse = float(squared_error.std(ddof=1) / (2.0 * rmse * np.sqrt(replications)))

    valid = frame[f"{prefix}_valid"].fillna(False).astype(bool).to_numpy(copy=True)
    wald_all = frame[f"{prefix}_wald"].to_numpy(dtype=float)
    valid &= np.isfinite(wald_all)
    wald = wald_all[valid]
    valid_count = len(wald)
    if valid_count < 2:
        raise ValueError(f"Fewer than two valid Wald statistics in {prefix}.")

    wald_sd = float(wald.std(ddof=1))
    centered = wald - wald.mean()
    second_moment = float(np.mean(centered**2))
    fourth_moment = float(np.mean(centered**4))
    wald_sd_mcse = math.sqrt(
        max(fourth_moment - second_moment**2, 0.0)
        / (4.0 * valid_count * second_moment)
    )
    coverage = float(np.mean(np.abs(wald) <= 1.96))
    coverage_mcse = math.sqrt(coverage * (1.0 - coverage) / valid_count)

    return {
        "replications": replications,
        "valid_wald": valid_count,
        "bias": bias,
        "bias_mcse": bias_mcse,
        "rmse": rmse,
        "rmse_mcse": rmse_mcse,
        "wald_sd": wald_sd,
        "wald_sd_mcse": wald_sd_mcse,
        "coverage": coverage,
        "coverage_mcse": coverage_mcse,
    }


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", "distribution", "persistence", "n"]
    for group_key, frame in data.groupby(group_columns, sort=False, observed=True):
        identifiers = dict(zip(group_columns, group_key))
        for tau_key in TARGETS:
            for scale in ("xi", "ce"):
                result = monte_carlo_statistics(frame, f"{scale}_{tau_key}")
                rows.append({**identifiers, "tau": tau_key, "scale": scale, **result})
    result = pd.DataFrame(rows)
    result.to_csv(SUMMARY, index=False, quoting=csv.QUOTE_MINIMAL)
    return result


def expectile_table(summary: pd.DataFrame, tau_key: str, scale: str) -> Path:
    is_unconditional = scale == "xi"
    adjective = "unconditional" if is_unconditional else "conditional"
    filename = f"mc_{adjective}_expectile_{tau_key}.tex"
    output = TABLE_DIR / filename
    base_label = "tab:mc-xi-wald-portrait" if is_unconditional else "tab:mc-ce-wald-portrait"
    label = base_label if tau_key == "tau_0p05" else f"{base_label}-{tau_key.replace('_', '-')}"
    z_symbol = r"Z_{\xi,n}" if is_unconditional else r"Z_{c,n+1}"
    tau_tex = {"tau_0p05": "0.05", "tau_0p01": "0.01"}[tau_key]

    if is_unconditional:
        caption = (
            r"Performance of the two-step estimator of the unconditional expectile\\ "
            r"{\footnotesize The target is the innovation expectile $\xi_0\equiv\xp^\eta$ "
            rf"for $\tau={tau_tex}$, which we estimate by "
            r"$\widehat\xi_n\equiv\xpb{\widehat{\bs\theta}_n}$. For each of the 10,000 "
            r"Monte Carlo replications, we report the bias and root mean squared error (RMSE) "
            r"of the expectile estimator for each experimental design. The latter considers "
            r"both standard GARCH and GJR-GARCH specifications, with different sample sizes "
            r"and persistent levels (low and high). The innovations come either from a Gaussian "
            r"distribution or from a $t$-student distribution (with 4 or 8 degrees of freedom). "
            r"In addition, we also report the standard deviation of the Wald statistic "
            rf"${z_symbol}$ in \eqref{{eq:mc-studentized}} across Monte Carlo replications in "
            rf"the columns SD(${z_symbol}$), as well as the empirical coverage of the nominal "
            r"95\% two-sided Wald interval. We compute the latter by the relative frequency at "
            rf"which ${z_symbol}$ does not exceed, in magnitude, 1.96 across replications.}}"
        )
    else:
        caption = (
            r"Performance of the two-step estimator of the conditional expectile\\ "
            r"{\footnotesize The target is the conditional expectile "
            r"$c_{n+1}\equiv\sigma_{n+1}\xi_0$ "
            rf"for $\tau={tau_tex}$, which we estimate by "
            r"$\widehat{c}_{n+1}\equiv\widetilde\sigma_{n+1}(\widehat{\bs\theta}_n)"
            r"\widehat{\xi}_n$. For each of the 10,000 Monte Carlo replications, we report "
            r"the bias and root mean squared error (RMSE) of the expectile estimator for each "
            r"experimental design. The latter considers both standard GARCH and GJR-GARCH "
            r"specifications, with different sample sizes and persistent levels (low and high). "
            r"The innovations come either from a Gaussian distribution or from a $t$-student "
            r"distribution (with 4 or 8 degrees of freedom). In addition, we also report the "
            rf"standard deviation of the Wald statistic ${z_symbol}$ in "
            r"\eqref{eq:mc-studentized} across Monte Carlo replications in the columns "
            rf"SD(${z_symbol}$), as well as the empirical coverage of the nominal 95\% "
            r"two-sided Wald interval. We compute the latter by the relative frequency at which "
            rf"${z_symbol}$ does not exceed, in magnitude, 1.96 across replications.}}"
        )

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{width=1\textwidth}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{llr*{9}{c}}",
        r"\toprule",
        r"&&&\multicolumn{4}{c}{GARCH}&&\multicolumn{4}{c}{GJR-GARCH}\\",
        r"\cline{4-7}\cline{9-12}",
        rf"distribution & persistence & sample size{'~~' if is_unconditional else ''} & bias & RMSE & SD(${z_symbol}$) & coverage "
        rf"&& bias & RMSE & SD(${z_symbol}$) & coverage\\",
        r"\midrule",
    ]

    selected = summary[(summary["tau"] == tau_key) & (summary["scale"] == scale)]
    for dist_index, distribution in enumerate(DISTRIBUTIONS):
        first_distribution = True
        for persistence_index, persistence in enumerate(PERSISTENCE):
            first_persistence = True
            for n in SAMPLE_SIZES:
                values: list[str] = []
                for model in MODELS:
                    cell = selected[
                        (selected["model"] == model)
                        & (selected["distribution"] == distribution)
                        & (selected["persistence"] == persistence)
                        & (selected["n"] == n)
                    ]
                    if len(cell) != 1:
                        raise ValueError(f"Expected one summary row, found {len(cell)} for {model}, {distribution}, {persistence}, {n}.")
                    row = cell.iloc[0]
                    values.extend(
                        [
                            f"{row['bias']:.4f}",
                            f"{row['rmse']:.4f}",
                            f"{row['wald_sd']:.3f}",
                            f"{row['coverage']:.3f}",
                        ]
                    )
                identifiers = [
                    DIST_TEX[distribution] if first_distribution else "",
                    persistence if first_persistence else "",
                    f"{n:,}{'~~' if is_unconditional else ''}",
                ]
                lines.append(
                    " & ".join(identifiers + values[:4])
                    + " && "
                    + " & ".join(values[4:])
                    + r"\\"
                )
                first_distribution = False
                first_persistence = False
            if persistence_index == 0:
                lines.append(r"\addlinespace[1pt]")
        if dist_index < len(DISTRIBUTIONS) - 1:
            lines.append(r"\addlinespace[3pt]")

    lines.extend(
        [
            r"\bottomrule",
            *([r"&\\"] if not is_unconditional else []),
            r"\end{tabular}",
            r"\end{adjustbox}\end{table}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def interval_ratio_box_statistics(data: pd.DataFrame) -> list[Path]:
    rows: list[dict[str, object]] = []
    for token, level in LEVELS:
        ratio_columns = {
            "VaR": f"ratio_ce_tau_{token}_to_var_alpha_{token}",
            "ES": f"ratio_ce_tau_{token}_to_es_delta_{token}",
        }
        for benchmark, ratio_column in ratio_columns.items():
            for (model, distribution, persistence, n), frame in data.groupby(
                ["model", "distribution", "persistence", "n"], observed=True
            ):
                values = frame[ratio_column].to_numpy(dtype=float)
                values = np.sort(values[np.isfinite(values) & (values > 0)])
                if len(values) == 0:
                    raise ValueError(f"No finite interval-length ratios for level={level}, {benchmark}, {model}, {distribution}, {persistence}, {n}.")
                q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
                iqr = q3 - q1
                lower_candidates = values[values >= q1 - 1.5 * iqr]
                upper_candidates = values[values <= q3 + 1.5 * iqr]
                rows.append(
                    {
                        "level": level,
                        "benchmark": benchmark,
                        "model": model,
                        "distribution": distribution,
                        "persistence": persistence,
                        "n": n,
                        "valid": len(values),
                        "lower_whisker": lower_candidates[0],
                        "q1": q1,
                        "median": median,
                        "q3": q3,
                        "upper_whisker": upper_candidates[-1],
                    }
                )

    stats = FIGURE_DIR / "mc_interval_length_ratio_boxstats.csv"
    pd.DataFrame(rows).to_csv(stats, index=False)

    outputs = [stats]
    for token, level in LEVELS:
        percent = int(round(100 * level))
        tex = FIGURE_DIR / f"mc_interval_length_ratios_{token}.tex"
        tex.write_text(
            "\n".join(
                [
                    r"\begin{figure}[!t]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth]{{../results/mc/figures/mc_interval_length_ratios_{token}.pdf}}",
                    rf"\caption{{Relative lengths of confidence intervals for conditional risk measures at {percent}\%. "
                    rf"Each box summarizes the replication-level ratio between the length of the nominal 95\% Wald interval "
                    rf"for the conditional expectile at $\tau={level:.2f}$ and the corresponding interval length for VaR at "
                    rf"$\alpha={level:.2f}$ (top row) or ES at $\delta={level:.2f}$ (bottom row). Thus, $\alpha=\delta=\tau$. "
                    r"The dashed reference line at one denotes equal interval lengths; values below one indicate a shorter "
                    r"expectile interval. Boxes show the interquartile range and median, whiskers extend to 1.5 times the "
                    r"interquartile range, and more extreme observations are omitted from the display. The calculations "
                    r"retain every finite, positive interval-length ratio in the raw Monte Carlo CSV.}",
                    rf"\label{{fig:mc-interval-length-ratios-{token}}}",
                    r"\end{figure}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(tex)
    return outputs


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = read_and_audit()
    summary = summarize(data)
    outputs: list[Path] = [SUMMARY]
    for tau_key in TARGETS:
        outputs.append(expectile_table(summary, tau_key, "xi"))
        outputs.append(expectile_table(summary, tau_key, "ce"))
    outputs.extend(interval_ratio_box_statistics(data))
    print("Generated artifacts:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
