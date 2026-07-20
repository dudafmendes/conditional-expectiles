# Conditional Expectiles: replication package

This repository contains Eduardo Mendes's Julia implementation, Monte Carlo experiments, and empirical financial application for conditional expectiles. It separates source, tests, workflows, inputs, and results; locks the Julia environment; and uses deterministic seeds.

## Contents

| Path | Purpose |
| --- | --- |
| `src/` | GARCH/GJR-GARCH and expectile code |
| `test/` | Unit and numerical tests |
| `scripts/mc/` | Monte Carlo workflows and exports |
| `scripts/applications_crypto.jl` | Rolling risk application |
| `data/` | Versioned application input |
| `results/mc/` | Monte Carlo reference outputs |
| `output/applications/` | Application reference outputs |

The manuscript under `docs/`, previews, caches, and full raw Monte Carlo grids are excluded. Each full raw grid is about 275 MB and can be regenerated.

## Requirements and setup

Use Julia 1.12.6; the exact dependency graph is in `Manifest.toml`. Python 3.10+ is needed only for paper figures.

```console
julia --project=. -e "using Pkg; Pkg.instantiate()"
python -m venv .venv
# activate .venv, then:
python -m pip install -r requirements.txt
```

## Quick verification

```console
julia --project=. -e "using Pkg; Pkg.test()"
julia -t auto --project=. scripts/mc/smoke_test.jl
```

## Monte Carlo replication

The paper design has 48 scenarios spanning GARCH/GJR-GARCH, Normal/Student-t innovations, low/high persistence, and four sample sizes. Each paper scenario uses 10,000 replications. Replication `r` uses `MersenneTwister(24681 + r)`, so thread scheduling does not change samples.

Run the 16-design pilot (2,000 replications by default):

```console
julia -t auto --project=. scripts/mc/run_pilot.jl
```

For a fast check, set `MC_PILOT_REPS=20` first. Run the full experiment and exports with:

```console
julia -t auto --project=. scripts/mc/mc_runner.jl
julia --project=. scripts/mc/export_paper_results.jl
python scripts/mc/make_paper_artifacts.py
```

The first command creates `results/mc/raw/mc_grid_results_48.jld2`; expect a long compute-intensive run and a file around 275 MB. Summaries and figures appear under `results/mc/tables/` and `results/mc/figures/`. Compare numerical results within floating-point tolerance, not byte-for-byte.

## Application replication

The sole input is `data/crypto_data.csv`, covering the S&P 500, EUR/USD, Bitcoin, Ether, BNB, and Cardano. For a short check set `APP_MAX_WINDOWS=20`; omit it for the complete run:

```console
julia -t auto --project=. scripts/applications_crypto.jl
python scripts/convert_application_svgs.py
python scripts/build_application_preview_pdf.py
```

Defaults are a 1,000-observation window, expectile and quantile levels of 0.01, and 90% confidence intervals. Override with `APP_WINDOW`, `APP_TAU`, `APP_ALPHA`, and `APP_CI_LEVEL`. Outputs appear in `output/applications/`.

Expected Monte Carlo outputs include `mc_inference_summary.csv`, `mc_xi_main.tex`, `mc_ce_main.tex`, and coverage figures. Application outputs include descriptive statistics, rolling forecasts, backtests, matched gain-loss summaries, and plots.

See `REPRODUCIBILITY.md` for the audit and release checklist, `data/README.md` for data caveats, and `CITATION.cff` for citation metadata. Report problems with the OS, Julia version, command, and complete error.

Copyright (c) 2026 Eduardo Mendes. See `LICENSE`.
