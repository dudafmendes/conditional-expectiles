# Conditional Expectiles: replication package

This repository contains the Julia implementation, Monte Carlo experiments, and empirical currency-risk application for conditional expectiles (XP), value at risk (VaR), and expected shortfall (ES). It includes reusable GARCH/GJR-GARCH estimation code, Gao--Song filtered-historical-simulation inference, deterministic and resumable simulations, application backtests, and programmatic paper tables and figures.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/` | GARCH/GJR-GARCH, expectile, and Gao--Song VaR/ES inference code |
| `test/` | Mathematical, numerical, and integration tests |
| `scripts/mc/` | Joint XP/VaR/ES Monte Carlo runners and artifact builders |
| `scripts/applications_crypto.jl` | Rolling four-currency risk forecasts |
| `scripts/application_backtests.jl` | Fixed-level VaR, ES, and XP diagnostics |
| `scripts/build_application_tables.jl` | Application Tables 1--3 |
| `scripts/build_selected_application_figures.jl` | Application Figures 1--5 |
| `data/crypto_data.csv` | Versioned application input |
| `results/mc/` | Generated Monte Carlo rows, tables, and figures (ignored by Git) |
| `output/applications/` | Generated application forecasts and artifacts (ignored by Git) |

Only code, tests, environments, documentation, and the input dataset are versioned. The manuscript under `docs/`, local backups, temporary files, and all generated `results/` and `output/` artifacts are excluded from Git. The complete joint CSV is about 839 MB and is deterministically regenerable.

## Requirements and setup

- Julia 1.12.6; `Project.toml` declares direct dependencies and `Manifest.toml` locks the complete environment.
- Python 3.10+ for Monte Carlo post-processing.
- Multiple CPU cores and substantial storage for the complete simulation and application.

From the repository root:

```console
julia --project=. -e "using Pkg; Pkg.instantiate()"
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
```

## Quick verification

```console
julia --project=. -e "using Pkg; Pkg.test()"
julia -t auto --project=. scripts/mc/smoke_test.jl
julia -t auto --project=. scripts/mc/run_joint_smoke.jl
```

The last command writes an ignored replication-level smoke CSV under `results/mc/raw/`. Set `MC_SMOKE_REPS` or `MC_SMOKE_OUTPUT` to change its size or name.

## Monte Carlo replication

The paper experiment contains 48 structural designs: GARCH and correctly specified GJR-GARCH models; Normal, standardized Student-t(8), and standardized Student-t(4) innovations; low and high persistence; and sample sizes 500, 1,000, 2,500, and 5,000. Each fitted replication is reused to evaluate:

- innovation and one-step conditional XP at `tau = 0.01` and `0.05`;
- one-step VaR at `alpha = 0.01` and `0.05`;
- one-step ES at `delta = 0.01` and `0.05`.

Thus `alpha = delta = tau` at each comparison level. The full grid uses 10,000 replications per design. Replication `r` uses `MersenneTwister(24681 + r)`, making simulated samples independent of thread scheduling. Every success or failure is immediately flushed to CSV with its seed, estimates, root-n variances, intervals, coverage indicators, diagnostics, and interval-length ratios.

Run the resumable 16-design pilot (50 replications by default):

```console
julia -t auto --project=. scripts/mc/run_pilot.jl
```

Configure it with `MC_PILOT_REPS`, `MC_PILOT_OUTPUT`, and `MC_RETRY_FAILED`. Run or resume the complete experiment with:

```console
julia -t auto --project=. scripts/mc/mc_runner.jl
```

The full output is `results/mc/raw/mc_joint_common_levels_replications.csv`. Repeating the command skips completed design/replication pairs; set `MC_RETRY_FAILED=true` to retry failures while retaining their earlier audit rows.

Generate paper tables and interval-width comparisons from successful rows:

```console
python scripts/mc/make_joint_risk_artifacts.py
julia --project=. scripts/mc/plot_interval_length_ratios.jl
```

This creates XP tables at 1% and 5%, summary CSVs, and XP/VaR and XP/ES interval-length ratio figures under `results/mc/tables/` and `results/mc/figures/`. See `scripts/mc/README.md` for restart semantics, schema details, and custom-run examples.

## Empirical application replication

The application uses `data/crypto_data.csv` and retains BNB, BTC, ETH, and EUR/USD. It fits rolling GJR-GARCH models using 1,000 observations and produces one-step forecasts and 95% intervals. Defaults are:

- VaR probability `APP_VAR_ALPHA=0.01`;
- primary ES probability `APP_ES_ALPHA=0.025` plus ES at 1% for common-level comparisons;
- fixed XP level `APP_FIXED_XP_TAU=0.01`;
- confidence level `APP_CI_LEVEL=0.95`.

Run the complete data-to-artifact pipeline in this order:

```console
julia -t auto --project=. scripts/applications_crypto.jl
julia --project=. scripts/application_backtests.jl
julia --project=. scripts/build_application_tables.jl
julia --project=. scripts/build_selected_application_figures.jl
```

For a quick forecasting check, set `APP_MAX_WINDOWS=20` before the first command. Other overrides are `APP_WINDOW`, `APP_VAR_ALPHA`, `APP_ES_ALPHA`, `APP_ES_ALPHA_1PCT`, `APP_FIXED_XP_TAU`, and `APP_CI_LEVEL`.

The pipeline writes:

- `output/applications/rolling_risk_forecasts.csv` and descriptive summaries;
- fixed-level calibration and residual-identification tests under `output/applications/backtests/`;
- manuscript-ready LaTeX tables under `output/applications/tables/`;
- color and black-and-white vector PDFs under `output/applications/paper_figures/`.

The input-data provenance caveat is recorded in `data/README.md`. See `REPRODUCIBILITY.md` for the audit and release checklist and `CITATION.cff` for citation metadata. Report problems with the OS, Julia version, command, thread count, and complete error message.

Copyright (c) 2026 Eduardo Mendes. See `LICENSE`.
