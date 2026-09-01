# Joint expectile, VaR, and ES Monte Carlo experiment

The default runner evaluates the paper's 48 structural designs. Each simulated
series is fitted once by Gaussian QMLE and the fitted replication is reused for:

- innovation and conditional XP at `tau = 0.01` and `tau = 0.05`;
- one-step-ahead VaR at `alpha = 0.01` and `alpha = 0.05`;
- one-step-ahead ES at `delta = 0.01` and `delta = 0.05`.

The configuration uses one common vector, `risk_levels = [0.01, 0.05]`, so that
`alpha = delta = tau` by construction at each comparison level.

Every fitted replication is immediately flushed to a wide CSV row containing all
point estimates, asymptotic variances, standard errors, confidence limits, interval
lengths, Wald statistics, coverage indicators, numerical diagnostics, and four
interval-length ratios (XP/VaR and XP/ES at both levels). Failed replications are
also recorded with their seed and error message.

## Full experiment

From the repository root, run:

```powershell
julia --project=. --threads=auto scripts/mc/mc_runner.jl
```

The output is:

```text
results/mc/raw/mc_joint_common_levels_replications.csv
```

The full grid contains 48 designs and 10,000 replications per design. This is a
large run. CSV rows may appear in a nonsequential replication order because the
replications are evaluated in parallel.

## Resume after interruption

Run the same command again. The runner reads the existing CSV and skips every
`(design_id, replication)` pair already recorded as either `success` or `failed`.
It never recomputes completed successful replications.

To retry rows previously recorded as failed, set `MC_RETRY_FAILED=true` before
running the same command.

The earlier failed row remains as an audit record; subsequent analysis should use
the successful row when both statuses exist for the same design and replication.

## Small pilot

Run the targeted Normal and `t4` pilot at sample sizes 500 and 5,000:

```powershell
$env:MC_PILOT_REPS = "50"
julia --project=. --threads=auto scripts/mc/run_pilot.jl
```

The pilot output is written to
`results/mc/raw/mc_joint_common_levels_pilot.csv`. To use a different file name,
set `MC_PILOT_OUTPUT` before launching the script.

Increase `MC_PILOT_REPS` after checking the pilot. Repeating the command with
the same output name resumes the pilot and skips completed replications.

For a minimal all-distribution check at `n=500`:

```powershell
$env:MC_SMOKE_REPS = "2"
julia --project=. --threads=auto scripts/mc/run_joint_smoke.jl
```

## Starting a separate run

Use a new `output_name`; do not overwrite the main replication CSV:

```powershell
julia --project=. --threads=auto -e 'include("scripts/mc/mc_runner.jl"); specs=build_full_grid(n_sim=1000); run_joint_grid(specs=specs, output_name="mc_joint_common_levels_1000.csv")'
```

All tables and figures should be constructed from rows with `status=success`.
The four `ratio_ce_tau_*` columns are the replication-level inputs for the
requested 1% and 5% box plots.

## Tables and interval-length figure

After the full CSV is complete, regenerate the four XP tables and the
boxplot statistics directly from the replication-level file:

```console
python scripts/mc/make_joint_risk_artifacts.py
```

Use the isolated Python environment installed from the root `requirements.txt`.

Then render the interval-length figure:

```powershell
julia --project=. scripts/mc/plot_interval_length_ratios.jl
```

The table generator produces unconditional and conditional XP results for
`tau = 0.01` and `tau = 0.05` only, using the paper's prescribed table structure.
It does not produce VaR or ES tables. The figure generator produces two figures:
one comparing XP with VaR and ES at 1%, and one making the same comparison at 5%.
