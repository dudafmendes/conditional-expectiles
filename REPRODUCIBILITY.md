# Reproducibility guidelines and audit

This package follows guidance from the Social Science Data Editors, CodeRefinery, and GitHub. Reproduction here means verifying computational outputs under the recorded code, data, environment, and protocol; it does not by itself establish the scientific truth of a method or empirical conclusion.

## Package standard

1. Document purpose, estimands, software, inputs, commands, outputs, resources, and limitations.
2. Separate reusable estimators, tests, workflows, immutable inputs, and generated results.
3. Declare direct dependencies and lock the complete environment.
4. Use stable per-replication random streams that are invariant to thread scheduling.
5. Provide unit tests, schema checks, a small joint smoke run, a pilot, and a resumable full run.
6. Record failed simulations rather than silently dropping them; summarize only valid successful rows.
7. Use repository-relative paths and programmatically generate tables and figures.
8. Version lightweight reference artifacts; exclude caches, previews, backups, and large regenerable raw outputs.
9. Document data lineage, reuse terms, citation metadata, and contribution expectations.
10. Run the unit and smoke suites in continuous integration.
11. Tag the publication version and archive it with a DOI and checksums for separately deposited raw results.

## Updated scientific-computational map

| Target | Implementation | Inputs and numerical procedure | Diagnostics and validation |
| --- | --- | --- | --- |
| GARCH/GJR-GARCH variance path and forecast | `src/GARCHModels.jl`, `src/recursion.jl` | Gaussian QMLE; recursive variance and analytic parameter gradients | Unit tests and finite-difference gradient comparisons |
| Innovation and conditional XP | `src/Expectiles.jl`, `src/derivatives.jl` | Empirical expectiles and root-n asymptotic variances at 1% and 5% | Variance validity, Wald statistics, coverage, interval length |
| Conditional VaR and ES | `src/GaoSongRisk.jl` | Gao--Song feasible two-step FHS; generalized-inverse quantile; Gaussian KDE with Scott bandwidth for VaR | Tail counts, density, information matrix, variance components, interval validity |
| Joint Monte Carlo | `scripts/mc/joint_risk_mc.jl` | 48 structural designs; shared fitted replication; 10,000 replications; seed `24681 + r` | Per-row status/error, boundary and stationarity distances, condition number, maximum fitted variance, runtime |
| Currency application | `scripts/applications_crypto.jl` | Rolling 1,000-observation GJR-GARCH forecasts for BNB, BTC, ETH, and EUR | Skipped-window messages, finite interval checks, forecast and sample metadata |
| Empirical evaluation | `scripts/application_backtests.jl` | Fixed 1% VaR/XP exceedance tests and ES/XP residual identification | Kupiec, independence, conditional-coverage, duration, and identification statistics |

## Artifact and provenance policy

- `Project.toml` and `Manifest.toml` record the Julia environment; `requirements.txt` records direct Python post-processing dependencies.
- Full Monte Carlo rows are written immediately to `results/mc/raw/mc_joint_common_levels_replications.csv`. The runner is resumable and retains failure records.
- All generated `results/` and `output/` artifacts, including replication-level CSVs currently up to about 839 MB, are excluded from Git and reconstructed from the versioned code and dataset.
- Legacy raw grids, `backups/`, `tmp/`, manuscript files, caches, and preview PDFs are also not part of the replication repository.
- Application input data are not modified in place. All application products are reconstructed from `data/crypto_data.csv` by the documented script sequence.

## Scope and unresolved release items

- Unit and smoke tests establish software/numerical checks for the implemented formulas; they do not independently prove the Gao--Song derivation or every statistical assumption.
- The 48-design full run is computationally expensive. A package update may validate smoke/pilot paths without rerunning all 480,000 fits; the final publication release should record the exact full-run completion counts, failure rates, wall time, CPU, memory, storage, OS, Julia version, and source commit.
- The application fixes several probability conventions explicitly. These should remain aligned with the paper: VaR at 1%, primary ES at 2.5%, ES at 1% for common-level comparisons, and XP at 1%.
- Add the original provider, retrieval date, query, citation, license/terms, and preprocessing details for `data/crypto_data.csv` before archival deposit.
- Complete article authors, ORCIDs, title, journal, year, and DOI in `CITATION.cff`.
- Deposit the full replication CSV outside ordinary Git history, publish a cryptographic checksum, tag the corresponding code commit, and archive the release.

## Package-update validation (2026-09-01)

- `julia --project=. test/runtests.jl`: 127/127 tests passed.
- `scripts/mc/smoke_test.jl`: schema and legacy single-design smoke checks passed.
- `scripts/mc/run_joint_smoke.jl` with one replication per design: 12/12 rows completed successfully.
- The existing full joint CSV contains 480,000 successful rows plus its header and no failed rows (48 designs times 10,000 replications); it remains excluded from Git because it is 879,999,266 bytes.
- A two-window-per-asset forecasting probe completed in an isolated temporary copy for BNB, BTC, ETH, and EUR without overwriting the full reference output.
- The current full application output contains 5,489 forecasts: BNB 1,243; BTC 1,921; ETH 1,243; EUR 1,082.
- Backtests regenerated three four-row CSVs; the artifact stages regenerated three LaTeX tables and ten vector PDFs (five figures in color and black-and-white).

Status: the updated package is **VERIFIED** for unit, smoke, reduced forecasting, and artifact-generation workflows under the recorded environment. The full 480,000-fit simulation was audited from its existing replication-level output but was **not rerun** during this packaging update.
