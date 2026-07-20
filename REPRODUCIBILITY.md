# Reproducibility guidelines and audit

This package follows guidance from the Social Science Data Editors, CodeRefinery, and GitHub.

## Checklist

1. Document purpose, contents, software, inputs, commands, outputs, runtime, and limitations.
2. Separate reusable code, tests, workflows, immutable inputs, and derived results.
3. Declare dependencies and compatibility; lock the complete environment.
4. Use fixed seeds and thread-order-independent replication streams.
5. Provide unit tests, a smoke test, a reduced pilot, and the full workflow.
6. Use repository-relative paths; never developer-specific absolute paths.
7. Preserve inputs and document provenance, licenses, and transformations.
8. Version useful small outputs; ignore caches and large regenerable artifacts.
9. Include reuse terms, machine-readable citation, and contribution guidance.
10. Run tests in continuous integration.
11. Exclude credentials, private data, local state, and manuscripts.
12. Tag the publication version and archive the release with a DOI.

## Project audit

| Area | Finding and action |
| --- | --- |
| Structure | Clear package/tests/scripts/data/results layout retained and documented. |
| Environment | Existing Julia 1.12.6 manifest retained; Julia compatibility and Python requirements added. |
| Randomness | Fixed, per-replication seeds are independent of thread scheduling. |
| Verification | Unit tests and smoke test documented; CI added. |
| Paths | Scripts derive paths from their location; no developer paths are required. |
| Data | Input is versioned, but provider, retrieval date, and terms require author confirmation. |
| Large files | Two roughly 275 MB raw grids are excluded and documented as regenerable. |
| Generated files | Previews, caches, manuscript material, and large intermediates are ignored. |
| Metadata | README, license, citation, contribution, and changelog files added. |
| Code | Scientific refactoring was avoided during packaging; future refactors should be test-led. |

## Before journal deposit

- Add data provider, retrieval date, query, citation, terms, and transformations to `data/README.md`.
- Benchmark full-run time, memory, CPU, thread count, storage, OS, and Julia version.
- Run the complete workflow from a fresh clone and record any numerical deviations.
- Complete article authors, ORCIDs, title, journal, year, and DOI in `CITATION.cff`.
- Tag and archive the submitted release; add the archive DOI.
- Deposit full raw draws in a data repository/release asset and publish checksums if required.
