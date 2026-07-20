# Contributing

Reports should include the command, complete error, OS, Julia version, thread count, and whether the locked manifest was used. Changes should be focused, tested, and explain changes to seeds, estimands, designs, dependencies, or reference results.

Run `julia --project=. -e "using Pkg; Pkg.test()"` and `julia -t auto --project=. scripts/mc/smoke_test.jl`. Do not commit full raw grids, credentials, private data, caches, or manuscripts.
