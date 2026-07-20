# Application data

`crypto_data.csv` is the input to `scripts/applications_crypto.jl`. It contains daily observations for `^GSPC`, `EURUSD=X`, `BTC-USD`, `ETH-USD`, `BNB-USD`, and `ADA-USD`.

The application reads `ticker`, `ref_date`, and `price_adjusted`, sorts by date, drops missing prices, and computes returns as `100 * log(P_t / P_{t-1})`.

Before archival deposit, add the original provider, retrieval date, download procedure, citation, terms, and preprocessing. These cannot be recovered reliably from the CSV.
