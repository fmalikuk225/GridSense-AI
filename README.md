# ⚡ GridSense AI

**🚀 Live Dashboard:** https://gridsense-ai-uk.streamlit.app/

**Status:** Deployed portfolio prototype | Synthetic demonstration data by default

**Electricity demand forecasting and peak-load intelligence** — an independent, reproducible AI portfolio prototype for energy-sector analytics.

> **Status:** Portfolio MVP, not production software. **No affiliation with National Grid or RWE.** Synthetic example data are not real energy-system observations.

## What it does
- Upload continuous hourly UTC demand data (`timestamp`, `demand_mw`) or use clearly marked synthetic demonstration data.
- Compare a Random Forest model against a seasonal-naive (same hour yesterday) baseline using a chronological 80/20 split.
- Visualise recent demand, a recursively generated 24-hour forecast, and predicted high-load hours.
- Export forecasts to CSV and inspect MAE, RMSE and R².

## Run locally

Requires Python 3.10+ (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (normally http://localhost:8501).

## Test

```bash
python -m pytest -q
```

## Input contract

CSV columns: `timestamp,demand_mw`. Timestamps must be valid, unique, consecutive hourly instants, expressed in UTC or timezone-aware ISO format, with at least 21 days of history. Demand must be nonnegative. Gaps and invalid values are rejected rather than silently imputed. `demand_mw` is megawatts; do not upload kWh readings without converting to the appropriate power units and time aggregation.

## Method and interpretation

- Features: lagged demand at 1, 24, 48 and 168 hours; cyclical hour, weekday, and month; weekend indicator.
- Model: RandomForestRegressor, fixed hyperparameters and seed 42. No test-set tuning.
- Evaluation: chronological final 20% holdout, with lagged **observed** demand in the holdout. This is a one-step-style historical evaluation, **not** an unbiased measurement of the recursive 24-hour forecast accuracy.
- Future prediction: recursively predict 24 hourly points, using earlier forecasts when observed lag values are unavailable.
- Peak flag: forecasts above the historical 90th-percentile demand threshold. This is a descriptive threshold, **not** an engineered grid capacity or safety limit.
- All timestamps are processed in UTC; local DST and regional demand definitions need dedicated handling for real deployment.

## Limitations / next steps

1. Validate on a properly licensed, public electricity demand dataset with explicit provenance.
2. Implement rolling-origin backtesting for the **full 24-hour horizon**, with interval estimates and multiple seasons.
3. Compare stronger baselines, tune on a separate validation period, and evaluate on untouched test data.
4. Add weather, calendar, operational constraints, and data drift monitoring where licensed data allow.
5. Deploy only after security review, reliability testing and domain-expert assessment.

## Repository layout

- `app.py` — Streamlit user interface
- `src/forecast.py` — data checks, features, training, evaluation, forecasting
- `tests/` — basic regression and validation tests
- `docs/` — business case and architecture
- `.streamlit/config.toml` — app appearance

## Streamlit Community Cloud deployment

1. Push this repository to GitHub (do not include `.venv`, private data or secrets).
2. Sign in to https://share.streamlit.io/ with GitHub.
3. Select repository, branch `main`, entry point `app.py`.
4. Deploy and check the app's synthetic demo and CSV upload.
5. Add the verified live app URL to the GitHub repository description and README.

Community hosting may sleep or become unavailable; deployment does not guarantee continuous uptime.

## Licence

MIT; see `LICENSE`.

## Industry-facing validation (v2)

The evaluation tab now includes a **recursive 24-hour backtest** across six forecast origins in the chronological holdout. Unlike one-step lagged holdout predictions, each forecast uses only observations available at its origin. A 24-hour seasonal naive baseline is evaluated at the same origins. The origins are sampled from the test period, with the model trained only on the initial 80% of supervised data. This is a demonstrator, not a fully nested, rolling-retraining evaluation.

**Important limitations:** Demonstration data are synthetic by default; no claims about UK grid performance, cost savings, calibrated forecast uncertainty, or operational readiness are made. There is no weather/calendar-holiday feed, automated external data integration, monitoring, access control, or model drift alerting. Real public UK demand data should be independently sourced, license-checked, converted to continuous hourly UTC observations, and evaluated before public performance claims.

## Container / continuous integration

```bash
docker build -t gridsense-ai .
docker run --rm -p 8501:8501 gridsense-ai
```

GitHub Actions runs automated tests on pushes and pull requests. For Streamlit Community Cloud, choose `app.py` as the entrypoint and install from `requirements.txt`.

## Intended business audience

Energy demand analysts, utility innovation teams and KTP partners. Forecasts and high-demand flags are *decision-support illustrations*, not grid balancing, dispatch or trading instructions. No affiliation with National Grid, RWE or other utilities.
