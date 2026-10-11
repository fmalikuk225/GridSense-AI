# Architecture

CSV upload or synthetic example → strict timestamp / demand validation → lag + calendar feature engineering → chronological 80/20 train/test split → Random Forest and seasonal-naive baseline → one-step historical metrics → recursive 24-hour forecast → threshold-based peak signals → Streamlit / Plotly dashboard and CSV export.

All computation occurs in the application process. No external APIs, credentials, or private PhD data are needed. The synthetic generator is explicitly for interface demonstrations, not evidence of operational predictive accuracy.
