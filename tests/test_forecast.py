import pytest
import pandas as pd
from src.forecast import sample_data, validate_frame, fit_evaluate, forecast_next_24

def test_end_to_end():
    data=sample_data(30)
    validated=validate_frame(data.rename_axis('timestamp').reset_index())
    model,results,scores=fit_evaluate(validated)
    future=forecast_next_24(model,validated)
    assert len(future)==24
    assert future.notna().all().all()
    assert (future.iloc[:,0]>=0).all()
    assert 'Random Forest' in scores.index
    assert len(results)>0

def test_missing_hours_rejected():
    frame=sample_data(30).rename_axis('timestamp').reset_index().drop(index=10)
    with pytest.raises(ValueError,match='continuous'):
        validate_frame(frame)

def test_duplicate_hours_rejected():
    frame=sample_data(30).rename_axis('timestamp').reset_index()
    frame.loc[10,'timestamp']=frame.loc[9,'timestamp']
    with pytest.raises(ValueError,match='Duplicate'):
        validate_frame(frame)

from src.forecast import evaluate_24h_origins

def test_recursive_backtest_is_past_only():
    data = sample_data(30)
    model, _, _ = fit_evaluate(data)
    rows, scores = evaluate_24h_origins(model, data, origins=2)
    assert len(rows) == 48
    assert (rows.timestamp > rows.origin_utc).all()
    assert rows.horizon_h.min() == 1
    assert rows.horizon_h.max() == 24
    assert scores.notna().all().all()
