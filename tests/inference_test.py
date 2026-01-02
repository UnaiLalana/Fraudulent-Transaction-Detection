import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import predict


def test_predict_returns_valid_output():
    """
    Test that inference returns valid probability and prediction
    """
    sample_input = {
    "TX_AMOUNT": 120.5,
    "TX_TIME_SECONDS": 14*3600 + 12*60,
    "TX_TIME_DAYS": 12,
    "TX_HOUR": 14,
    "TX_DAY": 12,
    "TX_DAYOFWEEK": 3,
    "TX_MONTH": 6,
    "TX_IS_WEEKEND": 0,
}

    result = predict(sample_input)

    assert "fraud_probability" in result
    assert "fraud_prediction" in result
    assert "decision_threshold" in result

    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["fraud_prediction"], int)

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["fraud_prediction"] in [0, 1]

def test_predict_probability_changes():
    sample_1 = {
        "TX_AMOUNT": 5.0,
        "TX_TIME_SECONDS": 10*3600 + 10*60,
        "TX_TIME_DAYS": 10,
        "TX_HOUR": 10,
        "TX_DAY": 10,
        "TX_DAYOFWEEK": 1,
        "TX_MONTH": 6,
        "TX_IS_WEEKEND": 0,
    }

    sample_2 = {
        "TX_AMOUNT": 5000.0,
        "TX_TIME_SECONDS": 2*3600 + 10*60,
        "TX_TIME_DAYS": 10,
        "TX_HOUR": 2,
        "TX_DAY": 10,
        "TX_DAYOFWEEK": 6,
        "TX_MONTH": 6,
        "TX_IS_WEEKEND": 1,
    }

    res1 = predict(sample_1)
    res2 = predict(sample_2)

    assert res1["fraud_probability"] != res2["fraud_probability"]
