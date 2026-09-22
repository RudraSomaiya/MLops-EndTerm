from unittest.mock import MagicMock, patch

import numpy as np


def test_predict_returns_expected_keys():
    # predict should return dict with prediction and probability keys
    with patch("src.prediction.predict.mlflow"):
        from src.prediction.predict import ModelServing

    serving = object.__new__(ModelServing)

    # mock pipeline with predict_proba
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])
    serving.pipeline = mock_pipeline

    input_dict = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 9600000,
        "loan_amount": 29900000,
        "loan_term": 12,
        "cibil_score": 778,
        "residential_assets_value": 2400000,
        "commercial_assets_value": 17600000,
        "luxury_assets_value": 22700000,
        "bank_asset_value": 8000000,
    }

    result = serving.predict(input_dict)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in ["approved", "rejected"]
    assert 0 <= result["probability"] <= 1


def test_predict_approved_when_high_proba():
    # high probability for class 1 should return approved
    with patch("src.prediction.predict.mlflow"):
        from src.prediction.predict import ModelServing

    serving = object.__new__(ModelServing)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    serving.pipeline = mock_pipeline

    input_dict = {
        "no_of_dependents": 0, "education": "Graduate", "self_employed": "No",
        "income_annum": 9600000, "loan_amount": 5000000, "loan_term": 10,
        "cibil_score": 800, "residential_assets_value": 5000000,
        "commercial_assets_value": 3000000, "luxury_assets_value": 10000000,
        "bank_asset_value": 4000000,
    }

    result = serving.predict(input_dict)
    assert result["prediction"] == "approved"


def test_explain_returns_none_on_failure():
    # explain should return None when dice fails, not crash
    with patch("src.prediction.predict.mlflow"):
        from src.prediction.predict import ModelServing

    serving = object.__new__(ModelServing)
    # explainer that raises an exception
    serving.explainer = MagicMock()
    serving.explainer.generate_counterfactuals.side_effect = RuntimeError("dice failed")
    serving.pipeline = MagicMock()

    input_dict = {
        "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
        "income_annum": 4000000, "loan_amount": 29900000, "loan_term": 12,
        "cibil_score": 400, "residential_assets_value": 1000000,
        "commercial_assets_value": 500000, "luxury_assets_value": 2000000,
        "bank_asset_value": 300000,
    }

    result = serving.explain(input_dict)
    assert result is None
