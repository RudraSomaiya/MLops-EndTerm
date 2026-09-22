from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # mock ModelServing before importing app to prevent mlflow calls at import time
    mock_serving = MagicMock()
    mock_serving.predict.return_value = {
        "prediction": "approved",
        "probability": 0.85,
    }
    mock_serving.explain.return_value = None

    with patch("app.ModelServing", return_value=mock_serving):
        from app import app
        with TestClient(app) as c:
            yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200_with_correct_schema(client):
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "counterfactual" in data


def test_predict_returns_counterfactual_on_rejection():
    # when model predicts rejected, counterfactual should be included
    mock_serving = MagicMock()
    mock_serving.predict.return_value = {
        "prediction": "rejected",
        "probability": 0.82,
    }
    mock_serving.explain.return_value = {
        "changes_needed": {"cibil_score": 720},
        "outcome_if_changed": "approved",
    }

    with patch("app.ModelServing", return_value=mock_serving):
        from app import app
        with TestClient(app) as c:
            payload = {
                "no_of_dependents": 2, "education": "Graduate",
                "self_employed": "No", "income_annum": 4000000,
                "loan_amount": 29900000, "loan_term": 12,
                "cibil_score": 400, "residential_assets_value": 1000000,
                "commercial_assets_value": 500000, "luxury_assets_value": 2000000,
                "bank_asset_value": 300000,
            }
            response = c.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == "rejected"
            assert data["counterfactual"] is not None
            assert "changes_needed" in data["counterfactual"]


def test_predict_missing_field_returns_422():
    # missing required field should return 422 validation error
    mock_serving = MagicMock()
    with patch("app.ModelServing", return_value=mock_serving):
        from app import app
        with TestClient(app) as c:
            payload = {"no_of_dependents": 2}
            response = c.post("/predict", json=payload)
            assert response.status_code == 422
