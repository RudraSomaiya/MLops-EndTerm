
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    # 10 rows with 4 approved / 6 rejected so stratified split works
    return pd.DataFrame({
        "loan_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "no_of_dependents": [2, 0, 3, 1, 5, 0, 2, 4, 1, 3],
        "education": [
            "Graduate", "Not Graduate", "Graduate", "Not Graduate", "Graduate",
            "Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate",
        ],
        "self_employed": ["No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "income_annum": [
            9600000, 4100000, 9100000, 8200000, 9800000,
            7500000, 3200000, 8800000, 6000000, 5500000,
        ],
        "loan_amount": [
            29900000, 12200000, 29700000, 30700000, 24200000,
            15000000, 10000000, 20000000, 18000000, 14000000,
        ],
        "loan_term": [12, 8, 20, 8, 20, 15, 10, 12, 6, 16],
        "cibil_score": [778, 417, 506, 467, 382, 750, 320, 690, 800, 450],
        "residential_assets_value": [
            2400000, 2700000, 7100000, 18200000, 12400000,
            5000000, 1500000, 9000000, 3000000, 4000000,
        ],
        "commercial_assets_value": [
            17600000, 2200000, 4500000, 3300000, 8200000,
            6000000, 1000000, 7000000, 4000000, 2000000,
        ],
        "luxury_assets_value": [
            22700000, 8800000, 33300000, 23300000, 29400000,
            15000000, 5000000, 20000000, 12000000, 8000000,
        ],
        "bank_asset_value": [
            8000000, 3300000, 12800000, 7900000, 5000000,
            6000000, 2000000, 10000000, 4500000, 3000000,
        ],
        "loan_status": [
            "Approved", "Rejected", "Rejected", "Rejected", "Rejected",
            "Approved", "Rejected", "Approved", "Approved", "Rejected",
        ],
    })


@pytest.fixture
def sample_valid_input():
    # a valid loan application input dict for prediction tests
    return {
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


@pytest.fixture
def tmp_data_dir(tmp_path):
    # create a temporary data directory structure
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    return tmp_path
