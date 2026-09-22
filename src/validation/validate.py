import json

import pandas as pd

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "loan_id", "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value", "loan_status"
]

def _is_string_like_dtype(series: pd.Series) -> bool:
    """pandas 3.x uses StringDtype by default, not object."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def run_checks(df: pd.DataFrame) -> list:
    results = []

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    results.append({
        "name": "columns_present",
        "status": "fail" if missing else "pass",
        "detail": f"missing: {missing}" if missing else "all columns present"
    })

    # if columns are missing, range/value checks will crash so skip them
    if missing:
        return results

    numeric_cols = ["loan_id", "no_of_dependents", "income_annum", "loan_amount",
                    "loan_term", "cibil_score", "residential_assets_value",
                    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
    obj_cols = ["education", "self_employed", "loan_status"]

    dtype_fail = False
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_fail = True
    for col in obj_cols:
        if not _is_string_like_dtype(df[col]):
            dtype_fail = True

    results.append({
        "name": "dtypes",
        "status": "fail" if dtype_fail else "pass",
        "detail": "numeric and object types match expected"
    })

    nulls = df.isnull().sum().sum()
    results.append({
        "name": "no_nulls",
        "status": "fail" if nulls > 0 else "pass",
        "detail": f"found {nulls} nulls"
    })

    cibil_invalid = ((df["cibil_score"] < 300) | (df["cibil_score"] > 900)).sum()
    results.append({
        "name": "cibil_range",
        "status": "fail" if cibil_invalid > 0 else "pass",
        "detail": f"{cibil_invalid} invalid rows"
    })

    loan_amt_invalid = (df["loan_amount"] <= 0).sum()
    results.append({
        "name": "loan_amount_positive",
        "status": "fail" if loan_amt_invalid > 0 else "pass",
        "detail": f"{loan_amt_invalid} invalid rows"
    })

    term_invalid = (df["loan_term"] <= 0).sum()
    results.append({
        "name": "loan_term_positive",
        "status": "fail" if term_invalid > 0 else "pass",
        "detail": f"{term_invalid} invalid rows"
    })

    inc_invalid = (df["income_annum"] <= 0).sum()
    results.append({
        "name": "income_positive",
        "status": "fail" if inc_invalid > 0 else "pass",
        "detail": f"{inc_invalid} invalid rows"
    })

    assets_invalid = ((df["commercial_assets_value"] < 0) |
                      (df["luxury_assets_value"] < 0) |
                      (df["bank_asset_value"] < 0)).sum()
    results.append({
        "name": "assets_non_negative",
        "status": "fail" if assets_invalid > 0 else "pass",
        "detail": f"{assets_invalid} invalid rows"
    })

    res_zero = (df["residential_assets_value"] == 0).sum()
    results.append({
        "name": "residential_assets_check",
        "status": "warning" if res_zero > 0 else "pass",
        "detail": f"{res_zero} zero values"
    })

    edu_invalid = (~df["education"].isin(["Graduate", "Not Graduate"])).sum()
    results.append({
        "name": "education_values",
        "status": "fail" if edu_invalid > 0 else "pass",
        "detail": f"{edu_invalid} invalid rows"
    })

    emp_invalid = (~df["self_employed"].isin(["Yes", "No"])).sum()
    results.append({
        "name": "self_employed_values",
        "status": "fail" if emp_invalid > 0 else "pass",
        "detail": f"{emp_invalid} invalid rows"
    })

    status_invalid = (~df["loan_status"].isin(["Approved", "Rejected"])).sum()
    results.append({
        "name": "loan_status_values",
        "status": "fail" if status_invalid > 0 else "pass",
        "detail": f"{status_invalid} invalid rows"
    })

    return results

def validate_data():
    project_root = get_project_root()
    ingested_path = project_root / "data" / "processed" / "ingested.csv"
    report_path = project_root / "data" / "processed" / "validation_report.json"

    logger.info(f"loading data from {ingested_path}")
    df = pd.read_csv(ingested_path)

    results = run_checks(df)

    fails = []
    for res in results:
        logger.info(f"check {res['name']}: {res['status']} - {res['detail']}")
        if res["status"] == "fail":
            fails.append(res["name"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"validation report saved to {report_path}")

    if fails:
        raise ValueError(f"validation failed for checks: {fails}")

if __name__ == "__main__":
    validate_data()
