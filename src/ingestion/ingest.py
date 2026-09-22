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

def ingest_data():
    project_root = get_project_root()

    raw_path = project_root / "data" / "raw" / "loan_approval_dataset.csv"
    processed_path = project_root / "data" / "processed" / "ingested.csv"

    logger.info(f"loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing expected columns: {missing_cols}")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"saved ingested data to {processed_path}")

if __name__ == "__main__":
    ingest_data()
