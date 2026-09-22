import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

def transform_data():
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    ingested_path = processed_dir / "ingested.csv"

    logger.info(f"loading data from {ingested_path}")
    df = pd.read_csv(ingested_path)

    df = df.drop(columns=["loan_id"])

    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)

    logger.info("transformation complete, train and test sets saved")

if __name__ == "__main__":
    transform_data()
