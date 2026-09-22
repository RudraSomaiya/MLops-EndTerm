import json
import sys

# mlflow prints emoji in run URLs -- reconfigure stdout to utf-8 so it doesn't crash on windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor(categorical_cols: list[str], numerical_cols: list[str]) -> ColumnTransformer:
    """Build the preprocessing step for the sklearn pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    categories=[
                        ["Not Graduate", "Graduate"],  # education: ordinal order
                        ["No", "Yes"],  # self_employed: ordinal order
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_cols,
            ),
            ("num", StandardScaler(), numerical_cols),
        ],
        remainder="drop",
    )


def get_candidates() -> dict[str, object]:
    """Return model candidates to evaluate."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def train_and_evaluate():
    config = load_config()
    project_root = get_project_root()
    processed_dir = project_root / config["data"]["processed_dir"]

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    categorical_cols = config["features"]["categorical"]
    numerical_cols = config["features"]["numerical"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    results = []
    best_f1 = -1
    best_run_id = None
    best_model_name = None

    candidates = get_candidates()

    for name, model in candidates.items():
        logger.info(f"training {name}...")

        preprocessor = build_preprocessor(categorical_cols, numerical_cols)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        with mlflow.start_run(run_name=name) as run:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }

            mlflow.log_param("model_type", name)
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_metrics(metrics)

            # log the full pipeline so it can be loaded directly for inference
            # skops requires explicit trust for tree internals -- safe since we trained the model
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.tree._classes.DecisionTreeClassifier",
                    "sklearn.ensemble._forest.RandomForestClassifier",
                    "numpy.dtype",
                    "numpy.ndarray",
                ],
            )

            # dice needs the raw training data to generate counterfactuals
            raw_path = processed_dir / "X_train_raw.csv"
            mlflow.log_artifact(str(raw_path), artifact_path="data")

            logger.info(f"{name} - f1: {metrics['f1']:.4f}, accuracy: {metrics['accuracy']:.4f}")

            results.append({"model": name, "run_id": run.info.run_id, **metrics})

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_run_id = run.info.run_id
                best_model_name = name

    # register best model
    logger.info(f"best model: {best_model_name} (f1={best_f1:.4f}), registering...")
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, config["mlflow"]["model_name"])
    logger.info(f"registered model version {mv.version} as '{config['mlflow']['model_name']}'")

    # save comparison for dvc tracking
    comparison_path = processed_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        comparison = {
            "results": results,
            "best_model": best_model_name,
            "best_run_id": best_run_id,
        }
        json.dump(comparison, f, indent=2)

    logger.info(f"model comparison saved to {comparison_path}")


if __name__ == "__main__":
    train_and_evaluate()
