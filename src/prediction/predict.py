import dice_ml
import mlflow
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# dice needs to know which features are continuous vs categorical
CONTINUOUS_FEATURES = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
FEATURE_ORDER = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


class ModelServing:
    def __init__(self):
        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        model_name = config["mlflow"]["model_name"]
        logger.info(f"loading model '{model_name}' from registry...")

        # load latest version of the registered model
        client = mlflow.tracking.MlflowClient()
        # get the latest version (not filtering by stage since newer mlflow deprecated stages)
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"no versions found for model '{model_name}'")

        # pick the latest version by version number
        latest = max(versions, key=lambda v: int(v.version))
        self.run_id = latest.run_id
        model_uri = f"models:/{model_name}/{latest.version}"

        # trusted types must match what was declared in log_model at training time
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info(f"loaded model version {latest.version} from run {self.run_id}")

        # load raw training data for dice from the same run's artifacts
        artifact_path = client.download_artifacts(self.run_id, "data/X_train_raw.csv")
        X_train_raw = pd.read_csv(artifact_path)
        logger.info(f"loaded training data snapshot: {X_train_raw.shape}")

        # dice needs a dataframe with features + target column
        # we don't have y_train here, so we predict on the training data to get labels
        # this is fine because dice just needs the data distribution, not perfect labels
        train_predictions = self.pipeline.predict(X_train_raw)
        X_train_with_target = X_train_raw.copy()
        X_train_with_target["loan_status"] = train_predictions

        # initialise dice
        dice_data = dice_ml.Data(
            dataframe=X_train_with_target,
            continuous_features=CONTINUOUS_FEATURES,
            outcome_name="loan_status",
        )
        dice_model = dice_ml.Model(
            model=self.pipeline,
            backend="sklearn",
            model_type="classifier",
        )
        # random method is fastest for api response times
        self.explainer = dice_ml.Dice(dice_data, dice_model, method="random")
        logger.info("dice explainer initialised")

    def predict(self, input_dict: dict) -> dict:
        """Return prediction label and probability."""
        df = pd.DataFrame([input_dict])[FEATURE_ORDER]
        proba = self.pipeline.predict_proba(df)[0]
        pred_class = int(np.argmax(proba))
        return {
            "prediction": "approved" if pred_class == 1 else "rejected",
            "probability": round(float(proba[pred_class]), 4),
        }

    def explain(self, input_dict: dict) -> dict | None:
        """Generate counterfactual explanation for a rejection."""
        try:
            df = pd.DataFrame([input_dict])[FEATURE_ORDER]
            dice_exp = self.explainer.generate_counterfactuals(
                query_instances=df,
                total_CFs=1,
                desired_class="opposite",
            )

            cf_example = dice_exp.cf_examples_list[0]
            cfs_df = cf_example.final_cfs_df

            if cfs_df is None or cfs_df.empty:
                logger.warning("dice could not generate a counterfactual")
                return None

            # extract only the features that changed
            cf_row = cfs_df.iloc[0]
            original = df.iloc[0]
            changes = {}

            for col in FEATURE_ORDER:
                orig_val = original[col]
                cf_val = cf_row[col]
                # compare with tolerance for floats
                if isinstance(orig_val, (int, float, np.integer, np.floating)):
                    if not np.isclose(float(orig_val), float(cf_val), atol=1e-2):
                        is_np_num = isinstance(cf_val, (np.integer, np.floating))
                        changes[col] = int(cf_val) if is_np_num else cf_val
                elif str(orig_val) != str(cf_val):
                    changes[col] = str(cf_val)

            if not changes:
                return None

            return {
                "changes_needed": changes,
                "outcome_if_changed": "approved",
            }

        except Exception as e:
            logger.error(f"counterfactual generation failed: {e}")
            return None
