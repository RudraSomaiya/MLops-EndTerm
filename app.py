from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from src.prediction.predict import ModelServing
from src.utils.logger import get_logger

logger = get_logger(__name__)

# custom prometheus metrics
PREDICTION_COUNTER = Counter(
    "prediction_total",
    "Total number of predictions made",
)
REJECTION_COUNTER = Counter(
    "prediction_rejected_total",
    "Total number of rejected predictions",
)
APPROVAL_RATE = Gauge(
    "approval_rate",
    "Rolling approval rate (approved / total)",
)

# track counts for rolling rate calculation
_total_predictions = 0
_total_approvals = 0


class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    counterfactual: list | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load model once, reuse across requests
    logger.info("loading model and dice explainer...")
    app.state.model = ModelServing()
    logger.info("model loaded, ready to serve")
    yield
    # shutdown
    logger.info("shutting down")


app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts loan approval with counterfactual explanations on rejection",
    version="1.0.0",
    lifespan=lifespan,
)

# auto-expose /metrics endpoint for prometheus
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    global _total_predictions, _total_approvals

    try:
        input_dict = application.model_dump()
        result = app.state.model.predict(input_dict)

        PREDICTION_COUNTER.inc()
        _total_predictions += 1

        response = {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "counterfactual": None,
        }

        if result["prediction"] == "rejected":
            REJECTION_COUNTER.inc()
            counterfactual = app.state.model.explain(input_dict)
            response["counterfactual"] = counterfactual
        else:
            _total_approvals += 1

        # update rolling approval rate
        if _total_predictions > 0:
            APPROVAL_RATE.set(_total_approvals / _total_predictions)

        return response

    except Exception as e:
        logger.error(f"prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
