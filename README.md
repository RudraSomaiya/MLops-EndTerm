# Loan Approval Prediction System (MLOps Pipeline)

This project implements an MLOps pipeline for loan approval prediction, including counterfactual explanations to help understand decision boundaries and provide actionable feedback for rejected applicants.

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Machine Learning | scikit-learn |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Explainability | DiCE |
| Metrics Monitoring | Prometheus |
| Dashboards | Grafana |
| Containerization | Docker |
| Orchestration | Kubernetes |

## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── transformation/
│   ├── training/
│   ├── prediction/
│   └── utils/
├── tests/
├── deployment/
│   ├── kubernetes/
│   └── docker/
├── monitoring/
├── .github/
│   └── workflows/
├── dvc.yaml
├── pyproject.toml
├── config.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── app.py
```

## Setup Instructions

1. Clone the repository.
2. Create virtual environment:
   ```bash
   uv venv --python 3.11 .venv
   ```
3. Activate virtual environment:
   * Windows: `.venv\Scripts\activate`
   * Unix: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   uv sync
   ```
5. Place the dataset at `data/raw/loan_approval_dataset.csv`.
6. Start MLflow server:
   ```bash
   mlflow ui --port 5000
   ```
7. Run the DVC pipeline:
   ```bash
   dvc repro
   ```
8. Start the API server:
   ```bash
   uvicorn app:app --reload
   ```

## API Endpoints

### GET /health
Health check endpoint.

### POST /predict
Predicts loan approval status and generates counterfactual explanations.

Request:
```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 5000000,
  "loan_amount": 10000000,
  "loan_term": 15,
  "cibil_score": 750,
  "residential_assets_value": 12000000,
  "commercial_assets_value": 3000000,
  "luxury_assets_value": 4000000,
  "bank_asset_value": 2000000
}
```

Response (approved):
```json
{
  "prediction": "approved",
  "probability": 0.92,
  "counterfactual": null
}
```

Response (rejected - includes counterfactual):
```json
{
  "prediction": "rejected",
  "probability": 0.18,
  "counterfactual": {
    "changes_needed": {
      "cibil_score": 720,
      "loan_amount": 8500000
    },
    "outcome_if_changed": "approved"
  }
}
```

## Running Tests

Run the test suite using pytest:
```bash
uv run pytest tests/ -v
```

## Docker

Build and run the containerized application:
```bash
docker build -t loan-approval-mlops .
docker run -p 8000:8000 loan-approval-mlops
```

## Monitoring

Metrics are exposed for Prometheus to scrape, which can then be visualized using Grafana dashboards. The metrics include request latency, error rates, and model prediction distributions.

## Explainability

A key differentiator of this system is the integration of Counterfactual Explanations using DiCE. This provides users with concrete examples of how input features could be changed to flip a negative outcome, offering actionable feedback to applicants.
