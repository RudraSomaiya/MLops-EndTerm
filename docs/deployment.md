# Deployment

This guide runs the full stack: the prediction API, the MLflow server with the model registry, Prometheus and Grafana. There are two routes, a local Kubernetes cluster and Docker Compose. Both assume you have already run the pipeline once (see the main [README](../README.md)), so `mlflow.db` and `mlartifacts/` contain a registered `loan-approval-model`.

## Kubernetes

`deployment/kubernetes/` has one Deployment and one Service for each component. Every Service is a `NodePort`, and the API finds MLflow through the `MLFLOW_TRACKING_URI` environment variable.

| Component | Deployment | Service | Port | Image |
|---|---|---|---|---|
| API | `loan-approval-app` | `loan-approval-service` | 8000 | `loan-approval-mlops:latest`, built from `Dockerfile` |
| MLflow | `mlflow-server` | `mlflow-service` | 5000 | `mlflow-server:3.16.1`, built from `deployment/docker/Dockerfile.mlflow` |
| Prometheus | `prometheus` | `prometheus-service` | 9090 | `prom/prometheus:v2.54.0` |
| Grafana | `grafana` | `grafana-service` | 3000 | `grafana/grafana:11.1.0` |

The API pod has readiness and liveness probes on `/health` and requests 250m CPU and 512 MiB of memory, with limits of 500m and 1 GiB. Prometheus reads its scrape config from the `prometheus-config` ConfigMap defined in `prometheus-deployment.yaml`, which scrapes `loan-approval-service:8000/metrics` every 15 seconds.

### 1. Point MLflow at your files

The MLflow pod does not train anything. It serves the `mlflow.db` and `mlartifacts/` you produced locally, mounted with `hostPath` volumes. `mlflow-deployment.yaml` currently points at the original checkout:

```yaml
path: /run/desktop/mnt/host/d/Uni Stuff/Final Independence Term/ML-OPS/EndTerm/mlflow.db
```

Change both `hostPath` entries to the location of your clone. On Docker Desktop, a Windows path such as `C:\code\MLops-EndTerm` becomes `/run/desktop/mnt/host/c/code/MLops-EndTerm`. On minikube, mount the folder into the node first (for example `minikube mount <your-clone>:/mlflow-host`) and use that path instead.

### 2. Build the images

```bash
docker build -t loan-approval-mlops:latest .
docker build -t mlflow-server:3.16.1 -f deployment/docker/Dockerfile.mlflow .
```

The API image installs dependencies with uv from the lock file and only copies `app.py`, `config.yaml` and `src/`. The MLflow image installs nothing except `mlflow==3.16.1`, which keeps it small and matches the version that created the database.

Docker Desktop's Kubernetes uses the local image cache directly. On minikube, load both images into the cluster:

```bash
minikube image load loan-approval-mlops:latest
minikube image load mlflow-server:3.16.1
```

### 3. Deploy

On Windows, `k8s-deploy.bat` does everything from here: it applies the manifests, waits up to three minutes for each pod to become ready, prints the pods and services, and opens a port-forward window for each component. Press a key in the script's window to close the forwards again.

To do it by hand:

```bash
kubectl apply -f deployment/kubernetes/
kubectl wait --for=condition=ready pod -l app=loan-approval-app --timeout=180s

kubectl port-forward service/loan-approval-service 8000:8000
kubectl port-forward service/mlflow-service 5000:5000
kubectl port-forward service/prometheus-service 9090:9090
kubectl port-forward service/grafana-service 3000:3000
```

Run each port-forward in its own terminal. The services are then reachable at:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 4. Set up Grafana

1. Log in at http://localhost:3000 with `admin` / `admin`.
2. Go to Connections, then Data sources, and add a Prometheus data source with the URL `http://prometheus-service:9090`. Save and test it.
3. Go to Dashboards, then Import, and upload `monitoring/grafana-dashboard.json`.

The dashboard shows the request rate, the approval rate, rejections over the last five minutes and p95 latency. Send a few requests to `/predict` to see the panels move.

## Docker Compose

`deployment/docker/docker-compose.yml` starts the same four services on one bridge network:

```bash
docker compose -f deployment/docker/docker-compose.yml up --build
```

The Compose MLflow service is a fresh `ghcr.io/mlflow/mlflow:v2.16.0` container with no access to your local `mlflow.db`, so its registry starts empty and the API cannot load a model until one is registered against it. Start the `mlflow` service on its own first, run `dvc repro` (it logs to `http://localhost:5000`), then start the rest.

For Grafana, use `http://prometheus:9090` as the data source URL, since the Compose service is named `prometheus`.

## Monitoring only

If you run the API directly with uvicorn, `monitoring/docker-compose.yml` starts just Prometheus and Grafana:

```bash
docker compose -f monitoring/docker-compose.yml up
```

Its Prometheus config has a `loan-approval-local` job that scrapes `host.docker.internal:8000`, which is the API running on your machine.
