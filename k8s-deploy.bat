:: k8s-deploy.bat
:: run this after docker build and minikube start to deploy everything to the cluster
:: usage: k8s-deploy.bat

@echo off
echo [1/5] loading docker image into minikube...
minikube image load loan-approval-mlops:latest

echo [2/5] applying all kubernetes manifests...
kubectl apply -f deployment/kubernetes/mlflow-deployment.yaml
kubectl apply -f deployment/kubernetes/mlflow-service.yaml
kubectl apply -f deployment/kubernetes/prometheus-deployment.yaml
kubectl apply -f deployment/kubernetes/prometheus-service.yaml
kubectl apply -f deployment/kubernetes/grafana-deployment.yaml
kubectl apply -f deployment/kubernetes/grafana-service.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

echo [3/5] waiting for pods to be ready (up to 3 minutes)...
kubectl wait --for=condition=ready pod -l app=mlflow-server --timeout=180s
kubectl wait --for=condition=ready pod -l app=prometheus --timeout=180s
kubectl wait --for=condition=ready pod -l app=grafana --timeout=180s
kubectl wait --for=condition=ready pod -l app=loan-approval-app --timeout=180s

echo [4/5] pod status:
kubectl get pods
kubectl get services

echo [5/5] starting port-forwards in background...
:: forward the app
start "loan-api" kubectl port-forward service/loan-approval-service 8000:8000
:: forward mlflow
start "mlflow-ui" kubectl port-forward service/mlflow-service 5000:5000
:: forward prometheus
start "prometheus" kubectl port-forward service/prometheus-service 9090:9090
:: forward grafana
start "grafana" kubectl port-forward service/grafana-service 3000:3000

echo.
echo all services forwarded:
echo   API:        http://localhost:8000
echo   MLflow:     http://localhost:5000
echo   Prometheus: http://localhost:9090
echo   Grafana:    http://localhost:3000
echo.
echo press any key to stop all port-forwards...
pause
taskkill /FI "WINDOWTITLE eq loan-api" /F 2>nul
taskkill /FI "WINDOWTITLE eq mlflow-ui" /F 2>nul
taskkill /FI "WINDOWTITLE eq prometheus" /F 2>nul
taskkill /FI "WINDOWTITLE eq grafana" /F 2>nul
