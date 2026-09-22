# startup commands

## 1. mlflow (terminal 1)
```
mlflow ui --port 5000
```

## 2. api server (terminal 2)
```
uvicorn app:app --reload
```

## 3. kubernetes deployments (terminal 3, run once)
```
kubectl apply -f deployment/kubernetes/
```

## 4. port-forwards (terminal 3, after apply)
```
kubectl port-forward service/prometheus-service 9090:9090
kubectl port-forward service/grafana-service 3000:3000
```

## 5. grafana setup (first time only)
- open http://localhost:3000 (admin / admin)
- connections > data sources > add prometheus > url: `http://prometheus-service:9090` > save & test
- dashboards > import > upload `monitoring/grafana-dashboard.json`

## endpoints
| service | url |
|---|---|
| api | http://localhost:8000 |
| api docs | http://localhost:8000/docs |
| mlflow | http://localhost:5000 |
| prometheus | http://localhost:9090 |
| grafana | http://localhost:3000 |
