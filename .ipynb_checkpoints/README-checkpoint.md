# MLOPS ASSIGNMENT 7
MLOPS IRIS pipline with CI/CD including deployments on Google Kubenetes Engine with Docker Image Along with Stress Testing using WRK


| File / Directory | Purpose |
| --- | --- |
| `.github/workflows/main.yml` | Automatically builds the Docker image and deploys the app to Kubernetes after Infrastructure setup is done. After which it stress tests the deployment and check HPA Scaling |
| `k8s/deployment.yaml` | A Kubernetes manifest that defines how to run the application's container as a service. Horizontal Scaling Service |
| `Dockerfile` | A text file containing instructions to build the application into a portable Docker container image. |
| `test_data.py` | Contains unit tests to validate the model's logic and prevent regressions. |
| `main.py` | The FastAPI app that loads the model using code from week 5's train and fetch files along with test routes and serves predictions via an API. |
