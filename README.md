# ML Model API - Iris Classification

Production-ready machine learning API for iris species classification using FastAPI and Docker.

**Live Demo:** [https://ml-model-api-niyy.onrender.com](https://ml-model-api-niyy.onrender.com)  
**API Docs:** [https://ml-model-api-niyy.onrender.com/docs](https://ml-model-api-niyy.onrender.com/docs)

---

## 🎯 Project Overview

This project demonstrates end-to-end ML deployment skills:
- Training and serializing ML models
- Building REST APIs for model inference
- Containerizing applications with Docker
- Deploying to cloud infrastructure
- Following production best practices

**Use Case:** Predict iris flower species (Setosa, Versicolor, Virginica) from 4 measurements (sepal/petal length and width).

---

## 🚀 Quick Start

### Test the Live API

```bash
# Health check
curl https://ml-model-api-niyy.onrender.com/health

# Make a prediction
curl -X POST https://ml-model-api-niyy.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Response:
{
  "predicted_class": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | scikit-learn 1.7.1 | Model training (RandomForest) |
| **API Framework** | FastAPI 0.116.1 | REST API with auto-generated docs |
| **Server** | Uvicorn | ASGI web server |
| **Validation** | Pydantic 2.11.7 | Request/response validation |
| **Containerization** | Docker | Multi-stage builds, reproducible environments |
| **Orchestration** | Docker Compose | Local development setup |
| **Deployment** | Render | Cloud hosting with auto-deploy |
| **CI/CD** | GitHub | Version control, automated deployment |

---

## 📦 Installation & Local Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/Regan-Milne/ml-model-api.git
cd ml-model-api

# Train the model (optional - model already included)
python train.py

# Option 1: Run with Docker Compose (recommended)
docker-compose up --build

# Option 2: Run locally with Python
pip install -r requirements.txt
uvicorn app.main:app --reload

# Access API
# Local: http://localhost:8080
# Docs: http://localhost:8080/docs
```

---

## 📊 Model Performance

- **Algorithm:** RandomForest Classifier
- **Training Accuracy:** 93.33%
- **Dataset:** Iris (150 samples, 3 classes)
- **Features:** 4 numerical measurements
- **Model Size:** ~50 KB

### Sample Predictions

| Input Features | Predicted Class | Confidence |
|---------------|----------------|------------|
| 5.1, 3.5, 1.4, 0.2 | Setosa | 100% |
| 6.7, 3.1, 4.7, 1.5 | Versicolor | 99.65% |
| 6.3, 2.9, 5.6, 1.8 | Virginica | 98% |

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
│ (Browser/   │
│   CLI/App)  │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────┐
│     Render Cloud            │
│  ┌────────────────────────┐ │
│  │   Docker Container     │ │
│  │  ┌──────────────────┐  │ │
│  │  │  FastAPI Server  │  │ │
│  │  │  (Uvicorn)       │  │ │
│  │  └────────┬─────────┘  │ │
│  │           │            │ │
│  │  ┌────────▼─────────┐  │ │
│  │  │  Model Service   │  │ │
│  │  │  (Singleton)     │  │ │
│  │  └────────┬─────────┘  │ │
│  │           │            │ │
│  │  ┌────────▼─────────┐  │ │
│  │  │  RandomForest    │  │ │
│  │  │  Model (.joblib) │  │ │
│  │  └──────────────────┘  │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
```

### Request Flow

1. Client sends POST request to `/predict`
2. FastAPI validates input with Pydantic schemas
3. Model service loads model (once, at startup)
4. RandomForest makes prediction
5. Response includes class, confidence, and probabilities
6. Logs capture request details for monitoring

---

## 🔌 API Endpoints

### `GET /`
Health check endpoint
```json
{
  "message": "Iris Classification API",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

### `GET /health`
Detailed health status with model info
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "accuracy": 0.9333,
    "features": ["sepal length (cm)", ...],
    "classes": ["setosa", "versicolor", "virginica"],
    "n_features": 4
  }
}
```

### `POST /predict`
Single prediction endpoint

**Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "predicted_class": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

### `POST /predict/batch`
Batch predictions for multiple inputs
```json
{
  "instances": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5}
  ]
}
```

---

## 🐳 Docker Details

### Multi-Stage Build

```dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Copy only what's needed
FROM python:3.11-slim
WORKDIR /app
RUN useradd -m -u 1000 apiuser
COPY --from=builder /root/.local /home/apiuser/.local
COPY --chown=apiuser:apiuser app/ ./app/
COPY --chown=apiuser:apiuser model/ ./model/
USER apiuser
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Benefits:**
- Smaller final image (only runtime dependencies)
- Non-root user for security
- Layer caching for faster rebuilds
- Production-ready configuration

### Image Size
- **Final image:** ~610 MB
- Includes: Python, FastAPI, scikit-learn, model weights

---

## 📁 Project Structure

```
ml-model-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   └── predict.py        # ML inference logic
├── model/
│   ├── iris_model.joblib # Trained model
│   └── metadata.joblib   # Model metadata
├── Dockerfile            # Production container
├── docker-compose.yml    # Local development
├── requirements.txt      # Python dependencies
├── train.py             # Model training script
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🔒 Production Best Practices

✅ **Security**
- Non-root container user
- Input validation with Pydantic
- No hardcoded secrets
- HTTPS in production

✅ **Reliability**
- Health check endpoints
- Graceful shutdown handling
- Model loaded once (singleton pattern)
- Error handling and logging

✅ **Performance**
- Multi-stage Docker builds
- Async FastAPI endpoints
- Model kept in memory
- Efficient serialization (joblib)

✅ **Maintainability**
- Type hints throughout
- Auto-generated API documentation
- Version-pinned dependencies
- Clear project structure

---

## 🚀 Deployment

### Render (Current)

**Automatic deployment from GitHub:**
1. Push to `main` branch
2. Render detects changes
3. Builds Docker image
4. Deploys to production
5. Health checks validate deployment

**Configuration:**
- Build command: Auto-detected (Dockerfile)
- Start command: Defined in Dockerfile
- Port: 8080
- Plan: Free tier

### Kubernetes Deployment (Local)

**Status:** ✅ Deployed to Docker Desktop Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment
kubectl get all -n ml-model-api

# Access API
curl http://localhost/health
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Current Setup:**
- 3 replicas for high availability
- LoadBalancer service at `localhost:80`
- Health checks with automatic restarts
- Rolling updates for zero-downtime deployments

### Alternative Platforms

This project can deploy to:
- **Kubernetes** (Local: Docker Desktop ✅ | Cloud: GKE, EKS, AKS)
- **AWS ECS/Fargate** (Docker native)
- **Google Cloud Run** (Serverless containers)
- **Azure Container Instances**
- **Railway** (Similar to Render)
- **Fly.io** (Edge deployment)

---

## 📈 Monitoring & Observability

### Available Metrics
- Request count (access logs)
- Latency per endpoint
- Model prediction distribution
- Error rates
- Health check status

### Logs
```bash
# View Render logs
render logs <service-name>

# Local logs
docker-compose logs -f

# Structured logging format
2025-10-06 15:41:04 - app.main - INFO - Predicting for features: [5.1, 3.5, 1.4, 0.2]
2025-10-06 15:41:04 - app.main - INFO - Prediction: setosa (confidence: 1.0000)
```

---

## 🧪 Testing

```bash
# Unit tests (future enhancement)
pytest tests/

# Manual API testing
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Load testing
ab -n 1000 -c 10 http://localhost:8080/health
```

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **ML Engineering**
   - Model training and evaluation
   - Model serialization and versioning
   - Inference optimization

2. **API Development**
   - RESTful API design
   - Request validation
   - Auto-generated documentation
   - Error handling

3. **DevOps/MLOps**
   - Docker containerization
   - Multi-stage builds
   - Cloud deployment
   - CI/CD with GitHub

4. **Production Skills**
   - Security best practices
   - Monitoring and logging
   - Health checks
   - Scalable architecture

---

## 🔮 Future Enhancements

- [x] Deploy to Kubernetes (Docker Desktop) ✅
- [ ] Deploy to cloud Kubernetes (GKE/EKS/AKS)
- [ ] Add HorizontalPodAutoscaler for load-based scaling
- [ ] Add comprehensive test suite (pytest)
- [ ] Implement model versioning (A/B testing)
- [ ] Add Prometheus metrics endpoint
- [ ] Set up Grafana dashboard
- [ ] Add data drift detection
- [ ] Implement request caching (Redis)
- [ ] Create CI/CD pipeline (GitHub Actions)
- [ ] Add model retraining pipeline
- [ ] Implement rate limiting
- [ ] Add authentication (API keys)
- [ ] Add model explainability (SHAP)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Regan Milne**
- GitHub: [@Regan-Milne](https://github.com/Regan-Milne)
- Email: reganmilne@gmail.com

---

## 🙏 Acknowledgments

- **Iris Dataset:** Fisher, R.A. (1936)
- **FastAPI:** Sebastián Ramírez
- **scikit-learn:** Pedregosa et al. (2011)
- **Docker:** Docker Inc.
- **Render:** Cloud hosting platform

---

## 📚 Related Projects

Part of the **MLOps Portfolio Series:**

1. ✅ **ML Model API** (This project) - Basic deployment + Kubernetes
2. ⏳ **Cloud Kubernetes** - GKE/EKS deployment with autoscaling
3. ⏳ **CI/CD Pipeline** - Automated testing and deployment
4. ⏳ **Monitoring Dashboard** - Prometheus + Grafana

---

**Built with ❤️ as part of an MLOps learning journey**

*Last updated: October 6, 2025*
