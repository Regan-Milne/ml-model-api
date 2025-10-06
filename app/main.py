from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse
)
from app.predict import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    try:
        model_service.load_model()
        logger.info("Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="Iris Classification API",
    description="Production ML API for iris species classification using RandomForest",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Iris Classification API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    metadata = model_service.get_metadata()
    
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "unhealthy",
        model_loaded=model_service.is_loaded(),
        model_info={
            "accuracy": metadata.get("accuracy", "unknown"),
            "features": metadata.get("feature_names", []),
            "classes": metadata.get("target_names", []),
            "n_features": metadata.get("n_features", 0)
        }
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        features = [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]
        
        logger.info(f"Predicting for features: {features}")
        
        predicted_class, confidence, probabilities = model_service.predict(features)
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    try:
        features_list = [
            [
                instance.sepal_length,
                instance.sepal_width,
                instance.petal_length,
                instance.petal_width
            ]
            for instance in input_data.instances
        ]
        
        logger.info(f"Batch prediction for {len(features_list)} instances")
        
        results = model_service.predict_batch(features_list)
        
        predictions = [
            PredictionOutput(
                predicted_class=pred_class,
                confidence=conf,
                probabilities=probs
            )
            for pred_class, conf, probs in results
        ]
        
        return BatchPredictionOutput(predictions=predictions)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
