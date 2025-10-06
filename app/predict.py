import joblib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None
    _model = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def load_model(self):
        if self._model is not None:
            logger.info("Model already loaded")
            return
        
        model_path = Path("model/iris_model.joblib")
        metadata_path = Path("model/metadata.joblib")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self._model = joblib.load(model_path)
        
        if metadata_path.exists():
            self._metadata = joblib.load(metadata_path)
            logger.info(f"Loaded metadata: {self._metadata}")
        else:
            logger.warning("Metadata file not found")
            self._metadata = {}
        
        logger.info("Model loaded successfully")
    
    def predict(self, features: list[float]) -> tuple[str, float, dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        X = np.array(features).reshape(1, -1)
        
        prediction = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        
        class_names = self._metadata.get("target_names", ["class_0", "class_1", "class_2"])
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        probs_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(class_names, probabilities)
        }
        
        return predicted_class, confidence, probs_dict
    
    def predict_batch(self, features_list: list[list[float]]) -> list[tuple[str, float, dict]]:
        return [self.predict(features) for features in features_list]
    
    def get_metadata(self) -> dict:
        return self._metadata if self._metadata else {}
    
    def is_loaded(self) -> bool:
        return self._model is not None

model_service = ModelService()
