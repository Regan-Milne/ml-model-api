import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_model():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {iris.target_names}")
    print(f"Features: {iris.feature_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nTraining RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    model_path = "model/iris_model.joblib"
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    metadata = {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "accuracy": float(accuracy),
        "n_features": len(iris.feature_names)
    }
    
    metadata_path = "model/metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"Saving metadata to {metadata_path}...")
    
    print("\nModel training complete!")
    print(f"Model file size: {joblib.load(model_path).__sizeof__()} bytes")
    
    print("\n" + "="*50)
    print("Sample prediction test:")
    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    
    print(f"Input features: {sample[0]}")
    print(f"Predicted class: {iris.target_names[prediction]}")
    print(f"Probabilities: {dict(zip(iris.target_names, probabilities))}")
    print("="*50)

if __name__ == "__main__":
    train_model()
