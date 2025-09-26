# serving/app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List, Dict
import numpy as np

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction: int  # 0 = ham, 1 = spam
    probability: float
    message: str

class BatchPredictionRequest(BaseModel):
    messages: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]

# Load model and vectorizer
def load_artifacts():
    """Load the trained model and vectorizer."""
    try:
        model_path = "../models/best_model.pkl"
        vectorizer_path = "../models/tfidf_vectorizer.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("Model and vectorizer loaded successfully")
        return model, vectorizer
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        raise

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="A machine learning API to classify messages as spam or ham",
    version="1.0.0"
)

# Load artifacts when the application starts
model, vectorizer = load_artifacts()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Spam Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict if a single message is spam or ham."""
    try:
        # Transform the message using the trained vectorizer
        message_tfidf = vectorizer.transform([request.message])
        
        # Make prediction
        prediction = model.predict(message_tfidf)[0]
        probability = model.predict_proba(message_tfidf)[0][1]  # Probability of spam
        
        # Determine human-readable label
        label = "spam" if prediction == 1 else "ham"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            message=label
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict spam/ham for multiple messages at once."""
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Transform all messages
        messages_tfidf = vectorizer.transform(request.messages)
        
        # Make predictions
        predictions = model.predict(messages_tfidf)
        probabilities = model.predict_proba(messages_tfidf)[:, 1]  # Spam probabilities
        
        # Prepare response
        results = []
        for i, message in enumerate(request.messages):
            results.append({
                "message": message,
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "label": "spam" if predictions[i] == 1 else "ham"
            })
        
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_type": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, 'get_params') else "Not available",
        "feature_count": len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else "Unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)