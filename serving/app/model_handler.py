# serving/app/model_handler.py
import joblib
import os
from typing import Any

class ModelHandler:
    def __init__(self, model_path: str = "./models/best_model.pkl", 
                 vectorizer_path: str = "./models/tfidf_vectorizer.pkl"):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.load_model()
    
    def load_model(self):
        """Load the model and vectorizer from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            if not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")
            
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Model artifacts loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str) -> dict:
        """Predict if a text is spam."""
        try:
            # Transform text
            text_tfidf = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.model.predict(text_tfidf)[0]
            probability = self.model.predict_proba(text_tfidf)[0][1]
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "label": "spam" if prediction == 1 else "ham"
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_type": type(self.model).__name__,
            "model_params": self.model.get_params() if hasattr(self.model, 'get_params') else {},
            "vectorizer_type": type(self.vectorizer).__name__,
            "feature_count": len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else 0
        }