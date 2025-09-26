import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
from prefect import flow, task
import os

@task
def load_data(data_path: str = "./data/processed/"):
    """Load the processed train/validation/test data."""
    train_df = pd.read_csv(f"{data_path}/train.csv")
    val_df = pd.read_csv(f"{data_path}/val.csv") 
    test_df = pd.read_csv(f"{data_path}/test.csv")
    
    print(f"Loaded data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    return train_df, val_df, test_df

@task
def prepare_features(train_df, val_df, test_df):
    """Prepare TF-IDF features and labels for training."""
    
    # Prepare text data
    X_train_text = train_df['message'].values
    X_val_text = val_df['message'].values
    X_test_text = test_df['message'].values
    
    # Prepare labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000,  
        stop_words='english',  
        ngram_range=(1, 2)
    )
    
    # Fit on training data
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer

@task
def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }
    
    print(f"{model_name} Validation Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return metrics

@task
def train_logistic_regression(X_train, y_train, X_val, y_val, hyperparams):
    """Train and evaluate a Logistic Regression model."""
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Train model
        model = LogisticRegression(**hyperparams)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, "Logistic Regression")
        
        # Log metrics to MLflow
        for metric, value in metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics

@task
def train_random_forest(X_train, y_train, X_val, y_val, hyperparams):
    """Train and evaluate a Random Forest model."""
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Train model
        model = RandomForestClassifier(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, "Random Forest")
        
        # Log metrics to MLflow
        for metric, value in metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics

@flow(name="spam_classifier_training")
def training_pipeline(data_path: str = "./data/processed/"):
    """Main training pipeline with experiment tracking."""
    
    # Set up MLflow
    mlflow.set_experiment("Spam_Detection")
    
    print("🚀 Starting Spam Classification Training Pipeline")
    print("=" * 50)
    
    with mlflow.start_run():
        # Load and prepare data
        train_df, val_df, test_df = load_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = prepare_features(train_df, val_df, test_df)
        
        # Define hyperparameter combinations to try
        lr_hyperparams = [
            {'C': 0.1, 'max_iter': 1000},
            {'C': 1.0, 'max_iter': 1000},
            {'C': 10.0, 'max_iter': 1000}
        ]
        
        rf_hyperparams = [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 20},
            {'n_estimators': 100, 'max_depth': None}
        ]
        
        best_score = 0
        best_model = None
        best_model_name = ""
        
        # Train multiple Logistic Regression models
        print("Training Logistic Regression models...")
        for i, params in enumerate(lr_hyperparams):
            print(f"   Experiment {i+1}: {params}")
            model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val, params)
            
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = model
                best_model_name = f"LogisticRegression_C{params['C']}"
        
        # Train multiple Random Forest models  
        print("Training Random Forest models...")
        for i, params in enumerate(rf_hyperparams):
            print(f"   Experiment {i+1}: {params}")
            model, metrics = train_random_forest(X_train, y_train, X_val, y_val, params)
            
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = model
                best_model_name = f"RandomForest_est{params['n_estimators']}"
        
        # Log best model info
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_f1_score", best_score)
        
        print("=" * 50)
        print(f"Best Model: {best_model_name}")
        print(f"Best F1 Score: {best_score:.4f}")
        
        # Save the best model
        os.makedirs("./models", exist_ok=True)
        import joblib
        joblib.dump(best_model, f"./models/{best_model_name}.pkl")
        joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")
        
        print("Best model saved to ./models/")
        
        return best_model, best_score

if __name__ == "__main__":
    training_pipeline()