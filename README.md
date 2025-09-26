# 🚀 Spam Detection ML System with MLOps

A production-ready spam classification system demonstrating end-to-end MLOps practices. This project showcases how to build, deploy, and monitor machine learning systems with proper engineering principles.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-purple)

## 📋 Project Overview

This project implements a classic spam detection system that classifies SMS messages as "spam" or "ham" (not spam). The interesting part isn't in the model itself, but the **MLOps infrastructure** that makes it production-ready.

### 🎯 Key Features

- **🔧 End-to-End Pipeline**: Data fetching → preprocessing → training → serving → monitoring
- **📊 Experiment Tracking**: MLflow for model versioning and performance comparison
- **⚙️ Workflow Orchestration**: Prefect for reliable pipeline execution
- **🚀 Production API**: FastAPI with Docker containerization
- **📈 Monitoring Ready**: Infrastructure for data drift and model monitoring

## 🏗️ System Architecture

Data Source → Prefect Pipeline → MLflow Tracking → FastAPI → Docker Container

↓ ↓ ↓ ↓ ↓

SMS Dataset Data Validation Experiment REST API Containerized
& Processing Management Deployment

## 📁 Project Structure
spam_mlops_project/
├── data/ # Processed datasets
├── models/ # Trained models and vectorizers
├── pipelines/ # Prefect data pipelines
│ └── data_pipeline.py
├── training/ # Model training and experiment tracking
│ └── train.py
├── serving/ # FastAPI application and Docker setup
│ ├── app/
│ │ ├── api.py
│ │ └── model_handler.py
│ ├── Dockerfile
│ └── requirements.txt
├── monitoring/ # Data drift and model monitoring
├── requirements.txt # Project dependencies
└── README.md

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerization, not totally implemented yet!)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spam-mlops-project.git
   cd spam-mlops-project
   ```
2. Set up Python environment

    ```python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run the data pipeline

    ```
    python pipelines/data_pipeline.py
    ```

4. Train the models

    ```
    python training/train.py
    ```

5. Start the API server

    ```
    cd serving
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
    ```

6. Access the API

    ```
    API Documentation: http://localhost:8000/docs
    Health Check: http://localhost:8000/health
    ```

## 📊 API Usage

### Single Prediction

    ```
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"message": "WIN FREE PRIZE MONEY NOW!!!"}'
    ```

Response:

    ```
    {
    "prediction": 1,
    "probability": 0.95,
    "message": "spam"
    }
    ```

### Batch Prediction

    ```
    curl -X POST "http://localhost:8000/batch_predict" \
        -H "Content-Type: application/json" \
        -d '{"messages": ["Hello there", "Free money now", "Meeting tomorrow"]}'
    ```

### Health Check

    ```
    curl http://localhost:8000/health
    ```