from fastapi import FastAPI, HTTPException
import json
import os
from app.config import METRICS_PATH
from app.schemas import PredictionRequest, PredictionResponse, ModelInfoResponse, HealthResponse
from app.predictor import SalesPredictor

# Initialize FastAPI App
app = FastAPI(
    title="Rossmann Retail Demand Forecasting Service",
    description="Production-grade ML forecasting API using XGBoost and on-the-fly feature engineering.",
    version="1.0.0"
)

# Global predictor instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    print("Starting ML service initialization...")
    predictor = SalesPredictor()

@app.get("/", response_model=HealthResponse, summary="Health Check")
def health_check():
    return HealthResponse(status="running")

@app.get("/model", response_model=ModelInfoResponse, summary="Get Model Metadata & Metrics")
def get_model_info():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Model metrics file not found.")
        
    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)
        
    return ModelInfoResponse(
        name=metrics_data["name"],
        rmse=metrics_data["rmse"],
        r2=metrics_data["r2"],
        trained=metrics_data["trained"]
    )

@app.post("/predict", response_model=PredictionResponse, summary="Generate Retail Demand Forecast")
def predict_sales(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service is not fully initialized.")
        
    try:
        prediction_results = predictor.predict(request)
        return PredictionResponse(
            predicted_sales=prediction_results["predicted_sales"],
            confidence=prediction_results["confidence"],
            demand_level=prediction_results["demand_level"],
            inventory_recommendation=prediction_results["inventory_recommendation"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
