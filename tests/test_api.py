import pytest
from fastapi.testclient import TestClient
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

@pytest.fixture(scope="module")
def client():
    # Utilizing TestClient as a context manager triggers the FastAPI startup event
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Test that health check endpoint returns running status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_model_info(client):
    """
    Test that model info endpoint returns valid model details.
    """
    response = client.get("/model")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["name"] == "XGBoost"
    assert "rmse" in json_data
    assert "r2" in json_data
    assert "trained" in json_data

def test_predict_valid_low_demand(client):
    """
    Test predict endpoint on a valid weekday request without promotions.
    """
    payload = {
        "store": 15,
        "date": "2015-08-01",  # Saturday
        "promo": 0,
        "state_holiday": "0",
        "school_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_sales" in json_data
    assert json_data["predicted_sales"] >= 0.0
    assert json_data["confidence"] in ["High", "Medium"]
    assert json_data["demand_level"] in ["High", "Normal", "Low"]
    assert "inventory_recommendation" in json_data

def test_predict_valid_high_demand(client):
    """
    Test predict endpoint on a valid request with active promotions.
    """
    payload = {
        "store": 15,
        "date": "2015-08-03",  # Monday (with Promo active)
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_sales" in json_data
    assert json_data["predicted_sales"] >= 0.0
    assert json_data["confidence"] in ["High", "Medium"]
    assert json_data["demand_level"] in ["High", "Normal", "Low"]
    assert "inventory_recommendation" in json_data

def test_predict_invalid_store(client):
    """
    Test that an invalid store ID (ge=1) returns a 422 validation error.
    """
    payload = {
        "store": 0,
        "date": "2015-08-01",
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_date(client):
    """
    Test that an invalid date string returns a 422 validation error.
    """
    payload = {
        "store": 15,
        "date": "not-a-date",
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_promo(client):
    """
    Test that an invalid promo value returns a 422 validation error.
    """
    payload = {
        "store": 15,
        "date": "2015-08-01",
        "promo": 3,
        "state_holiday": "0",
        "school_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_holiday(client):
    """
    Test that an invalid state_holiday string returns a 422 validation error.
    """
    payload = {
        "store": 15,
        "date": "2015-08-01",
        "promo": 1,
        "state_holiday": "invalid-holiday",
        "school_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
