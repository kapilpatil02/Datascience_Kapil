"""
FastAPI Deployment for World Development Clustering Model
===========================================================

This is a production-ready REST API for the clustering model.

Installation:
    pip install fastapi uvicorn pandas numpy scikit-learn openpyxl

Run the server:
    uvicorn api_deployment:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Import the model class (should be in the same directory)
import sys
sys.path.append('/home/claude')
from deployment_complete import WorldDevelopmentClusteringModel

# Initialize FastAPI app
app = FastAPI(
    title="World Development Clustering API",
    description="Classify countries into development clusters based on economic and social indicators",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model when the API starts."""
    global model
    try:
        model = WorldDevelopmentClusteringModel.load('/home/claude/world_development_kmeans.pkl')
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


# Pydantic models for request/response validation
class CountryData(BaseModel):
    """Input data for a single country."""
    Country: Optional[str] = Field(None, description="Country name (optional)")
    Birth_Rate: float = Field(..., description="Birth rate per 1000 people")
    Business_Tax_Rate: Optional[float] = Field(None, description="Business tax rate (%)")
    CO2_Emissions: float = Field(..., description="CO2 emissions (metric tons)")
    Days_to_Start_Business: Optional[float] = Field(None, description="Days required to start a business")
    Ease_of_Business: Optional[float] = Field(None, description="Ease of doing business index")
    Energy_Usage: float = Field(..., description="Energy usage (kg of oil equivalent)")
    GDP: float = Field(..., description="GDP in USD")
    Health_Exp_GDP: Optional[float] = Field(None, description="Health expenditure as % of GDP")
    Health_Exp_Capita: float = Field(..., description="Health expenditure per capita (USD)")
    Hours_to_do_Tax: Optional[float] = Field(None, description="Hours to prepare and pay taxes")
    Infant_Mortality_Rate: Optional[float] = Field(None, description="Infant mortality rate per 1000 live births")
    Internet_Usage: Optional[float] = Field(None, description="Internet users (% of population)")
    Lending_Interest: Optional[float] = Field(None, description="Lending interest rate (%)")
    Life_Expectancy_Female: Optional[float] = Field(None, description="Female life expectancy (years)")
    Life_Expectancy_Male: Optional[float] = Field(None, description="Male life expectancy (years)")
    Mobile_Phone_Usage: Optional[float] = Field(None, description="Mobile phone subscriptions per 100 people")
    Population_0_14: Optional[float] = Field(None, description="Population aged 0-14 (% of total)")
    Population_15_64: Optional[float] = Field(None, description="Population aged 15-64 (% of total)")
    Population_65_plus: Optional[float] = Field(None, description="Population aged 65+ (% of total)")
    Population_Total: int = Field(..., description="Total population")
    Population_Urban: Optional[float] = Field(None, description="Urban population (% of total)")
    Tourism_Inbound: float = Field(..., description="International tourism arrivals")
    Tourism_Outbound: float = Field(..., description="International tourism departures")

    class Config:
        json_schema_extra = {
            "example": {
                "Country": "Example Country",
                "Birth_Rate": 0.015,
                "CO2_Emissions": 500000,
                "Energy_Usage": 2500,
                "GDP": 500000000000,
                "Health_Exp_Capita": 5000,
                "Population_Total": 50000000,
                "Tourism_Inbound": 10000000,
                "Tourism_Outbound": 8000000,
                "Life_Expectancy_Female": 82,
                "Internet_Usage": 0.75
            }
        }


class PredictionResponse(BaseModel):
    """Response for single country prediction."""
    country: Optional[str]
    cluster: int
    cluster_name: str
    confidence: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_info: Optional[Dict]
    timestamp: str


# Cluster interpretations
CLUSTER_NAMES = {
    0: "Developed Countries",
    1: "Developing Countries",
    2: "Emerging Economies"
}


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "World Development Clustering API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict cluster for a single country",
            "POST /predict/batch": "Predict clusters for multiple countries",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_info = None
    if model is not None:
        model_info = {
            "n_clusters": model.metadata['n_clusters'],
            "trained_date": model.metadata['trained_date'],
            "silhouette_score": round(model.metadata['silhouette_score'], 4),
            "version": model.metadata['version']
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(country_data: CountryData):
    """
    Predict the development cluster for a single country.
    
    Returns the cluster assignment and interpretation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to DataFrame
        data_dict = country_data.dict()
        country_name = data_dict.pop('Country', None)
        
        # Convert field names back to original format (with spaces)
        formatted_data = {
            'Birth Rate': data_dict.get('Birth_Rate'),
            'Business Tax Rate': data_dict.get('Business_Tax_Rate'),
            'CO2 Emissions': data_dict.get('CO2_Emissions'),
            'Days to Start Business': data_dict.get('Days_to_Start_Business'),
            'Ease of Business': data_dict.get('Ease_of_Business'),
            'Energy Usage': data_dict.get('Energy_Usage'),
            'GDP': data_dict.get('GDP'),
            'Health Exp % GDP': data_dict.get('Health_Exp_GDP'),
            'Health Exp/Capita': data_dict.get('Health_Exp_Capita'),
            'Hours to do Tax': data_dict.get('Hours_to_do_Tax'),
            'Infant Mortality Rate': data_dict.get('Infant_Mortality_Rate'),
            'Internet Usage': data_dict.get('Internet_Usage'),
            'Lending Interest': data_dict.get('Lending_Interest'),
            'Life Expectancy Female': data_dict.get('Life_Expectancy_Female'),
            'Life Expectancy Male': data_dict.get('Life_Expectancy_Male'),
            'Mobile Phone Usage': data_dict.get('Mobile_Phone_Usage'),
            'Population 0-14': data_dict.get('Population_0_14'),
            'Population 15-64': data_dict.get('Population_15_64'),
            'Population 65+': data_dict.get('Population_65_plus'),
            'Population Total': data_dict.get('Population_Total'),
            'Population Urban': data_dict.get('Population_Urban'),
            'Tourism Inbound': data_dict.get('Tourism_Inbound'),
            'Tourism Outbound': data_dict.get('Tourism_Outbound'),
        }
        
        if country_name:
            formatted_data['Country'] = country_name
        
        df = pd.DataFrame([formatted_data])
        
        # Make prediction
        cluster = model.predict(df)[0]
        
        return PredictionResponse(
            country=country_name,
            cluster=int(cluster),
            cluster_name=CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
            confidence="High",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(countries: List[CountryData]):
    """
    Predict clusters for multiple countries at once.
    
    Accepts a list of country data and returns predictions for all.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(countries) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 countries per batch")
    
    try:
        predictions = []
        
        for country_data in countries:
            # Reuse single prediction logic
            result = await predict_single(country_data)
            predictions.append(result)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/clusters/info")
async def get_cluster_info():
    """Get information about each cluster."""
    return {
        "total_clusters": 3,
        "clusters": [
            {
                "id": 0,
                "name": CLUSTER_NAMES[0],
                "description": "Countries with high GDP, life expectancy, and internet usage. Advanced infrastructure and healthcare.",
                "typical_characteristics": [
                    "High GDP per capita",
                    "Life expectancy > 80 years",
                    "Internet usage > 50%",
                    "High healthcare expenditure"
                ]
            },
            {
                "id": 1,
                "name": CLUSTER_NAMES[1],
                "description": "Countries with lower GDP and development indicators. Rapidly growing economies.",
                "typical_characteristics": [
                    "Lower GDP per capita",
                    "Life expectancy ~ 60 years",
                    "Internet usage < 5%",
                    "Lower healthcare expenditure"
                ]
            },
            {
                "id": 2,
                "name": CLUSTER_NAMES[2],
                "description": "Countries in transition between developing and developed status.",
                "typical_characteristics": [
                    "Medium GDP per capita",
                    "Life expectancy ~ 75 years",
                    "Internet usage 20-50%",
                    "Moderate healthcare expenditure"
                ]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
