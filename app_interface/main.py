from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

import os
# print(f'Current directory: {os.getcwd()}')


# FastAPI endpoint accessible on EC2 instance: http://ec2-18-219-112-73.us-east-2.compute.amazonaws.com:8000/docs#
skyflow_model = joblib.load("../model_artifacts/best_xg_pipeline.pkl")
skyflow = FastAPI()

class InputVars(BaseModel):
    MONTH: float
    DAY_OF_WEEK: float
    DISTANCE: float
    DISTANCE_GROUP_DESC: float
    SEGMENT_NUMBER: int
    CONCURRENT_FLIGHTS: int
    NUMBER_OF_SEATS: int
    CARRIER_NAME: float
    AIRLINE_FLIGHTS_MONTH: int
    AIRLINE_AIRPORT_FLIGHTS_MONTH: int
    AVG_MONTHLY_PASS_AIRPORT: int
    FLT_ATTENDANTS_PER_PASS: float
    GROUND_SERV_PER_PASS: float
    PLANE_AGE: int
    DEPARTING_AIRPORT: float
    LATITUDE: float
    LONGITUDE: float
    PREVIOUS_AIRPORT: float
    PREVIOUS_DURATION: float
    PRCP: float
    SNOW: float
    SNWD: float
    TMAX: float
    AWND: float
    CARRIER_HISTORICAL: float
    DEP_AIRPORT_HIST: float
    PREV_AIRPORT_HIST: float
    DAY_HISTORICAL: float
    DEP_BLOCK_HIST: float
    SEASON: float
    DEP_PART_OF_DAY: float
    ARR_PART_OF_DAY: float
    FLIGHT_DURATION: float
    FLIGHT_DURATION_CATEGORY: float
    PREVIOUS_DURATION: float
    PREVIOUS_DURATION_CATEGORY: float
    PREVIOUS_ARR_DELAY: float
    PREVIOUS_DISTANCE: float

@skyflow.get("/", response_class=HTMLResponse)
def prime_skyflow():
    html_content = """
   <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SkyFlow: AI-Powered Flight Delay Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet"> <!-- Catchy font -->
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-image: url('https://drive.google.com/uc?export=view&id=1BoTupCmSDRoyQj0PsmT7X3w2swMKhKT4'); 
                background-size: cover; 
                background-position: center; 
                background-repeat: no-repeat; 
            }
            .container {
                text-align: center;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: absolute; /* Changed to absolute for top centering */
                top: 10%; /* Position from the top of the page */
                left: 50%;
                transform: translateX(-50%);
                width: 80%;
                max-width: 600px;
            }
            h1 {
                font-family: 'Pacifico', cursive; /* Catchy font for the title */
                color: #003366;
                font-size: 3rem; /* Increased size for emphasis */
                margin-bottom: 1rem;
            }
            p {
                color: #0066cc;
                font-size: 1.5rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SkyFlow</h1>
            <p>An AI-Powered Flight Delay Predictor</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@skyflow.post("/predict")
def predict_flight_delays(predictors: InputVars):
    pred_input = predictors.model_dump()
    print(pred_input)
    prediction = skyflow_model.predict(pd.DataFrame([pred_input]))
    print(prediction)
    return int(prediction)




