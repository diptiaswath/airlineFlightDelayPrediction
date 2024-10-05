import streamlit as st
import requests

# SkyFlow is a StreamLit Application deployed on EC2 instance ec2-18-219-112-73.us-east-2.compute.amazonaws.com
# Access SkyFlow at http://ec2-18-219-112-73.us-east-2.compute.amazonaws.com:8051
st.title("SkyFlow: AI-Powered Flight Delay Predictor")

# HTML content
st.markdown(
    """
    <style>
    .title {
        font-size: 24px;
        color: #4CAF50; /* Green color */
        font-weight: bold;
    }
    .description {
        font-size: 18px;
        color: #555; /* Darker gray */
    }
    </style>

    <div class="description">
        <p>Our AI-Powered Flight Delay Predictor analyzes various factors to predict flight delays.
        Simply enter the required information, and let our model do the rest!</p>
    </div>
    """,
    unsafe_allow_html=True
)

user_inputs_dict = dict()

for field in ['MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'DISTANCE_GROUP_DESC', 'SEGMENT_NUMBER',
              'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS', 'CARRIER_NAME', 'AIRLINE_FLIGHTS_MONTH',
              'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AVG_MONTHLY_PASS_AIRPORT', 'FLT_ATTENDANTS_PER_PASS',
              'GROUND_SERV_PER_PASS', 'PLANE_AGE', 'DEPARTING_AIRPORT', 'LATITUDE', 'LONGITUDE',
              'PREVIOUS_AIRPORT', 'PREVIOUS_DURATION', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND',
              'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST', 'PREV_AIRPORT_HIST', 'DAY_HISTORICAL', 
              'DEP_BLOCK_HIST', 'SEASON', 'DEP_PART_OF_DAY', 'ARR_PART_OF_DAY', 'FLIGHT_DURATION', 
              'FLIGHT_DURATION_CATEGORY', 'PREVIOUS_DURATION_CATEGORY', 
              'PREVIOUS_ARR_DELAY', 'PREVIOUS_DISTANCE']:
    user_inputs_dict[field] = st.number_input(field)

# FastAPI server deployed locally
# response = requests.post("http://127.0.0.1:8000/predict", json = user_inputs_dict)

# FastAPI server deployed on EC2 instance 
response = requests.post("http://18.219.112.73:8000/predict", json = user_inputs_dict)

if st.button("Submit"):
    if response.text == '2':
        st.error("Both departure and arrival time of the flight are delayed")
    elif response.text == '1':
        st.warning("Flight departure is delayed but arrival is on-time")
    else:
        st.success("Flight is on-time")