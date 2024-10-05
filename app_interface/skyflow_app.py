import streamlit as st
import requests
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")
import os

# SkyFlow is a StreamLit Application deployed on EC2 instance
# http://ec2-18-219-112-73.us-east-2.compute.amazonaws.com:8501
st.set_page_config(page_title="SkyFlow: AI-Powered Flight Delay Predictor", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .custom-title {
        font-size: 40px; 
        color: #1E90FF; 
        font-weight: bold;
        text-align: left; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); 
    }
    .stApp {
        background-image: url('https://skyflow.com/replace-later.jpg'); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .description {
        font-size: 18px;
        color: #555;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="custom-title">✈️ SkyFlow: Navigate Your Flight Delays with AI Precision!!</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="description">
        <p>Our AI-Powered Flight Delay Predictor analyzes various factors to predict flight delays.
        Simply enter the required information, and let our model do the rest!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create two main columns for the layout
left_column, right_column = st.columns([1, 2])

# Left column for inputs
with left_column:
    st.header("Share Your Flight Journey!")

    # Define input fields with appropriate types and ranges
    user_inputs_dict = {}

    # Categorical inputs
    categorical_fields = {
        'MONTH': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        'DAY_OF_WEEK': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'SEASON': ['Winter', 'Spring', 'Summer', 'Fall'],
        'DEP_PART_OF_DAY': ['Early Morning & Late Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
        'ARR_PART_OF_DAY': ['Early Morning & Late Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
        'DISTANCE_GROUP_DESC': ['Very Short Distance', 'Short Distance', 'Moderate Distance', 'Moderate to Long Distance',
                                 'Long Distance', 'Very Long Distance', 'Extended Distance', 'Far Distance', 'Distant Location',
                                 'Remote Distance', 'Very Remote Distance'],
        'PREVIOUS_DURATION_CATEGORY': ['Short', 'Medium', 'Long'],
        'FLIGHT_DURATION_CATEGORY': ['Short', 'Medium', 'Long']
    }

    # Numeric inputs
    numeric_fields = [
        'DISTANCE', 'SEGMENT_NUMBER', 'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS', 'AIRLINE_AIRPORT_FLIGHTS_MONTH',
        'AIRLINE_FLIGHTS_MONTH', 
        'AVG_MONTHLY_PASS_AIRPORT', 'FLT_ATTENDANTS_PER_PASS', 'GROUND_SERV_PER_PASS', 'PLANE_AGE',
        'LATITUDE', 'LONGITUDE', 'PREVIOUS_DURATION', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND',
        'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST', 'PREV_AIRPORT_HIST', 'DAY_HISTORICAL', 
        'DEP_BLOCK_HIST', 'FLIGHT_DURATION', 'PREVIOUS_ARR_DELAY', 'PREVIOUS_DISTANCE'
    ]

    for field, options in categorical_fields.items():
        user_inputs_dict[field] = st.selectbox(field, options)

    for field in numeric_fields:
        user_inputs_dict[field] = st.number_input(field, step=0.1)

    # Get directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get parent directory
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # Add airports and airlines selection
    airports_csv_path = os.path.join(parent_dir, 'raw_data', 'airports_list.csv')
    print(f"Attempting to read CSV from: {airports_csv_path}")
    airlines_csv_path = os.path.join(parent_dir, 'raw_data', 'CARRIER_DECODE.csv')
    print(f"Attempting to read CSV from: {airlines_csv_path}")
    
    # Load csvs
    try:
        airports = pd.read_csv(airports_csv_path)
        airlines = pd.read_csv(airlines_csv_path) 
        print("Columns in airports DataFrame:", airports.columns)
        print("Columns in airlines DataFrame:", airlines.columns)
    except FileNotFoundError as e:
        st.error(f"Error: Could not find either airlines or airports_list.csv file. {str(e)}")
        st.stop()
   
    user_inputs_dict['DEPARTING_AIRPORT'] = st.selectbox('DEPARTING_AIRPORT', airports['DISPLAY_AIRPORT_NAME'])
    user_inputs_dict['PREVIOUS_AIRPORT'] = st.selectbox('PREVIOUS_AIRPORT', airports['DISPLAY_AIRPORT_NAME'])
    user_inputs_dict['CARRIER_NAME'] = st.selectbox('CARRIER_NAME', airlines['CARRIER_NAME'])

    predict_button = st.button("Predict Flight Delay")

# Right column for results
with right_column:
    st.header("Flight Delay Insights Just for You!")

    if predict_button:
        try:
            # FastAPI server deployed to local instance
            # response = requests.post("http://127.0.0.1:8000/predict", json=user_inputs_dict)

            # FastAPI server deployed on EC2 instance 
            print(json.dumps(user_inputs_dict, indent=2))
            response = requests.post("http://ec2-18-219-112-73.us-east-2.compute.amazonaws.com:8000/predict", json=user_inputs_dict)
            
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            
            result = response.json()
            
            if response.text == '2':
                st.error("⚠️ Both departure and arrival time of the flight are likely to be delayed.")
            elif response.text == '1':
                st.warning("🕒 Either Flight departure or Flight arrival is predicted to be delayed.")
            else:
                st.success("✅ Flight is predicted to be on-time for both departure and arrival.")
            
            # TODO: Add more details or visualizations based on the prediction
            
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while making the prediction: {str(e)}")
    else:
        st.info("Share your Flight Journey and click 'Predict Flight Delay' to see results.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 16px; color: #4CAF50;">
        <p>✨ Developed with passion by <strong>Dipti Aswath</strong></p>
        <p>🔗 Explore the code on my <a href="https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/README.md" style="color: #1E90FF;">GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)