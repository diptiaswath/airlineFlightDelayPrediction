import streamlit as st
import requests
import pandas as pd
import json
import os
import calendar
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, time, date
from geopy.distance import geodesic

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

# Get directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add airports and airlines selection
airports_csv_path = os.path.join(parent_dir, 'raw_data', 'airports_list.csv')
# print(f"Attempting to read CSV from: {airports_csv_path}")
airlines_csv_path = os.path.join(parent_dir, 'raw_data', 'CARRIER_DECODE.csv')
# print(f"Attempting to read CSV from: {airlines_csv_path}")
airport_coords_csv_path = os.path.join(parent_dir, 'raw_data', 'AIRPORT_COORDINATES.csv')
    
# Load csvs and eliminate duplicates along specific columns
try:
    airports = pd.read_csv(airports_csv_path)
    airlines = pd.read_csv(airlines_csv_path) 
    airport_coords = pd.read_csv(airport_coords_csv_path)
   
    airports = airports.drop_duplicates(subset=['DISPLAY_AIRPORT_NAME'], keep='first')
    airlines = airlines.drop_duplicates(subset=['CARRIER_NAME'], keep='first')
    airport_coords = airport_coords.drop_duplicates(subset=['DISPLAY_AIRPORT_NAME'], keep='first')
    
   # print("Columns in airports DataFrame:", airports.columns)
   # print("Columns in airlines DataFrame:", airlines.columns)
   # print("Columns in airport coords DataFrame:", airport_coords.columns)
except FileNotFoundError as e:
    st.error(f"Error: Could not find either airlines or airports_list.csv file. {str(e)}")
    st.stop()
   
# Create two main columns for the layout
left_column, right_column = st.columns([1, 2])

# Left column of inputs
with left_column:
    st.header("Share Your Flight Journey!")

    # Define input fields
    user_inputs_dict = {}

    # First sub-row: Date of Flight and Airline
    st.subheader("Flight Date and Airline")
    col1, col2 = st.columns(2)
    with col1:
        flight_date = st.date_input("Date of Flight:")
    with col2:
        airlines = pd.read_csv(airlines_csv_path)
        user_inputs_dict['AIRLINE'] = st.selectbox('Airline:', airlines['CARRIER_NAME'])

    # Second sub-row: From and To Airport
    st.subheader("Airports")
    airports = pd.read_csv(airports_csv_path)
    col3, col4 = st.columns(2)
    with col3:
        user_inputs_dict['FROM_AIRPORT'] = st.selectbox('From Airport:', airports['DISPLAY_AIRPORT_NAME'])
    with col4:
        user_inputs_dict['TO_AIRPORT'] = st.selectbox('To Airport:', airports['DISPLAY_AIRPORT_NAME'])

    # Third sub-row: Flight Times
    st.subheader("Flight Times")
    col5, col6 = st.columns(2)
    with col5:
        departure_time = st.time_input("Departure Time:")
    with col6:
        arrival_time = st.time_input("Arrival Time:")

    # Predict button (spanning the full width)
    predict_button = st.button("Predict Flight Delay")

# Map user interface inputs to domain or model required features
def map_inputs_to_features(user_inputs, flight_date, departure_time, arrival_time):
    """
    Derive domain features from user inputs where able and hard-code others temporarily.
    This function maps the minimal user interface inputs to the comprehensive set of features
    required by the flight delay prediction model.

    Parameters:
    user_inputs (dict): A dictionary containing user-provided information about the flight.
        Expected keys:
        - 'FROM_AIRPORT': Departure airport name (str)
        - 'TO_AIRPORT': Arrival airport name (str)
        - 'CARRIER_NAME': Airline carrier name (str)
    flight_date (datetime.date): The date of the flight.
    departure_time (datetime.time): The scheduled departure time of the flight.
    arrival_time (datetime.time): The scheduled arrival time of the flight.

    Returns:
    dict: A dictionary containing all the features required by the model, including:
        - MONTH (str): Month of the flight (1-12)
        - DAY_OF_WEEK (str): Day of the week (1-7, where 1 is Monday)
        - SEASON (str): Season of the flight ('Winter', 'Spring', 'Summer', 'Fall')
        - DEP_PART_OF_DAY (str): Part of day for departure
        - ARR_PART_OF_DAY (str): Part of day for arrival
        - FLIGHT_DURATION (float): Duration of the flight in minutes
        - FLIGHT_DURATION_CATEGORY (str): Categorized flight duration ('Short', 'Medium', 'Long')
        - LATITUDE (float): Latitude of the departure airport
        - LONGITUDE (float): Longitude of the departure airport
        - DISTANCE (float): Distance between departure and arrival airports
        - DISTANCE_GROUP_DESC (str): Categorized distance
        - AIRLINE_FLIGHTS_MONTH (int): Estimated number of flights for the airline per month
        - AVG_MONTHLY_PASS_AIRLINE (int): Estimated average monthly passengers for the airline
        - Additional features with placeholder values

    Note:
    This function uses placeholder values for some features that are not directly
    derivable from the user inputs. These should be replaced with more accurate
    data lookups.
    """
    # Dict to store all mapped (or derived) features
    features = {}

    # Map date information
    features['MONTH'] = calendar.month_name[flight_date.month]
    features['DAY_OF_WEEK'] = calendar.day_name[flight_date.weekday() + 1]  # Monday is 1, Sunday is 7
    features['SEASON'] = get_season(flight_date)

    # Map time information
    features['DEP_PART_OF_DAY'] = get_part_of_day(departure_time)
    features['ARR_PART_OF_DAY'] = get_part_of_day(arrival_time)

    # Calculate flight duration
    duration = datetime.combine(flight_date, arrival_time) - datetime.combine(flight_date, departure_time)
    features['FLIGHT_DURATION'] = duration.total_seconds() / 60  # in minutes
    features['FLIGHT_DURATION_CATEGORY'] = categorize_flight_duration(features['FLIGHT_DURATION'])

    # Map airport information
    from_airport_info = get_airport_info(user_inputs['FROM_AIRPORT'], airport_coords)
    to_airport_info = get_airport_info(user_inputs['TO_AIRPORT'], airport_coords)
    features['DEPARTING_AIRPORT'] =  user_inputs['FROM_AIRPORT']
    features['LATITUDE'] = from_airport_info['latitude']
    features['LONGITUDE'] = from_airport_info['longitude']
    features['DISTANCE'] = calculate_distance(from_airport_info, to_airport_info)
    features['DISTANCE_GROUP_DESC'] = categorize_distance(features['DISTANCE'])


    # Map carrier information
    # TODO: Make a airline specific csv to read from 
    #carrier_info = get_carrier_info(user_inputs['AIRLINE'])
    features['CARRIER_NAME'] = user_inputs['AIRLINE']
    #features['AIRLINE_FLIGHTS_MONTH'] = carrier_info['flights_per_month']
    #features['AVG_MONTHLY_PASS_AIRLINE'] = carrier_info['avg_monthly_passengers']

    # Add temp placeholder values for other features 
    # TODO: Replace below hard-coded feature values with more accurate lookups
    features['PREVIOUS_AIRPORT'] = "string"
    features['AIRLINE_FLIGHTS_MONTH'] = 0
    features['SEGMENT_NUMBER'] = 0
    features['CONCURRENT_FLIGHTS'] = 10.0
    features['NUMBER_OF_SEATS'] = 150.0
    features['AIRLINE_AIRPORT_FLIGHTS_MONTH'] = 100.00
    features['AVG_MONTHLY_PASS_AIRPORT'] = 10000.00
    features['FLT_ATTENDANTS_PER_PASS'] = 0.05
    features['GROUND_SERV_PER_PASS'] = 0.1
    features['PLANE_AGE'] = 10.0
    features['PREVIOUS_DURATION'] = 0.0
    features['PREVIOUS_DURATION_CATEGORY'] = categorize_flight_duration(features['PREVIOUS_DURATION'])
    features['PRCP'] = 20.0
    features['SNOW'] = 0.0
    features['SNWD'] = 0.0
    features['TMAX'] = 20.0
    features['AWND'] = 5.0
    features['CARRIER_HISTORICAL'] = 0
    features['DEP_AIRPORT_HIST'] = 0
    features['PREV_AIRPORT_HIST'] = 0
    features['DAY_HISTORICAL'] = 0
    features['DEP_BLOCK_HIST'] = 0
    features['PREVIOUS_ARR_DELAY'] = 0.0
    features['PREVIOUS_DISTANCE'] = 0

    return features

# Return season based on given date
def get_season(date):
    """
    Determine the season based on the given date.
    
    Parameters:
    date (datetime.date): The date for which to determine the season.
    
    Returns:
    str: The season ('Winter', 'Spring', 'Summer', or 'Fall').
    """
    # print(f'In get_season, input is: {date}')
    # Check if input is of datetime type
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    
    # Get month as 1-12
    month = date.month

    # Get season based on month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # month in [9, 10, 11]
        return 'Fall'
    
# Return part of day based on given time
def get_part_of_day(time_obj):
    """
    Determine the part of day based on the given time.
    
    Paramaters:
    time(datetime.time): The time for which to determine the part of day.
    
    Returns:
    str: The part of day ('Early Morning & Late Night', 'Morning', 'Afternoon', 'Evening', or 'Night').
    """  
    # print(f'In get_part_of_day, input is: {time_obj}')
    # Check if input is a time object
    if isinstance(time_obj, str):
        time_obj = datetime.strptime(time_obj, '%H:%M').time()
    
    # Define time ranges for each part of the day
    early_morning_late_night = time(0, 0) <= time_obj < time(6, 0)
    morning = time(6, 0) <= time_obj < time(12, 0)
    afternoon = time(12, 0) <= time_obj < time(17, 0)
    evening = time(17, 0) <= time_obj < time(20, 0)
    night = time(20, 0) <= time_obj <= time(23, 59)
    
    # Determine part of day
    if early_morning_late_night:
        return 'Early Morning & Late Night'
    elif morning:
        return 'Morning'
    elif afternoon:
        return 'Afternoon'
    elif evening:
        return 'Evening'
    elif night:
        return 'Night'
    else:
        return 'Unknown'

# Retrieve Latitude and Longitude given airport name and coords
def get_airport_info(airport_name, airport_coords):
    """
    Get airport information (latitude, longitude) from the airport name.
    
    Parameters:
    airport_name (str): The name of the airport.
    airport_coords (pd.DataFrame): DataFrame containing airport coordinate information.
    
    Returns:
    dict: A dict containing airport information.
    """
    try:
        airport_info = airport_coords[airport_coords['DISPLAY_AIRPORT_NAME'] == airport_name].iloc[0]
        return {
            'airport_name': airport_name,
            'latitude': round(airport_info['LATITUDE'], 3),
            'longitude': round(airport_info['LONGITUDE'], 3)
        }
    except IndexError:
        print(f"Airport '{airport_name}' not found in the database.")
        return None

# Returns computed distance between two airports 
def calculate_distance(from_airport, to_airport):
    """
    Calculate the distance between two airports using their coordinates.
    
    Args:
    from_airport (dict): Dictionary containing latitude and longitude of the departure airport.
    to_airport (dict): Dictionary containing latitude and longitude of the arrival airport.
    
    Returns:
    int: The distance between the two airports in miles, rounded to the nearest integer.
    """
    from_coords = (from_airport['latitude'], from_airport['longitude'])
    to_coords = (to_airport['latitude'], to_airport['longitude'])
    return float(geodesic(from_coords, to_coords).miles)

# Returns distance group descriptors based on distance in miles
def categorize_distance(distance):
    """
    Categorize the distance into distance groups.
    
    Args:
    distance (float): The distance in miles.
    
    Returns:
    str: The distance group description.
    """
    if distance <= 250:
        return 'Very Short Distance'
    elif 250 < distance <= 500:
        return 'Short Distance'
    elif 500 < distance <= 750:
        return 'Moderate Distance'
    elif 750 < distance <= 1000:
        return 'Moderate to Long Distance'
    elif 1000 < distance <= 1250:
        return 'Long Distance'
    elif 1250 < distance <= 1500:
        return 'Very Long Distance'
    elif 1500 < distance <= 1750:
        return 'Extended Distance'
    elif 1750 < distance <= 2000:
        return 'Far Distance'
    elif 2000 < distance <= 2250:
        return 'Distant Location'
    elif 2250 < distance <= 2500:
        return 'Remote Distance'
    else:
        return 'Very Remote Distance'

# Returns flight duration category from duration in mins 
def categorize_flight_duration(duration):
    """
    Categorize flight duration into Short, Medium, or Long.

    Args:
    duration (float): The flight duration in minutes.

    Returns:
    str: The flight duration category.
    """
    if duration < 120:  # Less than 2 hours
        return "Short"
    elif 120 <= duration < 300:  # Between 2 and 5 hours
        return "Medium"
    else:  # 5 hours or more
        return "Long"

# TODO: Returns carrier specific features : AIRLINE_FLIGHTS_MONTH, AVG_MONTHLY_PASS_AIRLINE
def get_carrier_info(carrier_name):
    pass

# Right column for results
with right_column:
    st.header("Flight Delay Insights Just for You!")

    if predict_button:
        try:
            print(json.dumps(user_inputs_dict, indent=2))
            derived_features = map_inputs_to_features(user_inputs_dict, flight_date, departure_time, arrival_time)   
            print(json.dumps(derived_features, indent=2))         
            
            # FastAPI server deployed on EC2 instance 
            response = requests.post("http://ec2-18-219-112-73.us-east-2.compute.amazonaws.com:8000/predict", json = derived_features)
            
            # FastAPI server deployed to local instance 
            # response = requests.post("http://127.0.0.1:8000/predict", json = derived_features) 
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
        <p>✨ Developed by <a href="https://www.linkedin.com/in/dipti-aswath-60b9131/"><strong>Dipti Aswath</strong></a></p>
        <p>🔗 Explore the code on my <a href="https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/README.md" style="color: #1E90FF;">GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)