# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import requests
import pandas as pd
import json
import os
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, time as dt_time, date
import time
from geopy.distance import geodesic
from PIL import Image
from streamlit import session_state as state


# SkyFlow is a StreamLit Application deployed on AWS EC2
st.set_page_config(page_title="SkyFlow", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    /* Block Container Styling */
    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }

    /* Vertical Block Padding */
    [data-testid="stVerticalBlock"] {
        padding-left: 0rem;
        padding-right: 0rem;
    }

    /* Metric Component Styling */
    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 15px 0;
    }

    /* Metric Label Styling */
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Delta Icon Positioning */
    [data-testid="stMetricDeltaIcon-Up"],
    [data-testid="stMetricDeltaIcon-Down"] {
        position: relative;
        left: 38%;
        transform: translateX(-50%);
    }

    /* Custom Title */
    .custom-title {
        font-size: 36px; 
        color: #1E90FF; 
        font-weight: bold;
        text-align: left; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1); 
    }

    /* App Background */
    .stApp {
        background-image: url('https://skyflow.com/replace-later.jpg'); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Description Text */
    .description {
        font-size: 10px;
        color: #555;
        line-height: 2.5;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #555;
        color: white;
    }

    /* Hide Anchor Links for Headers */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<h1 class="custom-title" style="font-family: Arial, sans-serif; color: #1E90FF; line-height: 1.0;">✈️ SkyFlow: Elevating Air Travel Experience ✈️</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description" style="font-family: Arial, sans-serif; line-height: 1.2; padding: 6px; border-radius: 6px;  background-color: #E0E0E0;">
        <h3 style="color: #1E90FF; line-height: 1.0;">AI-Powered Platform</h3>            
        <p style="margin-top: 6px;">SkyFlow utilizes advanced machine learning algorithms and real-time analytics to deliver precise predictions and actionable insights. 
        By leveraging AI, we empower travelers to make informed decisions about flight delays and airline sentiment while optimizing operations for enhanced efficiency, resilience, and customer satisfaction. 
        This holistic approach improves operational performance and enriches the overall travel experience for both airlines and passengers.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add padding before the tabs
st.markdown("<div style='padding: 8px;'></div>", unsafe_allow_html=True)

# Tabs locked
if 'tab_locked' not in state:
    state.tab_locked = True

# Tab Titles
tab_titles = ["Flight Delay Insights Tracker", "Flight Delay Predictor", "Airline Sentiment Analyzer", "Personalized Flight and Trip Planner"]

# Create Tabs
tabs = st.tabs(tab_titles)

# Style the tabs with a custom CSS class 
st.markdown(
    """
    <style>
        .stTabs { 
            font-family: Arial, sans-serif; 
            color: #1E90FF; 
            font-size: 28px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)
###########################################################################################################
# Generic - used across SkyFlow tabs
###########################################################################################################
# Get directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Airports and Airlines selection path
airports_csv_path = os.path.join(parent_dir, 'raw_data', 'airports_list.csv')
# print(f"Attempting to read CSV from: {airports_csv_path}")
airlines_csv_path = os.path.join(parent_dir, 'raw_data', 'CARRIER_DECODE.csv')
# print(f"Attempting to read CSV from: {airlines_csv_path}")
airport_coords_csv_path = os.path.join(parent_dir, 'raw_data', 'AIRPORT_COORDINATES.csv')
# Static Image selection path
flights_image_path = os.path.join(parent_dir, 'images', 'b6cf1189a8363e9708a712a22171e35a.jpeg')
flight_delays_segment_image_path = os.path.join(parent_dir, 'images', 'Delays_by_Segment_Distance_Group.jpeg')
# Historical data (train) path 
prep_hist_data_train_path = os.path.join(parent_dir, 'combined_data', 'train.pkl') 

    
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

# Load historical flight data
try:
    df_train = pd.read_pickle(prep_hist_data_train_path)
except FileNotFoundError as e:
    st.error(f"Error: Could not locate and load historical flight data. {str(e)}")
    st.stop()

############################################################################################################
# Function to prompt for NDA Consult 
############################################################################################################
def locked_tab_content(tab_name):
    st.warning("🔒 Access Restricted")

    st.markdown(
        """
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #ffeeba;">
            <h3 style="color: #856404; font-family: Arial, sans-serif; margin: 0;">Access Locked: NDA Required</h3>
            <p style="color: #856404; font-family: Arial, sans-serif; margin-top: 10px; font-size: 14px;">
                This tab's features are currently restricted. To gain access, a Non-Disclosure Agreement (NDA) consultation is required.  
            </p>
            <h4 style="color: #856404; font-family: Arial, sans-serif; margin-top: 15px;">Next Steps:</h4>
            <ol style="color: #856404; font-family: Arial, sans-serif; font-size: 14px; padding-left: 20px;">
                <li>Visit our contact page <a href="http://kvgrowth.com/contact" target="_blank" style="color: #856404; text-decoration: underline;">here</a>.</li>
                <li>Request an NDA consultation.</li>
                <li>Once approved, you'll receive access to this tab's full functionality.</li>
            </ol>
            <p style="color: #856404; font-family: Arial, sans-serif; margin-top: 10px; font-size: 14px;">
                To ensure security and confidentiality, please use the official channels. If you have any questions, 
                <a href="http://kvgrowth.com" style="color: #856404; text-decoration: none;">feel free to reach out to us</a>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

############################################################################################################
# Functions for Flight Delay Predictor Tab
############################################################################################################
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
    early_morning_late_night = dt_time(0, 0) <= time_obj < dt_time(6, 0)
    morning = dt_time(6, 0) <= time_obj < dt_time(12, 0)
    afternoon = dt_time(12, 0) <= time_obj < dt_time(17, 0)
    evening = dt_time(17, 0) <= time_obj < dt_time(20, 0)
    night = dt_time(20, 0) <= time_obj <= dt_time(23, 59)
    
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



###########################################################################################################
# Function Plots for Flight Delay Insights Tracker
###########################################################################################################
# Display coming soon message
def display_other_trend_message(selected_delay_trend):
    st.info(f"You selected Delay Trends: {selected_delay_trend}. Feature coming soon!")

# Display default message
def display_default_message():
    st.info('Please select a delay trend to visualize!')


# Plot congested airports for selected historical year, where the fetch is cached
@st.cache_data
def get_congested_airports(selected_year_trend):
    # Filter for flights with delays
    delayed_flights = df_train[(df_train['DEP_DELAY'] > 0) | (df_train['ARR_DELAY'] > 0)]

    # Group by departing airport and count the number of delayed flights
    congested_airports = delayed_flights.groupby('DEPARTING_AIRPORT')['CONCURRENT_FLIGHTS'].count().reset_index()

    # Sort by the most congested airports
    congested_airports = congested_airports.sort_values(by='CONCURRENT_FLIGHTS', ascending=False)
    return congested_airports

def plot_congested_airports(selected_year_trend):
    congested_airports = get_congested_airports(selected_year_trend)

    # Add interactive slider for the number of airports to display
    num_airports = st.slider(f'For year {selected_year_trend}, select number of top congested airports to display:', 5, 30, 10)

    # Filter data for visualization
    top_congested_airports = congested_airports.head(num_airports)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='CONCURRENT_FLIGHTS',
        y='DEPARTING_AIRPORT',
        data=top_congested_airports,
        palette='Blues_d',
        ax=ax
    )
    ax.set_title(f'Top {num_airports} Congested Airports with Flight Delays')
    ax.set_xlabel('Number of Concurrent Flights with Delays')
    ax.set_ylabel('Departing Airport')
    st.pyplot(fig)

# Plot Delays by Airline where the fetch is cached
@st.cache_data
def get_avg_delays_by_carrier():
    # Filter data for departure delays
    avg_departure_delay = (
        df_train[df_train['DEP_DELAY'] > 0]
        .groupby('CARRIER_NAME')['DEP_DELAY']
        .mean()
        .sort_values(ascending=False)
    )

    # Filter data for arrival delays
    avg_arrival_delay = (
        df_train[df_train['ARR_DELAY'] > 0]
        .groupby('CARRIER_NAME')['ARR_DELAY']
        .mean()
        .sort_values(ascending=False)
    )
    return avg_departure_delay, avg_arrival_delay

def plot_avg_delays_by_carrier(selected_year_trend):
    # Render title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Departure & Arrival Delays by Airline</h5>',
        unsafe_allow_html=True
    )

    # Fetch processed delay data
    avg_departure_delay, avg_arrival_delay = get_avg_delays_by_carrier()

    # Add interactive slider for number of airlines
    num_airlines = st.slider(
        f'For year {selected_year_trend}, select number of top airlines with delays to display:',
        min_value=5, max_value=20, value=10
    )

    # Filter top airlines for plotting
    top_departure = avg_departure_delay.head(num_airlines)
    top_arrival = avg_arrival_delay.head(num_airlines)

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Average Departure Delay Plot
    sns.barplot(x=top_departure.index, y=top_departure.values, ax=axes[0], palette="Blues_d")
    axes[0].set_title(f'Average Departure Delay by Airline (Top {num_airlines})', fontsize=16)
    axes[0].set_xlabel('Airline', fontsize=12)
    axes[0].set_ylabel('Average Departure Delay (minutes)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=90)

    # Average Arrival Delay Plot
    sns.barplot(x=top_arrival.index, y=top_arrival.values, ax=axes[1], palette="Reds_d")
    axes[1].set_title(f'Average Arrival Delay by Airline (Top {num_airlines})', fontsize=16)
    axes[1].set_xlabel('Airline', fontsize=12)
    axes[1].set_ylabel('Average Arrival Delay (minutes)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout and display plots
    plt.tight_layout()
    st.pyplot(fig)

# Plot Delays by Part of Day
# Cached data processing function
@st.cache_data
def get_avg_delays_by_part_of_day():
    # Filter rows where either DEP_DELAY or ARR_DELAY are greater than 0
    filtered_df = df_train[(df_train['DEP_DELAY'] > 0) | (df_train['ARR_DELAY'] > 0)]

    # Calculate average delays by DEP_PART_OF_DAY
    avg_dep_delay = (
        filtered_df.groupby('DEP_PART_OF_DAY')['DEP_DELAY']
        .mean()
        .reset_index()
        .rename(columns={'DEP_DELAY': 'Average_DEP_DELAY'})
    )

    # Calculate average delays by ARR_PART_OF_DAY
    avg_arr_delay = (
        filtered_df.groupby('ARR_PART_OF_DAY')['ARR_DELAY']
        .mean()
        .reset_index()
        .rename(columns={'ARR_DELAY': 'Average_ARR_DELAY'})
    )
    return avg_dep_delay, avg_arr_delay

def plot_avg_delays_by_part_of_day(selected_year_trend):
    # Render title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Departure & Arrival Delays by Part of Day</h5>',
        unsafe_allow_html=True
    )

    # Fetch preprocessed data
    avg_dep_delay, avg_arr_delay = get_avg_delays_by_part_of_day()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Average Departure Delay',
            f'Average Arrival Delay'
        )
    )

    # Plot Average Departure Delay
    fig.add_trace(
        go.Bar(
            x=avg_dep_delay['DEP_PART_OF_DAY'],
            y=avg_dep_delay['Average_DEP_DELAY'],
            name='Average Departure Delay',
            marker_color='purple'
        ),
        row=1, col=1
    )

    # Plot Average Arrival Delay
    fig.add_trace(
        go.Bar(
            x=avg_arr_delay['ARR_PART_OF_DAY'],
            y=avg_arr_delay['Average_ARR_DELAY'],
            name='Average Arrival Delay',
            marker_color='orange'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Part of Day',
        yaxis_title='Average Departure Delay (minutes)',
        xaxis2_title='Part of Day',
        yaxis2_title='Average Arrival Delay (minutes)',
        showlegend=False,
        template='plotly_white',
        xaxis_tickangle=-90,
        xaxis2_tickangle=-90
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)


# Plot Delays by Day of Week
# Cached data processing function
@st.cache_data
def get_avg_delays_by_day_of_week():
    # Filter rows where either DEP_DELAY or ARR_DELAY are greater than 0
    filtered_df = df_train[(df_train['DEP_DELAY'] > 0) | (df_train['ARR_DELAY'] > 0)]

    # Calculate average delays by DAY_OF_WEEK
    avg_dep_delay = (
        filtered_df.groupby('DAY_OF_WEEK')['DEP_DELAY']
        .mean()
        .reset_index()
        .rename(columns={'DEP_DELAY': 'Average_DEP_DELAY'})
    )

    avg_arr_delay = (
        filtered_df.groupby('DAY_OF_WEEK')['ARR_DELAY']
        .mean()
        .reset_index()
        .rename(columns={'ARR_DELAY': 'Average_ARR_DELAY'})
    )
    return avg_dep_delay, avg_arr_delay

def plot_avg_delays_by_day_of_week(selected_year_trend):
    # Render title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Departure & Arrival Delays by Day of Week</h5>',
        unsafe_allow_html=True
    )

    # Fetch preprocessed data
    avg_dep_delay, avg_arr_delay = get_avg_delays_by_day_of_week()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Departure Delay', 'Average Arrival Delay')
    )

    # Plot Average Departure Delay
    fig.add_trace(
        go.Bar(
            x=avg_dep_delay['DAY_OF_WEEK'],
            y=avg_dep_delay['Average_DEP_DELAY'],
            name='Average Departure Delay',
            marker_color='purple'
        ),
        row=1, col=1
    )

    # Plot Average Arrival Delay
    fig.add_trace(
        go.Bar(
            x=avg_arr_delay['DAY_OF_WEEK'],
            y=avg_arr_delay['Average_ARR_DELAY'],
            name='Average Arrival Delay',
            marker_color='orange'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Average Departure Delay (minutes)',
        xaxis2_title='Day of Week',
        yaxis2_title='Average Arrival Delay (minutes)',
        showlegend=False,
        template='plotly_white'
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

# Plot Delays by Season with cached data preprocessing 
@st.cache_data
def get_avg_delays_by_season():
    # Filter rows where either DEP_DELAY or ARR_DELAY are greater than 0
    filtered_df = df_train[(df_train['DEP_DELAY'] > 0) | (df_train['ARR_DELAY'] > 0)]

    # Calculate average delays by SEASON
    average_delays = (
        filtered_df.groupby('SEASON')[['DEP_DELAY', 'ARR_DELAY']]
        .mean()
        .reset_index()
        .rename(columns={'DEP_DELAY': 'Average_DEP_DELAY', 'ARR_DELAY': 'Average_ARR_DELAY'})
    )
    return average_delays

def plot_avg_delays_by_season(selected_year_trend):
    # Display title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Departure & Arrival Delays by Season</h5>',
        unsafe_allow_html=True
    )

    # Fetch preprocessed data
    average_delays = get_avg_delays_by_season()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Departure Delay by Season', 'Average Arrival Delay by Season')
    )

    # Plot Average Departure Delay
    fig.add_trace(
        go.Bar(
            x=average_delays['SEASON'],
            y=average_delays['Average_DEP_DELAY'],
            name='Average Departure Delay',
            marker_color='purple'
        ),
        row=1, col=1
    )

    # Plot Average Arrival Delay
    fig.add_trace(
        go.Bar(
            x=average_delays['SEASON'],
            y=average_delays['Average_ARR_DELAY'],
            name='Average Arrival Delay',
            marker_color='orange'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Average Departure Delay (minutes)',
        xaxis2_title='Season',
        yaxis2_title='Average Arrival Delay (minutes)',
        showlegend=False,
        template='plotly_white'
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)


# Plot Delays by Distance Groups
@st.cache_data
def get_delay_distribution_by_distance_group():
    # Group data by DISTANCE_GROUP_DESC and DELAY_CLASS_NUMERIC, and calculate counts
    distribution = (
        df_train.groupby(['DISTANCE_GROUP_DESC', 'DELAY_CLASS_NUMERIC'])
        .size()
        .unstack(fill_value=0)  # Create a matrix-like structure with counts
    )
    return distribution

def plot_delays_by_distance_groups(selected_year_trend):
    # Display title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Distribution of Delays by Distance Groups</h5>',
        unsafe_allow_html=True
    )

    # Fetch preprocessed data
    distribution = get_delay_distribution_by_distance_group()

    # Create a grouped bar plot
    fig = go.Figure()

    # Add traces for each delay class
    for delay_class in distribution.columns:
        fig.add_trace(
            go.Bar(
                x=distribution.index,
                y=distribution[delay_class],
                name=f'Delay Class {delay_class}'
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title='Distance Group Description',
        yaxis_title='Count',
        barmode='group',  # Group bars for each distance group
        template='plotly_white',
        xaxis_tickangle=-90,
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)


# Plot Delays by Segment Number across distance groups
def plot_delays_by_segments_and_distance_groups(selected_year_trend):
    st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">Delays by Flight Segments & Distance Groups</h5>', unsafe_allow_html=True)
    # Load static image since dynamic compute here slows down page load
    image = Image.open(flight_delays_segment_image_path) 
    st.image(image, use_container_width=False)

# Plot Delays by Airline and Airport as a Heatmap
@st.cache_data
def get_median_delays_by_airline_and_airport():
    # Calculate median delays per carrier and departing airport
    median_delay = (
        df_train.groupby(['CARRIER_NAME', 'DEPARTING_AIRPORT'])[['DEP_DELAY', 'ARR_DELAY']]
        .median()
        .reset_index()
    )

    # Filter rows where DEP_DELAY and ARR_DELAY are greater than 0
    median_delay_filtered = median_delay[
        (median_delay['DEP_DELAY'] > 0) & (median_delay['ARR_DELAY'] > 0)
    ]

    # Calculate the overall median delay across DEP_DELAY and ARR_DELAY
    median_delay_filtered['MEDIAN_DELAY'] = median_delay_filtered[['DEP_DELAY', 'ARR_DELAY']].median(axis=1)
    return median_delay_filtered

def plot_delays_by_airline_and_airport_as_heatmap(selected_year_trend):
    # Display title
    st.markdown(
        '<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">'
        'Departure & Arrival Delays by Airline and Departing Airport</h5>',
        unsafe_allow_html=True
    )

    # Fetch preprocessed data
    median_delay_filtered = get_median_delays_by_airline_and_airport()

    # Allow user to select the number of top airlines to display
    top_n_carriers = st.slider(
        f'For year {selected_year_trend}, select number of top Airlines to display:', 5, 15, 10
    )

    # Identify top N carriers by median delay
    top_carriers = (
        median_delay_filtered.groupby('CARRIER_NAME')['MEDIAN_DELAY']
        .median()
        .nlargest(top_n_carriers)
        .index
    )

    # Function to get top 20 airports for each carrier by median delay
    def get_top_20_airports(group):
        return group.nlargest(20, 'MEDIAN_DELAY')

    # Filter data for top carriers and their top 20 airports
    median_delay_top = (
        median_delay_filtered[median_delay_filtered['CARRIER_NAME'].isin(top_carriers)]
        .groupby('CARRIER_NAME')
        .apply(get_top_20_airports)
        .reset_index(drop=True)
    )

    # Pivot the data to create a matrix suitable for a heatmap
    heatmap_data = median_delay_top.pivot(
        index='CARRIER_NAME', columns='DEPARTING_AIRPORT', values='MEDIAN_DELAY'
    )

    # Sort carriers by their median delay
    carrier_order = (
        median_delay_top.groupby('CARRIER_NAME')['MEDIAN_DELAY']
        .median()
        .sort_values(ascending=False)
        .index
    )

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.loc[carrier_order].values,
        x=heatmap_data.columns,
        y=carrier_order,
        colorscale='YlOrRd',
        colorbar=dict(title='Median Delay (minutes)'),
        text=heatmap_data.loc[carrier_order].values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        hoverongaps=False,
        zmin=0.1  # Minimum value to slightly exclude zeros
    ))

    # Update the layout
    fig.update_layout(
        xaxis_title='Departing Airport',
        yaxis_title='Airline',
        xaxis_tickangle=-90,
        width=800,
        height=800
    )

    # Display 
    st.plotly_chart(fig, use_container_width=True)


# Plot for Delay Class Distribution from historical data
def plot_delay_class_distribution(selected_year_trend):
    with st.expander('Flight Delay and On-Time Distribution Metrics', expanded=False):
        st.write(
            f"""
            <p style="color: #808080;">
            Distribution of flight delays across different classes in historical data for {selected_year_trend} is below. 
            Understanding this distribution is crucial for analyzing trends in on-time flights
            and in predicting and mitigating delays. 
            This chart shows the proportion of flights in each class.</p>
            """,
            unsafe_allow_html=True
        )

        class_mapping = {
            'Class 0: On-time departure and arrival': 0,
            'Class 1: Either departure or arrival delayed': 1,
            'Class 2: Both departure and arrival delayed': 2
        }

        # Calculate normalized value counts for delay classes
        delay_class_numeric_counts = df_train['DELAY_CLASS_NUMERIC'].value_counts(normalize=True)
       
        # Reverse the class_mapping dictionary
        reverse_class_mapping = {v: k for k, v in class_mapping.items()}

        # Map numeric classes to labels
        class_labels = [reverse_class_mapping[num] for num in delay_class_numeric_counts.index]

        # Create the pie chart with "equal" aspect ratio to visualize as a circle
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.pie(delay_class_numeric_counts.values, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal') 
        plt.title('DELAY_CLASS_NUMERIC Distribution')

        # Display 
        st.pyplot(fig)

# Plot for Flight Duration Category Distribution from historical data
def plot_flight_duration_distribution(selected_year_trend):
    # Calculate distribution of flight duration categories
    duration_counts = df_train['FLIGHT_DURATION_CATEGORY'].value_counts()

    # Use a pastel color palette
    colors = sns.color_palette('pastel')

    # Create figure for the pie chart
    fig, ax = plt.subplots(figsize=(3, 3))

    # Create pie chart for Training Data
    ax.pie(duration_counts.values,
           labels=duration_counts.index,
           autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100. * sum(duration_counts)):,})',
           startangle=90,
           colors=colors,
           wedgeprops=dict(width=0.6, edgecolor='white'))
    
    ax.set_title('Flight Duration Categories', fontsize=8, pad=8)
    ax.axis('equal')

    # Display within an expander
    with st.expander("Flight Duration Category Distribution Metrics", expanded=False):
        st.write(f"""
            <p style="color: #808080;">
            Distribution of flight duration categories in historical data for {selected_year_trend} is below. 
            Understanding this distribution is crucial for analyzing trends in on-time flights
            and in predicting and mitigating delays. The categories are defined as follows: </p>
            <ul style="color: #808080; list-style-type: disc; margin-left: 20px;">
                <li>Short: Flights lasting less than 60 minutes (under 1 hour)</li>
                <li>Medium: Flights lasting between 60 and 179 minutes (1 to 3 hours)</li>
                <li>Long: Flights lasting 180 minutes or more (3 hours or more)</li>
            </ul>""", 
            unsafe_allow_html=True)
        st.pyplot(fig)


# Display Insights and Recommendations that are statitically inferred from Exploratory Data Analysis
def display_findings_and_recommendations():
    findings = [
        "Highest Departure and Arrival Delays by Airlines",
        "Top Congested Airports with Flight Delays",
        "Delay Trends Across Distance Groups and Flight Segments",
        "Seasonal Trends",
        "Time of Day",
        "Weekly Patterns"
    ]

    recommendations = [
        [
            "Implement targeted training and support programs for high-delay carriers to improve operational efficiency.",
            "Use delay data to manage customer communications proactively."
        ],
        [
            "Allocate more resources and staff during peak times at congested airports to minimize delays.",
            "Develop contingency plans for high-traffic airports to handle surges in passenger volume effectively."
        ],
        [
            "Analyze operational schedules to optimize turnaround times for flights, especially those with multiple segments.",
            "Review scheduling for short and moderate-distance flights to reduce potential delays."
        ],
        [
            "Increase staffing and operational resources during summer months to manage higher delay rates effectively.",
            "Monitor weather patterns and adjust scheduling in advance to minimize disruptions during winter months."
        ],
        [
            "Consider adjusting flight schedules to reduce the number of early morning and late-night flights that experience high arrival delays.",
            "Increase capacity and resources during afternoon and evening hours to mitigate departure delays."
        ],
        [
            "Evaluate operational strategies to understand the factors contributing to increased delays on specific days.",
            "Promote Saturday travel incentives to balance the load and improve operational efficiency."
        ]
    ]

    for finding, recommendation_list in zip(findings, recommendations):
        with st.expander(f"{finding}", expanded=False):
            for recommendation in recommendation_list:
                st.write(f"""<p style="color: #808080;">- {recommendation}</p>""", unsafe_allow_html=True)

########################################################################################################################
# SkyFlow Tabs Rendering
########################################################################################################################
# Tab1 - Flight Delay Insights Tracker Tab
with tabs[0]:
    state.tab_locked = False
    
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #555; font-family: Arial, sans-serif; line-height: 0.8;">Flight Delay Insights Tracker</h3>
            <p style="color: #555; font-family: Arial, sans-serif; line-height: 0.8; margin-top: 10px;">Our Flight Delay Insights Tracker analyzes various factors influencing historical flight delays. Simply select the trend and gain more insights into historical flight delays!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if state.tab_locked:
        locked_tab_content(tabs[0])
    else:
        col = st.columns((1.5, 2.5, 4.5, 3.0), gap='medium')
        
        with col[0]:  
            st.markdown('<h5 style="color: #808080 ; font-family: Arial, sans-serif; line-height: 1.0;">Tracker</h5>', unsafe_allow_html=True)
            delay_years_list = ['2019']
            selected_year_trend = st.selectbox('Select Year:', delay_years_list)
            delay_trends_list = ['None', 'By Airline', 'By Season', 'By Day of Week', 'By Part of Day', 'By Distance Groups', 'By Flight Segments & Distance Groups', 'By Airlines & Departing Airports', 'By Routes']
            selected_delay_trend = st.selectbox(f'View Delay Trends for Year {selected_year_trend}:', delay_trends_list)
            
        with col[1]:
            st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">Insights</h5>', unsafe_allow_html=True)
            user_question = st.text_input("Ask me anything about SkyFlow's Insights:", "")
            if user_question:
                st.info(f"Feature coming soon to respond to your ask for: {user_question}")
            st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">Finding and Recommendation</h5>', unsafe_allow_html=True)
            display_findings_and_recommendations()
            
        with col[2]:
            st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">Number of Flights Departing from Airports</h5>', unsafe_allow_html=True)
            # Display static image since it takes a while to compute and render over the large data-set dynamically
            image = Image.open(flights_image_path) 
            desired_height = 350  
            aspect_ratio = image.width * 1.00 / image.height
            new_width = int(desired_height * aspect_ratio)
            resized_image = image.resize((new_width, desired_height))
            st.image(resized_image, caption=f'For selected year {selected_year_trend}', use_container_width=False)

            # Create mapping of options to functions
            option_map = {
                'None': lambda: display_default_message(),
                'By Airline': lambda: plot_avg_delays_by_carrier(selected_year_trend),
                'By Distance Groups': lambda: plot_delays_by_distance_groups(selected_year_trend),
                'By Season': lambda: plot_avg_delays_by_season(selected_year_trend),
                'By Day of Week': lambda: plot_avg_delays_by_day_of_week(selected_year_trend),
                'By Part of Day': lambda: plot_avg_delays_by_part_of_day(selected_year_trend), 
                'By Airlines & Departing Airports': lambda: plot_delays_by_airline_and_airport_as_heatmap(selected_year_trend), 
                'By Flight Segments & Distance Groups': lambda: plot_delays_by_segments_and_distance_groups(selected_year_trend),
                'By Routes': lambda: display_other_trend_message(selected_delay_trend)
            }

            # Get and Call the corresponding function for the selected option
            option_map.get(selected_delay_trend)()

        with col[3]:
            st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">Top Congested Airports</h5>', unsafe_allow_html=True)
            plot_congested_airports(selected_year_trend)
            st.markdown('<h5 style="color: #808080; font-family: Arial, sans-serif; line-height: 1.0;">About: Historical Data</h5>', unsafe_allow_html=True)
            plot_delay_class_distribution(selected_year_trend)
            plot_flight_duration_distribution(selected_year_trend)

######################################################################################################################################
# Tab2 - Flight Delay Predictor Tab
######################################################################################################################################
# Flight Delay Predictor Tab
with tabs[1]:
    state.tab_locked = True
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #555; font-family: Arial, sans-serif; line-height: 0.8;">Flight Delay Predictor</h3>
            <p style="color: #555; font-family: Arial, sans-serif; line-height: 0.8; margin-top: 10px;">Our Flight Delay Predictor analyzes various factors to forecast potential flight delays. Simply enter the required information, and let our model do the rest!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if state.tab_locked:
        locked_tab_content(tabs[1])
    else:
        # Create two main columns for the layout
        left_column, right_column = st.columns([1, 2])

        # Left column of inputs
        with left_column:
            st.markdown('<h4 style="color: #555; font-family: Arial, sans-serif; line-height: 1.0;">Share Your Flight Journey</h3>', unsafe_allow_html=True)

            # Define input fields
            user_inputs_dict = {}

            # First sub-row: Date of Flight and Airline
            st.markdown(
                    "<h6 style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 1.0;'>Flight Date and Airline</h4>",
                    unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                flight_date = st.date_input("Date of Flight:")
            with col2:
                airlines = pd.read_csv(airlines_csv_path)
                user_inputs_dict['AIRLINE'] = st.selectbox('Airline:', airlines['CARRIER_NAME'])

            # Second sub-row: From and To Airport
            st.markdown(
                    "<h6 style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 1.0;'>Airports</h4>",
                    unsafe_allow_html=True
            )
            airports = pd.read_csv(airports_csv_path)
            col3, col4 = st.columns(2)
            with col3:
                user_inputs_dict['FROM_AIRPORT'] = st.selectbox('From Airport:', airports['DISPLAY_AIRPORT_NAME'])
            with col4:
                user_inputs_dict['TO_AIRPORT'] = st.selectbox('To Airport:', airports['DISPLAY_AIRPORT_NAME'])

            # Third sub-row: Flight Times
            st.markdown(
                    "<h6 style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 1.0;'>Flight Times</h4>",
                    unsafe_allow_html=True
            )
            col5, col6 = st.columns(2)
            with col5:
                departure_time = st.time_input("Departure Time:")
            with col6:
                arrival_time = st.time_input("Arrival Time:")

            # Predict button 
            predict_button = st.button("Predict Flight Delay")
            
            # Right column for results
        with right_column:
            st.markdown('<h4 style="color: #555; font-family: Arial, sans-serif; line-height: 1.0;">Flight Delay Insights Just for You!</h3>', unsafe_allow_html=True)

            if predict_button:
                try:
                    print(json.dumps(user_inputs_dict, indent=2))
                    derived_features = map_inputs_to_features(user_inputs_dict, flight_date, departure_time, arrival_time)    
                    print(json.dumps(derived_features, indent=2))         
                        
                    # FastAPI server deployed on EC2 instance
                    response = requests.post("http://ec2-18-116-112-50.us-east-2.compute.amazonaws.com:8000/predict", json = derived_features) 
                        
                    # Uncomment for local testing
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
                st.info('Share your Flight Journey and click "Predict Flight Delay" to see results')       
######################################################################################################################################
# Tab3 - Airline Sentiment Analyzer Tab
######################################################################################################################################
# Airline Sentiment Analyzer Tab
with tabs[2]:
    state.tab_locked = True
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #555; font-family: Arial, sans-serif; line-height: 0.8;">Airline Sentiment Analyzer</h3>
            <p style="color: #555; font-family: Arial, sans-serif; line-height: 0.8; margin-top: 10px;">Stay informed with our live X feed that analyzes airline sentiments in real-time. Gain valuable insights and notifications on public perceptions about various airlines!</p> 
        </div>
        """,
        unsafe_allow_html=True
    )
    if state.tab_locked:
        locked_tab_content(tabs[2])
    else:
        # Sample List of sentiment data for different airlines
        airline_sentiments = [
            {
                "Airline": "Southwest Airlines",
                "Positive Sentiment": 0.8,
                "Negative Sentiment": 0.2,
                "DelayedFlights": 0.36,
                "Date": "Oct 31, 2024"
            },
            {
                "Airline": "Delta Airlines",
                "Positive Sentiment": 0.75,
                "Negative Sentiment": 0.25,
                "DelayedFlights": 0.30,
                "Date": "Nov 1, 2024"
            },
            {
                "Airline": "American Airlines",
                "Positive Sentiment": 0.70,
                "Negative Sentiment": 0.30,
                "DelayedFlights": 0.40,
                "Date": "Oct 30, 2024"
            },
            {
                "Airline": "United Airlines",
                "Positive Sentiment": 0.65,
                "Negative Sentiment": 0.35,
                "DelayedFlights": 0.45,
                "Date": "Nov 1, 2024"
            },
            {
                "Airline": "JetBlue Airways",
                "Positive Sentiment": 0.85,
                "Negative Sentiment": 0.15,
                "DelayedFlights": 0.25,
                "Date": "Nov 1, 2024"
            },
            {
                "Airline": "Alaskan Airlines",
                "Positive Sentiment": 0.55,
                "Negative Sentiment": 0.45,
                "DelayedFlights": 0.65,
                "Date": "Nov 1, 2024"
            }
        ]

        # Create two columns with widths 2 for the first column and the remaining width for the second column
        col1, col2 = st.columns((1.0, 4.5), gap='medium')

        # Adding content to each column
        with col1:
            # Create a dropdown to select the airline
            selected_airline = st.selectbox("Select Airline", [sentiment['Airline'] for sentiment in airline_sentiments])

        with col2:
            # TODO: Analyze the sentiment text (start with a static CSV dump, and then move to leveraging real-time Twitter feed) instead of dummy reads of airline_sentiments 
            # Find the selected airline's sentiment data
            for sentiment in airline_sentiments:
                if sentiment['Airline'] == selected_airline:
                    # Display the sentiment data for the selected airline with custom styling
                    st.markdown(
                        f"<h5 style='color: #555; font-family: Arial, sans-serif; line-height: 1.0;'>{sentiment['Airline']}</h5>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<h6 style='color: #808080; font-family: Arial, sans-serif; line-height: 0.8;'>Sentiment Analysis</h6>", unsafe_allow_html=True)
                    st.write(f"<p style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 0.6;'><strong>Positive Sentiment:</strong> {sentiment['Positive Sentiment'] * 100:.1f}%</p>", unsafe_allow_html=True)
                    st.write(f"<p style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 0.6;'><strong>Negative Sentiment:</strong> {sentiment['Negative Sentiment'] * 100:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown("<h6 style='color: #808080; font-family: Arial, sans-serif; line-height: 0.8;'>Flight Delay Information</h6>", unsafe_allow_html=True)
                    st.write(f"<p style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 0.6;'><strong>Percentage of Delayed Flights:</strong> {sentiment['DelayedFlights'] * 100:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #A9A9A9; font-family: Arial, sans-serif; line-height: 0.4;'><em>*Tweets analyzed for the 7-day period ending: {sentiment['Date']}</em></p>", unsafe_allow_html=True)

######################################################################################################################################
# Tab4 - Flight and Trip Planner
######################################################################################################################################
# Flight and Trip Planner Tab
with tabs[3]:
    state.tab_locked = True
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #555; font-family: Arial, sans-serif; line-height: 0.8;">Personalized Flight and Trip Planner</h3>
            <p style="color: #555; font-family: Arial, sans-serif; line-height: 0.8; margin-top: 10px;">Experience seamless travel planning with our Agentic AI-powered Planner.<br><br>
            It dynamically tailors flight and trip recommendations based on your preferences, budget, and schedule, offering real-time updates and smart suggestions.<br><br>
            From booking flights to curating personalized itineraries, it adapts to your unique travel needs, ensuring a hassle-free and unforgettable journey.</p> 
        </div>
        """,
        unsafe_allow_html=True
    )
    if state.tab_locked:
        locked_tab_content(tabs[3])
    else:
        st.markdown("<h6 style='color: #808080; font-family: Arial, sans-serif; line-height: 0.8;'>Coming Soon!</h6>", unsafe_allow_html=True)

################################################################################################################################
# Bottom Separator used across all of SkyFlow's tabs
################################################################################################################################   
# Add a separator
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #555;">
        <p>✨ Developed by <a href="https://www.linkedin.com/in/dipti-aswath-60b9131/"><strong>Dipti Aswath</strong></a></p>
        <p>🔍 Explore SkyFlow's comprehensive documentation with operational insights and technical details <a href="https://diptiaswath.github.io/airlineFlightDelayPrediction/" style="color: #1E90FF;">here</a></p>
        <p style="margin-top: 20px;">© Pending. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
