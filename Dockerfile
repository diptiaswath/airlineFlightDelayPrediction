# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

RUN pwd

# Copy the requirements file
COPY ./app_interface/requirements.txt /code/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app folder into the container
COPY ./app_interface /code/app_interface

RUN pwd

# Copy the config.toml file into the container
COPY ./app_interface/.streamlit/config.toml /code/app_interface/.streamlit/config.toml

# Copy the model directory (with the saved model file) into the container
COPY ./model_artifacts /code/model_artifacts

# Copy the required data and image files into the container
COPY ./raw_data/airports_list.csv /code/raw_data/airports_list.csv
COPY ./raw_data/CARRIER_DECODE.csv /code/raw_data/CARRIER_DECODE.csv
COPY ./raw_data/AIRPORT_COORDINATES.csv /code/raw_data/AIRPORT_COORDINATES.csv
COPY ./combined_data/train.pkl /code/combined_data/train.pkl
COPY ./images/b6cf1189a8363e9708a712a22171e35a.jpeg /code/images/b6cf1189a8363e9708a712a22171e35a.jpeg
COPY ./images/Delays_by_Segment_Distance_Group.jpeg /code/images/Delays_by_Segment_Distance_Group.jpeg

# Copy the start.sh script into the container
COPY ./start.sh /code/start.sh

# Set executable permission for start.sh
RUN chmod +x /code/start.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run both FastAPI and Streamlit with and wothout a start script
# CMD ["sh", "-c", "uvicorn app_interface.main:skyflow --reload --host 0.0.0.0 --port 8000 & streamlit run app_interface/skyflow_app.py --server.port 8501 --server.enableCORS=false --server.address 0.0.0.0"]
# CMD ["sh", "-c", "uvicorn app_interface.main:skyflow --reload --host 0.0.0.0 --port 8000 & streamlit run app_interface/skyflow_app.py"]
CMD ["/code/start.sh"]