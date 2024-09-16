# Optimizing Airline Operations: Predicting Delays for Smarter Routes & Resource Management

Dipti Aswath @ [LinkedIn](www.linkedin.com/in/dipti-aswath-60b9131)

## Executive Summary

### Problem Statement:

Airlines and airports face significant operational challenges due to flight delays, which can be caused by a variety of factors including flight status, weather conditions, air traffic congestion, aircraft specifics, and inefficiencies in ground and passenger handling. The objective is to predict flight delays by developing a multi-class classification model that considers both departure and arrival delays, helping improve operational planning and customer satisfaction.

### Rationale:

Flight delays can have widespread consequences for airlines, from passenger dissatisfaction to operational disruptions. Developing a predictive model for flight delays not only addresses the core issue of minimizing delays but also enhances decision-making processes across various facets of airline operations.

#### Business Use Case: Airline Route Optimization:

-   **Proactive Rerouting**: Predicting delays enables airlines to reroute flights, helping avoid congested airspace or airports experiencing delays, thereby reducing the overall impact on flight schedules and minimizing delay time.

-   **Weather Avoidance**: Forecasting weather-related delays allows for dynamic adjustments to flight routes, ensuring safer and more efficient travel by avoiding adverse weather conditions.

#### Business Use Case: Airline Operations Management:

-   **Scheduling Adjustments**: Delay predictions empower airlines to adjust schedules in real time, reallocating resources like gates, ground crew, and equipment to manage disruptions effectively.

-   **Resource Allocation**: Advanced knowledge of delays facilitates more efficient allocation of resources, such as repositioning aircraft and crew, reducing the cascading impact of delays on subsequent flights.

-   **Passenger Communication**: With accurate delay predictions, airlines can inform passengers in advance, manage expectations, and provide better customer service, such as offering rebooking options or compensation for delays.

#### Example Usage: An AI system that predicts flight delays could:

1.  Suggest alternate flight paths that are less likely to experience delays.

2.  Provide passengers with timely updates and rebooking options.

3.  Dynamically adjust flight schedules to manage disruptions effectively.

4.  Allocate resources efficiently to minimize the impact on subsequent flights.

### Research Question:

How can a multi-class classification model be developed to accurately predict flight delays by assessing multiple factors, including departure and arrival delays, using data related to flight status, weather conditions, air traffic, aircraft specifics, and ground operations?

### Key Findings from Exploratory Data Analysis:

1. **Highest Departure and Arrival delays by Carriers (2019):**
Identifying the carriers with the highest delays directly relates to **Improved Customer Experience and Financial Impact**. By pinpointing these carriers, airlines can better manage customer expectations, offer targeted support, and address issues that could lead to costly disruptions and compensation claims.

![A graph showing the average departure of a flight Description automatically generated](images/bc7e179bc1861433458bf6810faa5295.jpeg)

![A graph showing the average arrival of passengers Description automatically generated](images/bed4de04b8db99bd5041395fbf01c60f.jpeg)

2.  **Top 30 Congested Airports with Flight Delays (2019):**
This finding supports **Enhanced Operational Efficiency and Operational Resilience**. By focusing on the most congested airports, airlines can optimize resource allocation and improve scheduling to alleviate delays at these critical points, leading to smoother operations and better crisis management.   

![A graph showing the number of airports Description automatically generated](images/2861aecffc78aed9ff14a1b9b60c99d4.jpeg)

![A map of the united states with different colored spots Description automatically generated](images/b6cf1189a8363e9708a712a22171e35a.jpeg)

3.  **SMOTE Resampling on Training Data**
Demonstrates the importance of **Data-Driven Decision Making**. By improving model performance through resampling, airlines can make more accurate predictions about delays, leading to better strategic planning and performance monitoring.

![A white background with black text Description automatically generated](images/f6d59c3a193cfd80aad2cd0a0252e569.jpeg)

![A blue and purple circle with white text Description automatically generated](images/aa0551b11ca73ae5fb6c1eda05220caf.jpeg)

4.  **Delay Trends Across Distance Groups and Flight Segments**
This finding helps provide valuable insights into how aircraft operational schedules and the number of daily flights contributed to 2019 delays, effectively addressing **Operational Efficiency and Contingency Planning**. Understanding how delay patterns vary with flight distance and segment numbers helps airlines plan better turnaround times and manage operational schedules more effectively to prevent delays.

**Segment Number Decreases with Distance**: As flight distance increases, the number of segments (flights) decreases. Aircraft flying longer routes complete fewer flights in a day due to time constraints.
**Delays Correlate with Higher Segment Numbers**: Flights scheduled for more segments in a day are more prone to delays, regardless of distance. These delays are likely due to operational factors, such as shorter turnaround times, leading to delayed departures and arrivals.

![A graph of different colored lines Description automatically generated with medium confidence](images/6972858dec585d485ce8ef20325ef477.jpeg)

5.  **Median Departure and Arrival Delays per Carrier:**
Identified the top 20 carriers with the highest median delays, considering both departure and arrival delays. For each carrier, the top 20 airports with the most significant contribution to delays were identified. By examining median delays, airlines can gain insights into typical delay experiences and ensure compliance with regulations. Focussing on specific carriers and airports with high delays can enhance **Overall Safety and Customer Satisfaction**.

**Comprehensive Delay Analysis:** By considering both departure and arrival delays, we provide a more holistic view of 2019 airline performance and airport efficiency. Endeavor Air Inc shows a highest delay at Miami International Airport. Comair Inc follows with the next highest delay at Portland International Airport.
**Focus on median delays**: The use of median delays helped identify typical delay experiences, filtering out the effect of extreme delays that skewed averages.
**Unique Operational Factors:** The variation in delay trends suggests that delays may be influenced by distinct factors specific to each carrier and airport, rather than being caused by common issues across multiple locations. For instance, both Endeavor Air Inc and Comair Inc experienced higher-than-usual precipitation at the airports on their flight day, which could have contributed to their delays.

![A screenshot of a graph Description automatically generated](images/fcb34c72898e35e1a0bfb19cd5d85403.jpeg)

### ![A graph of a number of aircraft carrier names Description automatically generated](images/01857b3b3a37af99de404915b9763511.jpeg)

### Model Evaluation Summary and Performance Metrics:

**Baseline Model with Dummy Classifier:**

![](images/7be16164f37e089d386ab62185feaa07.jpeg)

**Multinomial Logistic Regression Classifier:**

***Performance Summary:***
- Class 0: Excellent precision (0.99), high recall (0.92), and top F1-score (0.95). The model performs very well on this majority class.
- Classes 1 & 2: Low precision and F1-scores (0.50 and 0.64 for Class 1; 0.37 and 0.59 for Class 2) indicate many false positives. These classes have high recall, suggesting the model detects them but struggles with classification accuracy.
- Class 3: Balanced performance with a good F1-score (0.81), showing moderate effectiveness.

***Bias vs. Variance Analysis Summary:***
- High Variance: The model exhibits high variance by overfitting to the majority class (Class 0) and performing poorly on minority classes (Classes 1 and 2). This discrepancy indicates overfitting, as the model captures the majority class well but generalizes poorly to others.
- High Bias: The model exhibits high bias as well, as the model’s overall performance metrics (macro-averaged) are lower, reflecting that the model is too simplistic to handle the complexities of minority classes.
  
***Overall Summary:***

- This model exhibits high variance, as evidenced by its strong performance on the majority class but weaker performance on the minority classes. This indicates overfitting to the majority class. It also shows signs of high bias, with poor performance on minority classes. To address this, we will next evaluate ensemble bagging methods, such as Random Forest, using a Decision Tree as a base model to potentially improve generalization.

![A screenshot of a computer Description automatically generated](images/b2ad7f137a2c3862172b05c834951282.jpeg)

![A close-up of a graph Description automatically generated](images/d237759ac6cabbb60b6fdc4520186927.jpeg)


**Decision Tree Classifier:**

***Performance Summary:***
- Class 0: Exceptional performance with perfect precision (1.00), high recall (0.99), and an excellent F1-score (0.99). This model is highly effective at identifying this majority class.
- Class 1: Strong performance with high precision (0.94) and recall (0.95), resulting in a high F1-score (0.94). This model effectively identifies this class with a good balance of precision and recall.
- Class 2: Good performance with a precision of 0.88 and recall of 0.91, yielding an F1-score of 0.89. This model shows strong ability to identify this class, though there is room for improvement in precision.
- Class 3: Excellent performance with high precision (0.94) and recall (0.95), resulting in a high F1-score (0.94). This model effectively identifies this class as well.
- Overall Metrics: The overall metrics suggest that the model has excellent overall performance, with very high accuracy (0.98) and strong ability to differentiate between classes. The high PR AUC and ROC AUC scores, both macro-averaged (PR AUC of 0.90 and ROC AUC of 0.97) and weighted (PR AUC of 0.97 and ROC AUC of 0.99), indicate that this model handles class imbalances effectively and maintains good performance across different metrics.

***Bias vs. Variance Analysis Summary:***
- Low Bias: The model shows low bias, as evidenced by high precision, recall, and F1-scores across all classes. It is effectively capturing the underlying patterns in the data.
- Low Variance: The model exhibits low variance, as it generalizes well across all classes without overfitting. This is indicated by consistent performance metrics and high AUC scores.

***Overall Summary:***

- This model demonstrates strong performance with low bias and variance, effectively handling all classes with high accuracy and robust AUC scores. However, to further enhance model robustness and generalization, exploring ensemble methods such as a bagging model like Random Forest Classifier could be beneficial. Bagging can improve performance by reducing potential overfitting and increasing stability, particularly if the base Decision Tree model has high depth as in this case. Given this model's already high performance, the addition of an ensemble method will be explored next to help fine-tune and further validate results.

![A screenshot of a computer Description automatically generated](images/82e8d7539e44461c7f9cc389798a5157.jpeg)

![A screenshot of a graph Description automatically generated](images/0e541f00ed4c5eec639015d2748b19dc.jpeg)

**Random Forest Classifier:**

![A screenshot of a computer Description automatically generated](images/54ad56fdba07026f3a86f09457414d51.jpeg)

![A screenshot of a computer Description automatically generated](images/737e04ca9862678adb62a4506ec67c00.jpeg)

### Interpretation of Model Evaluations:
1. **Model Ranking**: Based on performance metrics, the models can be ranked as a) Random Forest, b) Decision Tree, c) Logistic Regression, with Random Forest being a superior model for predicting flight delays, particularly in handling class imbalance and maintaing high performance across all classes. The Decision Tree classifier also handles class imbalance much better than Logistic Regression and its performance is more consistent across classes compared to Logistic Regression.

2. **Simplicity vs. Performance**: Decision Tree performs nearly as well as Random Forest, despite being a simpler model. Its significantly better than Logistic Regression. This suggests that the decision boundaries in the data are relatively clear and can be captured by a single tree.

3. **Interpretability**: Decision Trees offer better interpretability compared to Random Forest, while maintaining high performance. This could be valuable for understanding the key decision points in flight delay prediction.

4. **Robustness**: The high performance of the Decision Tree suggests that the important patterns in the data are strong and consistent

5. **Feature Importance**: Analyzing the structure of the Decision Tree could provide clear insights into the most critical factors for predicting flight delays.

6. **Deployment**: Given its high performance and interpretability, the Decision Tree could be an excellent choice for deployment, especially if model explanation is important. 
   
## Data Sources:

Kaggle Dataset from [here](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data), that is comprised of multiple csv's listed below.

1.  Air Carrier Summary

2.  Aircraft Inventory

3.  Air Carrier employee support (Ground Crew, Flight Attendants)

4.  Flight On Time Reporting Status with Air Carrier info for 2019-2020

5.  Airport Weather

6.  Airport and Carrier look-up codes

## Methodology Used:

1.  Data Preparation

    Involved cleaning and merging multiple raw CSV files to create a unified data-set with \~4M entries (for training) and \~2M entries (for testing) with 34 predictor variables and 1 target variable. Raw data-set description is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/raw_data/raw_data_documentation.txt).

2.  Feature Engineering

    1.  Delay Categories: Classified delays into four classes/categories. a) **On-time Departure and Arrival:** Flights that depart and arrive within their scheduled times. b) **Delayed Departure, On-time Arrival:** Flights that experience delays during departure but still arrive on time or within a minimal delay window. c) **On-time Departure, Delayed Arrival**. d) **Delayed Departure and Arrival:** Flights that experience delays both in departure and arrival times.

        ![A pie chart with numbers and percentages Description automatically generated](images/37466f6e1dd66bc41f26323971cadf9e.jpeg)![A graph showing different types of classes Description automatically generated with medium confidence](images/1c21c80d52ac3af3c5475634cc711073.jpeg)

    2.  Aggregation Features: Created historical delay averages such as,

        CARRIER_HISTORICAL = captures the historical average delay rate of each carrier per month

        DEP_AIRPORT_HIST = captures historical average delay rates for flights departing from specific airports per month

        PREV_AIRPORT_HIST = captures historical average delay rate for the airport from which the aircraft arrived before the current departure

        DAY_HISTORICAL = captures historical average delays associated with each day of the week, adjusted monthly

        DEP_BLOCK_HIST = captures historical average delay rate for different departure time blocks, aggregated by month

    3.  Time Based Features: Extracted season from month and part of the day from departure and arrival time blocks

        ![A comparison of different colored bars Description automatically generated](images/c85203ce6491ccef94dedf1330bc73fd.jpeg)

        ![A group of bars with numbers Description automatically generated with medium confidence](images/109e7b83d38d2bfe4e13dd5c67060ea6.jpeg)

    4.  Distance Based Features: Mapped distance groups to descriptive text

        ![A close-up of a graph Description automatically generated](images/30988bff062a1543f4a633070acbba1f.jpeg)

    5.  Delay Based Features: Created new features by combining actual departure and arrival times with planned times to create new delay features, ELAPSED_TIME_DIFF, DEP_DELAY, ARR_DELAY

    6.  Employee Features: Created employee statistics features FLT_ATTENDANTS_PER_PASS and PASSENGER_HANDLING for analyzing airline and carrier operations

    7.  Removed highly correlated features with VIF

        ![A close-up of a document Description automatically generated](images/c2f445131a51350dbad395f03b0b4aad.png)![A close-up of a number Description automatically generated](images/7ce889c572198a3b6907833d26644d84.jpeg)

    8.  Engineered features with their descriptions can be found [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/combined_data/dataset_documentation.txt)

3. Data Pre-Processing: Missing values and outliers detected were removed. SMOTETomek was applied to just the training data-set. This combined SMOTE's oversampling of the minority classes (classes 1,2 and 3) and Tomek links' under-sampling. Categorical features were also target encoded and Numerical features were scaled.

4. Modeling and Evaluation: Classification algorithms used were Decision Trees, Random Forest, and multi-nomial Logistic Regression, with evaluation metrics: F1 Score, PR AUC, ROC AUC and Accuracy scores. Sequential Feature Selection was used to select 5 features from among the 34 predictor variables.

## Outline of Project:

1.  Engineered Features and description [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/combined_data/dataset_documentation.txt)

2.  Merged Data for [train](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/combined_data/train.pkl) and [test](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/combined_data/test.pkl) is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/tree/main/combined_data)

3.  Raw Data used for creating merged data is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/tree/main/raw_data)

4.  Exploratory Data Analysis plots from Auto Viz Visualization Library is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/tree/main/plots). Credits: Seshadri, Ram (2020). GitHub - AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. source code: https://github.com/AutoViML/AutoViz)

5.  Images captured in this README are located [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/tree/main/images)

6.  Jupyter notebooks, 
a) Notebook for data preparation, merges and feature engineering from raw data-set [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/notebooks/flight-delays-data-prep-and-eda_v1.ipynb), 
b) Notebook for data-explorations in addition to AutoViz is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/notebooks/flight-delays-data-exploration_v1-checkpoint.ipynb), 
c) Notebook for data pre-processing, modeling and evaluation is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/notebooks/flight-delays-data-preproc-and-modeling_v1.ipynb)
d) Common Utility functions used across notebooks is [here](https://github.com/diptiaswath/airlineFlightDelayPrediction/blob/main/notebooks/utils/)

## Next Steps:

1.  Use Dimensionality Reduction and Clustering to reduce dimensions, and cluster features together to reduce the count of 34 predictors. Relying on Feature Selection techniques alone, takes a while to train any of the classification models.

2.  Investigate features and decision rules contributing the most to predicting flight delays with both the Decision Tree and the Random Forest classifier.

3.  Explore ensemble methods that could combine Decision Tree with other models to potentially improve performance further.

4.  Use StreamLit and Fast API to serve flight prediction delays via an application interface
