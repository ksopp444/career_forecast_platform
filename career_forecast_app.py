import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the adjusted dataset for income projections
data = pd.read_csv('Adjusted_Synthetic_Time_Series_Income_Data.csv')

# Function to run the forecast with adjustable factors
def run_forecast_with_adjustments(major, cpi_values, gdp_values, experience_values):
    data_filtered = data[data['Major'] == major]
    data_agg = data_filtered.groupby('Year').agg({'Income': 'mean', 'CPI': 'mean', 'GDP': 'mean', 'Years_of_Experience': 'mean'}).reset_index()
    
    income_ts = data_agg['Income'].values
    cpi_ts = np.array(cpi_values)  # Using the CPI values set by students
    gdp_ts = np.array(gdp_values)  # Using the GDP values set by students
    experience_ts = np.array(experience_values)  # Using the experience values set by students
    
    # Create exogenous variables (CPI, GDP, experience)
    exog_vars = np.column_stack([cpi_ts, gdp_ts, experience_ts])
    
    # Fit the ARIMA model
    model = SARIMAX(income_ts, exog=exog_vars, order=(1, 1, 1)).fit(disp=False)
    
    # Forecast for the next 4 years using the student inputs
    forecast = model.get_forecast(steps=4, exog=np.column_stack([cpi_values, gdp_values, experience_values]))
    
    return income_ts, forecast.predicted_mean

# Streamlit Interface
st.title("Interactive Career Path Forecast")

# User input for major
major = st.selectbox("Select Major", options=["Accounting", "Finance"])

# Sliders for CPI, GDP, and years of experience
st.subheader("Adjust Economic Factors:")
cpi_values = []
gdp_values = []
experience_values = []

for i in range(4):
    cpi = st.slider(f"Year {i+1} CPI", 250, 300, 265)
    gdp = st.slider(f"Year {i+1} GDP Growth (%)", 1.0, 5.0, 2.5)
    experience = st.slider(f"Year {i+1} Average Years of Experience", 1, 30, 10)
    
    cpi_values.append(cpi)
    gdp_values.append(gdp)
    experience_values.append(experience)

# Run forecast with the adjusted values
income_ts, forecast_mean = run_forecast_with_adjustments(major, cpi_values, gdp_values, experience_values)

# Plot the results
st.subheader(f"Income Forecast for {major} Major with Adjusted Factors")
fig, ax = plt.subplots()
ax.plot(np.arange(len(income_ts)), income_ts, label="Actual Income")
ax.plot(np.arange(len(income_ts), len(income_ts) + len(forecast_mean)), forecast_mean, label="Forecasted Income", linestyle='--')
ax.set_title(f'Income Forecast for {major} Major with Adjusted Factors')
ax.legend()
st.pyplot(fig)
