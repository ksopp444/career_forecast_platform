
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load API key from Streamlit secrets
API_KEY = st.secrets[739908b86129b53cd835f3646a3de8c8 ]

# Function to fetch real-time data from FRED API
def fetch_fred_data(series_id):
    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': API_KEY,
        'file_type': 'json'
    }
    response = requests.get(FRED_BASE_URL, params=params)
    data = response.json()
    observations = data['observations']
    return pd.DataFrame(observations)

# Fetch real-time CPI data
cpi_real_time = fetch_fred_data("CPIAUCSL")  # Replace with actual CPI series ID

# Load the adjusted dataset for income projections
data = pd.read_csv('Adjusted_Synthetic_Time_Series_Income_Data.csv')

# Function to run forecast with real-time data
def run_forecast_with_real_time(major, cpi_values, steps=4):
    data_filtered = data[data['Major'] == major]
    data_agg = data_filtered.groupby('Year').agg({'Income': 'mean', 'CPI': 'mean'}).reset_index()
    
    income_ts = data_agg['Income'].values
    cpi_ts = data_agg['CPI'].values
    
    exog_vars = np.column_stack([cpi_ts])
    
    # Fit the ARIMA model
    model = SARIMAX(income_ts, exog=exog_vars, order=(1, 1, 1)).fit(disp=False)
    
    # Use real-time CPI data for future predictions
    future_cpi = np.array(cpi_values).reshape(-1, 1)
    forecast = model.get_forecast(steps=steps, exog=future_cpi)
    
    return income_ts, forecast.predicted_mean

# Streamlit Interface
st.title("Career Path Forecast with Real-Time Data")

# Select major
major = st.selectbox("Select Major", options=["Accounting", "Finance"])

# Use real-time CPI data
st.subheader("Real-Time CPI Values for Future Projections")
cpi_values = cpi_real_time['value'][-4:].astype(float).tolist()  # Use last 4 CPI values for forecast

# Run forecast with real-time data
income_ts, forecast_mean = run_forecast_with_real_time(major, cpi_values)

# Plot the results
st.subheader(f"Income Forecast for {major} Major with Real-Time CPI")
fig, ax = plt.subplots()
ax.plot(np.arange(len(income_ts)), income_ts, label="Actual Income")
ax.plot(np.arange(len(income_ts), len(income_ts) + len(forecast_mean)), forecast_mean, label="Forecasted Income", linestyle='--')
ax.set_title(f'Income Forecast for {major} Major with Real-Time CPI')
ax.legend()
st.pyplot(fig)
