# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# --- App Setup ---
st.set_page_config(page_title="Smart Grid Energy Dashboard", layout="wide")
st.title("⚡ Smart Grid Energy Demand Forecasting")

# --- Load Model and Scaler ---
model = load_model('model/energy_lstm.keras')
scaler = joblib.load('model/scaler.pkl')

# --- Upload or Use Default Data ---
uploaded = st.file_uploader("Upload your energy data CSV (with Datetime & MW columns)", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)
else:
    st.info("Using sample dataset (AEP_hourly.csv)")
    data = pd.read_csv("data/AEP_hourly.csv")

# --- Standardize Columns ---
data.columns = [col.strip() for col in data.columns]
if 'AEP_MW' in data.columns:
    data.rename(columns={'AEP_MW': 'MW'}, inplace=True)
if 'Datetime' not in data.columns:
    data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values('Datetime')

# --- Show Data ---
st.subheader("📊 Recent Data Preview")
st.dataframe(data.tail())

# --- Visualization: Historical Demand ---
fig1 = px.line(data, x='Datetime', y='MW', title='Energy Consumption Over Time', template='plotly_dark')
st.plotly_chart(fig1, use_container_width=True)

# --- Forecasting Section ---
st.subheader("🔮 Forecast Future Energy Demand")

n_hours = st.slider("Select how many future hours to predict:", 1, 48, 24)

scaled_data = scaler.transform(np.array(data['MW']).reshape(-1,1))
last_24 = scaled_data[-24:]

predictions = []
temp_input = list(last_24.flatten())  # ✅ flatten the list of arrays to 1D

for i in range(n_hours):
    # convert last 24 numbers into proper shape
    x_input = np.array(temp_input[-24:], dtype=float).reshape(1, 24, 1)
    
    # make prediction
    yhat = model.predict(x_input, verbose=0)
    next_value = float(yhat[0][0])  # ✅ ensure it's a scalar float
    
    # append prediction for next iteration
    temp_input.append(next_value)
    predictions.append(next_value)


forecast = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

future_dates = pd.date_range(start=data['Datetime'].iloc[-1]+timedelta(hours=1), periods=n_hours)
forecast_df = pd.DataFrame({'Datetime': future_dates, 'Predicted_MW': forecast})

fig2 = px.line(forecast_df, x='Datetime', y='Predicted_MW', title='Predicted Future Energy Demand', template='plotly_white', markers=True)
st.plotly_chart(fig2, use_container_width=True)

# --- Metrics ---
st.metric(label="Last Recorded Demand (MW)", value=f"{data['MW'].iloc[-1]:.2f}")
st.metric(label=f"Predicted Demand after {n_hours} hours", value=f"{forecast[-1]:.2f}")

# --- Download Forecast ---
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast Data", data=csv, file_name="forecast.csv", mime="text/csv")

st.success("✅ Forecast Generated Successfully!")
