import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Retail Sales Forecasting", layout="centered")
st.title("📈 Retail Sales Forecasting App")

st.markdown("This app uses Facebook Prophet to forecast sales from your dataset.")

# Load your fixed file directly
file_path = "mock_kaggle.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    df = df.rename(columns={"data": "ds", "venda": "y"})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# Load and display data
df = load_data()
st.subheader("Raw Data Preview")
st.dataframe(df.head())

# Fit the Prophet model
model = Prophet()
model.fit(df)

# Make future predictions
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Evaluate
actual = df.set_index('ds')
predicted = forecast.set_index('ds')[['yhat']].loc[actual.index]
rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
mae = mean_absolute_error(actual['y'], predicted['yhat'])

st.subheader("📊 Model Performance")
st.markdown(f"- **RMSE:** `{rmse:.2f}`")
st.markdown(f"- **MAE:** `{mae:.2f}`")

# Plot forecast
st.subheader("📉 Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot components
st.subheader("🧩 Seasonal Trends")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Show and download forecast table
st.subheader("📅 Forecast Table (Last 10 Rows)")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_output.csv", mime='text/csv')
