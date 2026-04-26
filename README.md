
# Smart Grid Energy Demand Forecasting

Forecasting electricity demand at hourly resolution using LSTM deep learning — built to optimize power generation, reduce wastage, and support stable grid distribution.

**Live Demo →** [smart-grid-energy-demand-forecasting.streamlit.app](https://smart-grid-energy-demand-forecasting-zvjgfqxycvvqgcqerc5dzp.streamlit.app/)

---
<img width="1498" height="712" alt="image" src="https://github.com/user-attachments/assets/0cdfcfe4-0fdf-4c94-a18a-50f76cd27c4d" />
## The Problem

Energy providers need to know how much electricity will be consumed hours in advance — too little generation causes outages, too much causes waste and cost overruns. Traditional statistical methods struggle with the non-linear, cyclical nature of energy demand patterns.

This project applies LSTM (Long Short-Term Memory) neural networks to historical hourly consumption data to produce accurate short-term forecasts, surfaced through an interactive Streamlit dashboard.

---

## What It Does

- Forecasts electricity demand (in MW) up to 48 hours ahead
- Trains an LSTM model on real-world hourly consumption data (AEP dataset, Kaggle)
- Serves predictions through an interactive web dashboard with Plotly visualizations
- Supports custom `.csv` dataset uploads for reuse across grid zones
- Allows forecast export as `.csv` for downstream use

---

## Architecture

### Model Design

The core model is a stacked LSTM network — chosen for its ability to capture long-range temporal dependencies in sequential data, which simpler models like ARIMA cannot do reliably.

```
Input: 24-hour sliding window of normalized demand values
       ↓
LSTM(64, return_sequences=True)   # Captures broad temporal patterns
       ↓
LSTM(32)                          # Refines sequence representation
       ↓
Dense(16, activation='relu')      # Non-linear feature combination
       ↓
Dense(1)                          # Single-step demand prediction (MW)

Loss: Mean Squared Error
Optimizer: Adam
Scaling: MinMaxScaler (fitted on training set, persisted via joblib)
```

### Inference Pipeline

```
Raw CSV → Timestamp parsing → MinMaxScaler → 24-step window →
LSTM model → Inverse transform → Forecast output → Plotly chart / CSV export
```

---

## Results

- The model captures daily demand cycles (morning peaks, overnight troughs) with high fidelity
- RMSE values reflect stable generalization on held-out data
- Forecast curves show realistic demand shape without over-smoothing

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Keras `.keras` format, `joblib` for scaler |
| Version Control | Git + GitHub |

---

## Project Structure

```
smart-grid-energy-forecasting/
├── data/
│   └── AEP_hourly.csv        # Hourly energy consumption dataset
├── model/
│   ├── energy_lstm.keras     # Trained LSTM model
│   └── scaler.pkl            # Fitted MinMaxScaler
├── app.py                    # Streamlit dashboard
├── train_model.py            # Model training script
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/sreenugopireddy/Smart-grid-energy-demand-forecasting.git
cd Smart-grid-energy-demand-forecasting
```

### 2. Set up a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This loads `data/AEP_hourly.csv`, fits the scaler, trains the LSTM, and saves both artifacts to `model/`.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Dataset

**AEP Hourly Energy Consumption** — publicly available on [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).  
Contains years of hourly MW readings from American Electric Power's service territory.

---

## Dependencies

```
pandas, numpy, scikit-learn, tensorflow, plotly, matplotlib, seaborn, streamlit, joblib
```

---

## Roadmap

- Integrate weather and temperature features to improve forecast accuracy
- Benchmark against Prophet and ARIMA baselines
- Add anomaly detection for irregular consumption patterns
- Extend to multi-step multi-horizon forecasting
- Deploy on Hugging Face Spaces with persistent model hosting

---

## Author

**Sreenu Gopireddy**  
[sreenugopireddy65@gmail.com](mailto:sreenugopireddy65@gmail.com) · [GitHub](https://github.com/sreenugopireddy)
