⚡ Smart Grid Energy Demand Forecasting

Predicting future energy demand using Machine Learning & Deep Learning (LSTM) for smart grid optimization.
<img width="1498" height="712" alt="image" src="https://github.com/user-attachments/assets/0cdfcfe4-0fdf-4c94-a18a-50f76cd27c4d" />
🧠 Project Overview
priview link--https://smart-grid-energy-demand-forecasting-zvjgfqxycvvqgcqerc5dzp.streamlit.app/
This project uses Machine Learning and Deep Learning models to forecast electricity demand in a smart grid system.
Accurate energy demand forecasting helps optimize power generation, reduce wastage, and ensure stable electricity distribution.

Using historical hourly energy consumption data, the system predicts future energy demand (in Megawatts) and visualizes trends through an interactive Streamlit dashboard.

🚀 Key Features

✅ Predicts future electricity demand using LSTM neural networks
✅ Interactive Streamlit web dashboard for visualization
✅ Real-time forecast horizon control (1–48 hours)
✅ Plotly graphs for interactive and dynamic charts
✅ Supports custom dataset upload (.csv files)
✅ Displays last recorded demand and future predicted demand
✅ Downloadable forecast data in .csv format

🧩 Tech Stack
Category	Tools / Libraries
Language	Python 3
Frameworks	TensorFlow / Keras, Scikit-learn
Visualization	Plotly, Matplotlib, Seaborn
Web App	Streamlit
Data Handling	Pandas, NumPy
Version Control	Git + GitHub
📊 Model Used
🔹 LSTM (Long Short-Term Memory)

LSTM is a type of Recurrent Neural Network (RNN) that captures long-term temporal dependencies in sequential data, making it ideal for time series forecasting such as energy demand.

Model Summary:

Input: Last 24 hours of energy demand

Output: Next hour’s predicted demand

Layers:

LSTM(64, return_sequences=True)

LSTM(32)

Dense(16, activation='relu')

Dense(1)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

🧰 Project Structure
smart-grid-energy-forecasting/
│
├── data/
│   └── AEP_hourly.csv                # Dataset (hourly energy consumption)
│
├── model/
│   ├── energy_lstm.keras             # Trained LSTM model
│   └── scaler.pkl                    # Saved MinMaxScaler
│
├── app.py                            # Streamlit dashboard
├── train_model.py                    # Model training script
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation

🧠 How It Works

Data Preprocessing

Loads historical energy consumption data

Converts timestamps and handles missing values

Scales features using MinMaxScaler

Model Training

Uses past 24 hours of consumption to predict the next hour

Trains an LSTM model and saves it for future inference

Prediction

Model forecasts future demand for user-selected time horizons (1–48 hours)

Visualization

Interactive dashboard plots actual vs predicted energy usage

Displays demand metrics and allows CSV download of predictions

📈 Results

The LSTM model successfully learns the temporal patterns in energy consumption.

The prediction graph shows realistic rises and falls matching daily demand cycles.

RMSE (Root Mean Square Error) values indicate good accuracy and stable model performance.

🖥️ Run Locally
1️⃣ Clone the repository
git clone https://github.com/sreenugopireddy/Smart-grid-energy-demand-forecasting.git
cd Smart-grid-energy-demand-forecasting

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # (Windows)
# or
source venv/bin/activate    # (Mac/Linux)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model
python train_model.py

5️⃣ Launch the Streamlit dashboard
streamlit run app.py


Then open 👉 http://localhost:8501 in your browser.

📦 Requirements

All dependencies are listed in requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
plotly
streamlit
joblib

📸 Dashboard Preview
Section	Description
📊 Energy Demand Chart	Interactive Plotly graph showing actual & forecasted demand
🔮 Forecast Panel	User chooses how many future hours to predict
📈 Download Option	Export predicted data as CSV
💡 Metrics	Displays last recorded and predicted demand values
🧾 Dataset

The dataset used is the Hourly Energy Consumption Dataset from Kaggle:
🔗 AEP Hourly Energy Consumption

💡 Future Enhancements

Integrate weather and temperature data for improved accuracy

Add Prophet and ARIMA models for comparison

Deploy the dashboard on Streamlit Cloud or Hugging Face Spaces

Include anomaly detection for irregular power usage patterns

👨‍💻 Author

Sreenu Gopireddy
📧 [sreenugopireddy65@gmail.com]
