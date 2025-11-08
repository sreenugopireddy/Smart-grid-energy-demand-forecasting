import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib
import os
import math

# --- Load Dataset ---
print("📥 Loading dataset...")
data = pd.read_csv('data/AEP_hourly.csv')
data.columns = [col.strip() for col in data.columns]

# Rename columns if necessary
if 'AEP_MW' in data.columns:
    data.rename(columns={'AEP_MW': 'MW'}, inplace=True)
if 'Datetime' not in data.columns:
    data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values('Datetime')

print("✅ Dataset loaded successfully!")
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())

# --- Scale Values ---
print("⚙️ Scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(np.array(data['MW']).reshape(-1, 1))

# --- Create Features ---
def create_dataset(dataset, time_step=24):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 24
X, y = create_dataset(scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --- Split Data ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- Build LSTM Model ---
print("🧠 Building LSTM model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# --- Evaluate Model ---
print("📈 Evaluating model...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scale predictions
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compute metrics
train_rmse = math.sqrt(mean_squared_error(y_train_inv, train_pred))
test_rmse = math.sqrt(mean_squared_error(y_test_inv, test_pred))

print(f"✅ Train RMSE: {train_rmse:.2f}")
print(f"✅ Test RMSE: {test_rmse:.2f}")

# --- Save Model and Scaler ---
os.makedirs('model', exist_ok=True)
model.save('model/energy_lstm.keras')  # ✅ modern format
joblib.dump(scaler, 'model/scaler.pkl')
print("💾 Model and scaler saved successfully!")

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[-200:], label='Actual', color='blue')
plt.plot(test_pred[-200:], label='Predicted', color='red')
plt.legend()
plt.title("Energy Demand Forecast (Validation)")
plt.show()
