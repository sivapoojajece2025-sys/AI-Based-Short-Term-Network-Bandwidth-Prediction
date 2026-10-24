
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Project: AI-Based Short-Term Network Bandwidth Prediction

# 1. Load dataset
# CSV file should contain a column named 'bandwidth'
file_name = "bandwidth_data.csv"

try:
    df = pd.read_csv(file_name)
    if "bandwidth" not in df.columns:
        raise ValueError("CSV must contain a 'bandwidth' column.")
    bandwidth = pd.to_numeric(df["bandwidth"], errors="raise").to_numpy()

except FileNotFoundError:
    print("Dataset not found. Generating demo data...")
    np.random.seed(42)

    time = np.arange(500)
    bandwidth = (
        50
        + 10 * np.sin(time / 15)
        + 5 * np.sin(time / 5)
        + np.random.normal(0, 2, 500)
    )

# Validate data
if len(bandwidth) < 30:
    raise ValueError("Please provide at least 30 bandwidth readings.")

if not np.isfinite(bandwidth).all():
    raise ValueError("Dataset contains missing or invalid values.")

# 2. Create time-series features
data = pd.DataFrame({"bandwidth": bandwidth})

data["previous_1"] = data["bandwidth"].shift(1)
data["previous_2"] = data["bandwidth"].shift(2)
data["previous_3"] = data["bandwidth"].shift(3)
data["previous_5"] = data["bandwidth"].shift(5)
data["rolling_mean"] = data["bandwidth"].shift(1).rolling(5).mean()

data = data.dropna()

features = [
    "previous_1",
    "previous_2",
    "previous_3",
    "previous_5",
    "rolling_mean",
]

X = data[features]
y = data["bandwidth"]

# 3. Split data chronologically
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Train the Machine Learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Predict bandwidth
predictions = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nAI-Based Short-Term Network Bandwidth Prediction")
print("-" * 45)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 7. Predict the next bandwidth reading
last_values = pd.Series(bandwidth)

next_features = pd.DataFrame([{
    "previous_1": last_values.iloc[-1],
    "previous_2": last_values.iloc[-2],
    "previous_3": last_values.iloc[-3],
    "previous_5": last_values.iloc[-5],
    "rolling_mean": last_values.iloc[-5:].mean(),
}])

next_prediction = model.predict(next_features)[0]

print(f"\nLast bandwidth reading: {bandwidth[-1]:.2f}")
print(f"Predicted next bandwidth: {next_prediction:.2f}")

# 8. Plot actual vs predicted values
plt.figure(figsize=(10, 5))

plt.plot(
    y_test.to_numpy(),
    label="Actual Bandwidth"
)

plt.plot(
    predictions,
    label="Predicted Bandwidth"
)

plt.title("Short-Term Network Bandwidth Prediction")
plt.xlabel("Test Time Steps")
plt.ylabel("Bandwidth (dataset units)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bandwidth_prediction.png")
plt.show()
