
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Generate sample network bandwidth data
np.random.seed(42)

data = pd.DataFrame({
    "Bandwidth": 50 + np.random.randn(200).cumsum() * 0.5
})

# Create previous bandwidth features
for i in range(1, 4):
    data[f"Previous_{i}"] = data["Bandwidth"].shift(i)

data.dropna(inplace=True)

# Prepare input and output
X = data[["Previous_1", "Previous_2", "Previous_3"]]
y = data["Bandwidth"]

# Split data chronologically
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predict bandwidth
predictions = model.predict(X_test)

# Display results
error = mean_absolute_error(y_test, predictions)
print("AI-Based Network Bandwidth Prediction")
print("Mean Absolute Error:", round(error, 2))

# Predict next bandwidth reading
last = data["Bandwidth"].iloc[-3:].to_numpy()[::-1]
next_bandwidth = model.predict([last])[0]

print("Current Bandwidth:", round(data["Bandwidth"].iloc[-1], 2))
print("Predicted Next Bandwidth:", round(next_bandwidth, 2))

# Plot results
plt.plot(y_test.to_numpy(), label="Actual Bandwidth")
plt.plot(predictions, label="Predicted Bandwidth")
plt.xlabel("Time")
plt.ylabel("Bandwidth (sample units)")
plt.title("Short-Term Network Bandwidth Prediction")
plt.legend()
plt.show()
