import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("mock_covid_data_200_rows.csv")
df["date"] = pd.to_datetime(df["date"])

# Combine cases across all locations for each date
daily = (
    df.groupby("date", as_index=False)["new_cases"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

# Use previous 3 days to predict the next day
for lag in range(1, 4):
    daily[f"previous_day_{lag}"] = daily["new_cases"].shift(lag)

daily = daily.dropna().reset_index(drop=True)

X = daily[["previous_day_1", "previous_day_2", "previous_day_3"]]
y = daily["new_cases"]

# Chronological train-test split (no random shuffling)
split = int(len(daily) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test cases
predictions = model.predict(X_test)

# Evaluate
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, "o-", label="Actual Cases")
plt.plot(range(len(predictions)), predictions, "o-", label="Predicted Cases")
plt.xlabel("Test Days")
plt.ylabel("Daily New Cases Across Locations")
plt.title("COVID-19 Prediction Using Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_results.png")
plt.show()

# Predict the next day using the last 3 observed days
last_three = daily["new_cases"].tail(3).values[::-1]
next_input = pd.DataFrame([last_three], columns=[
    "previous_day_1", "previous_day_2", "previous_day_3"
])
next_prediction = model.predict(next_input)[0]

print("Predicted new cases for the next day:", round(max(0, next_prediction), 2))
