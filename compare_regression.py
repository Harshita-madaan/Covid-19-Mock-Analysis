import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load and prepare data
df = pd.read_csv("mock_covid_data_200_rows.csv")
df["date"] = pd.to_datetime(df["date"])

daily = df.groupby("date", as_index=False)["new_cases"].sum()
daily = daily.sort_values("date").reset_index(drop=True)

# Use previous 3 days to predict the next day
for lag in range(1, 4):
    daily[f"day_{lag}"] = daily["new_cases"].shift(lag)

daily = daily.dropna().reset_index(drop=True)

X = daily[["day_1", "day_2", "day_3"]]
y = daily["new_cases"]

# Split data by time: first 80% for training, last 20% for testing
split = int(len(daily) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Three regression models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    ),
    "Ridge Regression": Ridge(alpha=1.0)
}

results = {}
predictions = {}

# Train, predict and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    predictions[name] = pred
    results[name] = {
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "R2 Score": round(r2_score(y_test, pred), 4)
    }

# Display comparison
comparison = pd.DataFrame(results).T
print("\nREGRESSION MODEL COMPARISON")
print(comparison.to_string())

best_model = comparison["MAE"].idxmin()
print("\nBest model based on lowest MAE:", best_model)

# Plot actual and predicted cases
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, "o-", label="Actual Cases")

for name, pred in predictions.items():
    plt.plot(pred, "o--", label=name)

plt.xlabel("Test Days")
plt.ylabel("Daily New Cases Across Locations")
plt.title("COVID-19 Regression Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("regression_comparison.png")
plt.show()
