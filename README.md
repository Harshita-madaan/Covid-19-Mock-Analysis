# COVID-19 Global Data Analysis (Mock Data)

## Project Overview
This project explores the global spread and impact of COVID-19 using a mock dataset created for learning and demonstration purposes. The analysis focuses on practicing data cleaning, aggregation, trend analysis, and data visualization using Python.

## Disclaimer
This project uses **synthetically generated mock data** and does **not** represent real COVID-19 statistics. It is intended solely for educational and skill-building purposes.

## Objectives

- Practice data cleaning, grouping, and filtering using Pandas  
- Analyze trends in COVID-19 cases across multiple countries  
- Calculate death rates and rolling averages  
- Visualize trends using bar and line plots  
- Generate insights and document findings  

## Dataset

- **File:** mock_covid_data_200_rows.csv  
- **Records:** 200 synthetic rows  
- **Fields:**  
  - location  
  - date  
  - total_cases  
  - new_cases  
  - total_deaths  
  - population  
  - continent  

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  

## Key Analysis Performed

- Country-wise grouped analysis of COVID-19 cases  
- Death rate calculation based on total cases and deaths  
- Rolling 7-day average of new cases  
- Line plots to visualize daily case trends  
- Bar plots highlighting highly affected countries  

## Key Learnings

- Handling time-series data and rolling metrics  
- Understanding the importance of normalization and rate-based metrics  
- Visualizing trends to extract meaningful insights  
- Interpreting analytical results using synthetic datasets  

## Notes
This is a mock data project designed to demonstrate data analysis workflows and visualization techniques. It is not intended for medical, epidemiological, or public health decision-making.



## Note on Dataset

This project uses a mock COVID-19 dataset for educational and analytical purposes. The values are synthetic and should not be interpreted as actual historical COVID-19 statistics or used for real-world forecasting.

## Predictive Analysis Using Regression

Regression models were added to extend the existing COVID-19 data analysis project with a basic machine learning component.

### Models Implemented

- **Linear Regression:** Predicts daily new cases using the previous three days' case counts.
- **Polynomial Regression:** Uses polynomial features to model nonlinear relationships.
- **Ridge Regression:** Applies regularization to Linear Regression.

### Methodology

- Aggregated new COVID-19 cases across locations by date.
- Created features using the previous three days of case counts.
- Split the data chronologically into 80% training data and 20% testing data.
- Trained and compared three regression models.
- Evaluated predictions using Mean Absolute Error (MAE) and R² score.
- Visualized actual and predicted case counts.

### Model Comparison Results

| Model | MAE | R² Score |
|---|---:|---:|
| Linear Regression | 43178.64 | 0.0364 |
| Polynomial Regression | 67843.71 | -1.0048 |
| Ridge Regression | 43178.64 | 0.0364 |

Linear Regression and Ridge Regression achieved the lowest MAE in this comparison. However, the low R² scores indicate limited predictive performance on the mock dataset. Polynomial Regression performed worse on the test data.

### Additional Files

- `linear_regression.py` — Linear Regression model for COVID-19 case prediction.
- `compare_regression.py` — Comparison of Linear, Polynomial, and Ridge Regression.
- `linear_regression_results.png` — Actual versus predicted cases for Linear Regression.
- `regression_comparison.png` — Visual comparison of the regression models.

### Limitations

The models are intended for educational demonstration. Their performance is limited on this synthetic dataset, and the predictions should not be interpreted as reliable real-world COVID-19 forecasts.
