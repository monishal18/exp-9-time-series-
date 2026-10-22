# EXPERIMENT 9: BOX–JENKINS MODEL 
# NAME: MONISHA L
# REG NO :2305001019

## Aim
To understand and implement the Box–Jenkins approach to time-series forecasting using an ARIMA model in Python.

## Objective
1. To create time-series data using Python.
2. To check stationarity using the Augmented Dickey–Fuller (ADF) test.
3. To apply differencing to make the series stationary.
4. To identify AR and MA orders using ACF and PACF plots.
5. To fit an ARIMA model and estimate its parameters.
6. To perform diagnostic checking and forecast future values.

## Algorithm
1. Start the program.
2. Import the required libraries: Pandas, Matplotlib, and Statsmodels.
3. Create sample monthly sales data for 24 months.
4. Convert the sales data into a Pandas Series.
5. Plot the original time-series data.
6. Check stationarity using the ADF test.
7. Apply first-order differencing and check stationarity again.
8. Plot the ACF and PACF graphs.
9. Fit the ARIMA model with order (1, 1, 1).
10. Display the model summary, AIC, and BIC values.
11. Perform diagnostic checking using residual plots and the Ljung–Box test.
12. Forecast sales for the next six months.
13. Plot actual sales and forecasted values with confidence intervals.
14. Display the result and stop the program.

## Program

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Create sample monthly sales data
sales = [
    100, 105, 103, 110, 115, 112,
    120, 125, 123, 130, 135, 132,
    140, 145, 143, 150, 155, 152,
    160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("Original Sales Data")
print(data.to_string(index=False))

# Plot original time series
plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test on original data
adf_result = adfuller(data)

print("\nADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))

if adf_result[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Insufficient evidence of stationarity.")

# First differencing
diff_data = data.diff().dropna()

print("\nFirst-Differenced Data")
print(diff_data.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test after differencing
adf_diff = adfuller(diff_data)

print("\nADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))

if adf_diff[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Stationarity is not confirmed.")

# ACF and PACF plots
plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()

plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()

# Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nARIMA Model Summary")
print(fitted_model.summary())

# AIC and BIC
print("\nModel Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))

# Residual diagnostic checking
residuals = fitted_model.resid

plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()

# Ljung-Box test
lb_result = acorr_ljungbox(
    residuals,
    lags=[5],
    return_df=True
)

print("\nLjung-Box Test Result")
print(lb_result)

if lb_result["lb_pvalue"].iloc[0] > 0.05:
    print("No significant residual autocorrelation detected at lag 5.")
else:
    print("Residual autocorrelation may remain; review the model.")

# Forecast next six months
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nForecast for the Next Six Months")

for month, value in enumerate(forecast_mean, start=25):
    print(f"Month {month}: {value:.2f}")

print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))

# Plot actual data and forecast
future_months = range(len(data), len(data) + 6)

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(data)),
    data,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    future_months,
    forecast_mean.to_numpy(),
    marker="o",
    linestyle="--",
    label="Forecast"
)

plt.fill_between(
    future_months,
    forecast_ci.iloc[:, 0].to_numpy(),
    forecast_ci.iloc[:, 1].to_numpy(),
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Result
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed.")
print("Identification, estimation, diagnostics, and forecasting performed.")
```

## Expected Output
## Original monthly sales data and graph.

- <img width="898" height="558" alt="image" src="https://github.com/user-attachments/assets/ec1af91d-64f8-44e9-a4e6-0d14968b2dc6" />

<img width="983" height="367" alt="image" src="https://github.com/user-attachments/assets/42a8e953-bb7e-4203-83f5-86bacaddf95c" />


  ## ADF test results before and after differencing.

  <img width="667" height="116" alt="image" src="https://github.com/user-attachments/assets/8e5aaeb4-fb1d-4bc1-89ad-a10a684954df" />

  ## First-differenced sales data and graph.

  <img width="470" height="526" alt="image" src="https://github.com/user-attachments/assets/00cc789f-ed03-4892-a322-18cecca5d09a" />

  <img width="995" height="376" alt="image" src="https://github.com/user-attachments/assets/f39f12fb-da9a-4e81-82ee-8483102295ae" />


 ##  ACF and PACF plots.

 
 <img width="810" height="572" alt="image" src="https://github.com/user-attachments/assets/d3dc2137-9540-4c0b-8d34-c35c1fb3e438" />
 <img width="800" height="567" alt="image" src="https://github.com/user-attachments/assets/9bfdecbd-9e2e-4605-98f5-5a17493dc3cc" />


  ## VARIMA model summary with AIC and BIC values.

  
  <img width="845" height="502" alt="image" src="https://github.com/user-attachments/assets/df465b19-42d7-4416-a881-574879e1bd7a" />

  ## Residual diagnostic plots and Ljung–Box test.

  <img width="847" height="590" alt="image" src="https://github.com/user-attachments/assets/512a98d2-c403-4a42-9431-30b954c8a302" />

  <img width="715" height="122" alt="image" src="https://github.com/user-attachments/assets/d5e96127-ff99-416c-a924-911a3bd5e1a8" />


 ## Six-month sales forecast with confidence intervals.

 <img width="592" height="355" alt="image" src="https://github.com/user-attachments/assets/59df681d-7616-4c8c-b101-bf885b467b22" />

##  Actual versus forecasted sales graph.
<img width="1001" height="470" alt="image" src="https://github.com/user-attachments/assets/a5838ff5-d040-4f72-9505-a68f9f8eadb4" />


## Result
Thus, the Box–Jenkins approach was implemented successfully using the ARIMA model in Python. Stationarity testing, differencing, ACF/PACF analysis, model estimation, diagnostic checking, and forecasting were performed.


