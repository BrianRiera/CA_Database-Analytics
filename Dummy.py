import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming you have a DataFrame 'df' with a datetime index and a column 'value'
#df_combined = pd.read_csv('your_data.csv', parse_dates=['date_column'], index_col='date_column')

# Visualize the time series
df_combined['CarsEmission'].plot(figsize=(12, 6), title='Original Time Series')
plt.show()

# Function to find the best parameters for ARIMA
def find_best_arima_params(data, p_values, d_values, q_values):
    best_score, best_params = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(data, order=order)
                    results = model.fit()
                    mse = mean_squared_error(data, results.fittedvalues)
                    if mse < best_score:
                        best_score, best_params = mse, order
                except Exception as e:
                    print(f"Error for order {order}: {e}")

    return best_params

# Specify the ranges for p, d, q
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

# Find the best parameters
best_params = find_best_arima_params(df_combined['CarsEmission'], p_values, d_values, q_values)

# Fit the ARIMA model with the best parameters
final_model = ARIMA(df_combined['CarsEmission'], order=best_params)
final_results = final_model.fit()

# Make predictions
forecast_steps = 12  # Adjust the number of steps into the future
forecast = final_results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df_combined.index.max(), periods=forecast_steps + 1, freq='M')[1:]

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_combined.index, df_combined['value'], label='Original Data')
plt.plot(final_results.fittedvalues.index, final_results.fittedvalues, label='Fitted Values', color='green')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast.conf_int()['lower value'], forecast.conf_int()['upper value'], color='red', alpha=0.2)

plt.title('Time Series Forecasting with ARIMA')
plt.xlabel('Year')
plt.ylabel('CarsEmission')
plt.legend()
plt.show()
