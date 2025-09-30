# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

```
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('seattle_weather.csv')

# Inspect the first few rows to understand the structure
print(data.head())

# Convert 'DATE' to datetime format and set it as the index
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Resample data to monthly frequency, calculating the mean precipitation per month
monthly_data = data['PRCP'].resample('M').mean().dropna()

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test
result = adfuller(monthly_data)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = monthly_data.iloc[:int(0.8 * len(monthly_data))]
test_data = monthly_data.iloc[int(0.8 * len(monthly_data)):]

# Define the lag order for the AutoRegressive model based on ACF/PACF plots
lag_order = 12  # Monthly lag (seasonality) might be appropriate
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(monthly_data, lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Monthly Precipitation')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(monthly_data, lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Monthly Precipitation')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for predictions
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Test Data - Monthly Precipitation', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Monthly Precipitation', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Precipitation (inches)')
plt.title('AR Model Predictions vs Test Data (Monthly Precipitation)')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:

GIVEN DATA
<img width="255" height="622" alt="image" src="https://github.com/user-attachments/assets/eb0e84c0-e58c-4733-9a95-ce61f23d9647" />


PACF - ACF

<img width="346" height="98" alt="image" src="https://github.com/user-attachments/assets/01609931-0469-4438-b4e2-d5cbea40ed1f" />
<img width="759" height="509" alt="image" src="https://github.com/user-attachments/assets/a9be15dc-a50c-4666-ae8e-d08f217888b1" />

<img width="780" height="511" alt="image" src="https://github.com/user-attachments/assets/a2d9206a-10ec-4106-879d-a10d8a7cdfac" />



FINIAL PREDICTION
<img width="1048" height="584" alt="image" src="https://github.com/user-attachments/assets/ce926ed4-0e17-4414-95f8-639a9c62094a" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
