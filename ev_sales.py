#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import csv
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import constants as ct

def createDB(database):
    try:
        dbConnection = psycopg2.connect(
            user=ct.psql_user,
            password=ct.psql_password,
            host=ct.psql_host,
            port=ct.psql_port,
            database=ct.psql_masterDb)

        dbConnection.set_isolation_level(0)
        dbCursor = dbConnection.cursor()
        dbCursor.execute(f'CREATE DATABASE {database};')
        dbCursor.close()
    except (Exception, psycopg2.Error) as dbError:
        print("Error while connecting to PostgreSQL", dbError)
    finally:
        if (dbConnection):
            dbConnection.close()

createDB(ct.psql_database)

print("Inside ev_sales")

df = pd.read_csv(ct.ev_global_data_csvpath_and_file)

result = df.loc[(df['region'] == 'EU27') & (df['category'] == 'Historical') & (df['parameter'] == 'EV stock') & (df['mode'] == 'Cars') ].groupby(["year"]).agg(EVStock=('value', 'sum'))

# plotting a line graph 
fig, ax = plt.subplots()

n = 2 # Variable to Adjust the data point marking gaps
ax.plot(result.index, result['EVStock'], marker='o', markersize=5, label='EVStock')

for i, (year, ev_stock) in enumerate(result['EVStock'].items()):
    if i % n == 0:
        ax.text(year, ev_stock, "{:,.0f}".format(ev_stock), ha="center", va="bottom")

plt.xlabel('Year')
plt.ylabel('Number of Units')
plt.legend()
plt.title("Stock of Electric Vehicles in EU27 Countries")
plt.show()
print('Plotting "Stock of Electric Vehicles in EU27 Countries" is complete')

createStringGlobalEVDataTable = """
CREATE TABLE IF NOT EXISTS GLOBAL_EV_DATA (
Region	varchar(50),
Category varchar(50),
Parameter	varchar(50),
Mode	varchar(200),
Powertrain	varchar(50),
Year	integer,
unit	varchar(300),
value	numeric
);
"""

readGlobalEVData = 'select count(*) from GLOBAL_EV_DATA;'

try :
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute("drop table IF EXISTS GLOBAL_EV_DATA;")
    dbCursor.execute(createStringGlobalEVDataTable)
    dbCursor.execute(readGlobalEVData)
    print (dbCursor.fetchall())
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()
print('Successfully created table GLOBAL_EV_DATA')

try:
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    insertString = "INSERT INTO GLOBAL_EV_DATA VALUES ('{}'," + "'{}',"*6 + "{})"

    with open(ct.ev_global_data_csvpath_and_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header
        for row in reader:
            dbCursor.execute(insertString.format(*row))
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error:", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print('Successfully inserted data in to table GLOBAL_EV_DATA')

totalEVStock_Cars = '''Select Year,  
    Sum(Value) as TotalEVCars
    from GLOBAL_EV_DATA
    where Region = 'EU27' and Category = 'Historical' and Parameter = 'EV stock' and mode = 'Cars'
    group by  Year
'''


totalEVStock_Percent = '''Select Year,  
    Value as TotalEVCarsPercentage
    from GLOBAL_EV_DATA
    where Region = 'EU27' and Category = 'Historical' and Parameter = 'EV stock share' and mode = 'Cars'
'''


try:
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute(totalEVStock_Cars)
    allEVCars = dbCursor.fetchall(); 
    dbCursor.execute(totalEVStock_Percent)
    allEVCarsPercentage = dbCursor.fetchall(); 
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print('Successfully read data from GLOBAL_EV_DATA')




df = pd.DataFrame(allEVCars, columns =['Year', 'EVCars'])
sorted_df_cars = df.sort_values('Year').reindex()

df1 = pd.DataFrame(allEVCarsPercentage, columns =['Year', 'EVCarsPercentage'])
sorted_df_all = df1.sort_values('Year').reindex()

df_combined = pd.merge(sorted_df_cars, sorted_df_all, on='Year')

# using apply function to create a new column
df_combined['AllCars_Total'] = df_combined.apply(lambda row: 100*row.EVCars/row.EVCarsPercentage, axis = 1)

df_combined.set_index("Year", inplace=True)

#print(df_combined)

plt.figure(figsize=(12, 6))
plt.plot(df_combined.index, df_combined['EVCars'], label = 'EV Cars Total', linestyle='--', marker='o')
plt.plot(df_combined.index, df_combined['AllCars_Total'], label = 'Total of All Cars', linestyle='--', marker='*')

n = 3 # Variable to Adjust the data point marking gaps

for i, (year, ev_cars, all_cars) in enumerate(zip(df_combined.index, df_combined['EVCars'], df_combined['AllCars_Total'])):
    if i % n == 0:
        plt.text(year, ev_cars, f'{ev_cars:,.0f}', ha='left', va='bottom', color='yellow')
        plt.text(year, all_cars, f'{all_cars:,.0f}', ha='left', va='bottom', color='green')

plt.title("Stock of EV vs All Types of Cars in EU27 Countries")
plt.xlabel('Year')
plt.ylabel('Number of Units')
plt.legend()
plt.show()

print('Plotting "Stock of EV vs All Types of Cars in EU27 Countries" is complete')

df_raw = df_combined

# Splitting the data into training and testing sets
train_size = int(len(df_raw) * 0.8)
train, test = df_raw[:train_size], df_raw[train_size:]


# Fit ARIMA model
order = (8,2,1)  # ARIMA parameters (p, d, q)
model = ARIMA(df_raw['EVCars'].astype(float), order=order)
fit_model = model.fit()

forecast_steps = 10  # Adjust the number of steps as needed

# Forecasting
forecast = fit_model.get_forecast(steps=forecast_steps)
conf_int = forecast.conf_int()
forecast_values = forecast.predicted_mean

# Extend the time index for plotting
forecast_index = np.arange(df_raw.index.max() +1, df_raw.index.max() + forecast_steps + 1)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_raw.index, df_raw['EVCars'], label='Original Data')
plt.plot(train.index, train['EVCars'], label='Training Data', linestyle='--', marker='o')
plt.plot(test.index, test['EVCars'], label='Test Data', linestyle='--', marker='o')

# Plotting the ARIMA Forecast along with confidence interval
plt.plot(forecast_index[-forecast_steps:], forecast_values, label='ARIMA Forecast', color='red', linestyle='--', marker='o')

n = 2  

for i, (year, ev_cars) in enumerate(zip(df_raw.index, df_raw['EVCars'])):
    if i % n == 0:
        plt.text(year, ev_cars, f'{ev_cars:,.0f}', ha='center', va='bottom', color='yellow')

for year, forecast_value in zip(forecast_index[-forecast_steps:], forecast_values):
    if year % n == 0:
        plt.text(year, forecast_value, f'{forecast_value:,.0f}', ha='center', va='bottom', color='green')


plt.title('Time Series Forecasting - Stock of Electric Vehicles in EU27 Countries')
plt.xlabel('Year')
plt.ylabel('Units of EV Cars')
plt.legend()
plt.show()

print('Plotting of "Time Series Forecasting - Stock of Electric Vehicles in EU27 Countries" is complete')


df_allCars = df_combined

# Splitting the data into training and testing sets
train_size_all_cars = int(len(df_allCars) * 0.8)
train_all_cars, test_all_cars = df_allCars[:train_size_all_cars], df_allCars[train_size_all_cars:]


# Fit ARIMA model
order_all_cars = (1,1,1)  # ARIMA parameters (p, d, q)
model_all_cars = ARIMA(df_allCars['AllCars_Total'].astype(float), order=order_all_cars)
fit_model_all_cars = model_all_cars.fit()

forecast_steps_all_cars = 10  # Adjust the number of steps as needed

# Forecasting
forecast_all_cars = fit_model_all_cars.get_forecast(steps=forecast_steps_all_cars)
conf_int = forecast_all_cars.conf_int()
forecast_values_all_cars = forecast_all_cars.predicted_mean

#print(df_allCars.index.min())
# Extend the time index for plotting
forecast_index_all_cars = np.arange(df_allCars.index.max() +1, df_allCars.index.max() + forecast_steps_all_cars + 1)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_allCars.index, df_allCars['AllCars_Total'], label='All Cars Original Data', marker='x')
#plt.plot(train_all_cars.index, train_all_cars['AllCars_Total'], label='All Cars Training Data', linestyle='--', marker='o')
#plt.plot(test_all_cars.index, test_all_cars['AllCars_Total'], label='All Cars Test Data', linestyle='--', marker='o')

# Plot the ARIMA Forecast along with confidence interval
plt.plot(forecast_index_all_cars[-forecast_steps_all_cars:], forecast_values_all_cars, label='All Cars ARIMA Forecast', color='yellow', linestyle='--', marker='x')


plt.plot(df_raw.index, df_raw['EVCars'], label='EV Cars Original Data', marker='*')
#plt.plot(train.index, train['EVCars'], label='EV Cars Training Data', linestyle='--', marker='*')
#plt.plot(test.index, test['EVCars'], label='EV Cars Test Data', linestyle='--', marker='*')

# Plot the ARIMA Forecast along with confidence interval
plt.plot(forecast_index[-forecast_steps:], forecast_values, label='EV Cars ARIMA Forecast', color='cyan', linestyle='--', marker='*')

n = 3 # Variable to Adjust the data point marking gaps

for i, (year, ev_cars, all_cars) in enumerate(zip(df_combined.index, df_combined['EVCars'], df_combined['AllCars_Total'])):
    if i % n == 0:
        plt.text(year, ev_cars, f'{ev_cars:,.0f}', ha='center', va='bottom', color='yellow')
        plt.text(year, all_cars, f'{all_cars:,.0f}', ha='center', va='bottom', color='yellow')


n = 3  

for year, forecast_value, forecast_value_all_cars in zip(forecast_index[-forecast_steps:], forecast_values, forecast_values_all_cars):
    if year % n == 0:
        plt.text(year, forecast_value, f'{forecast_value:,.0f}', ha='center', va='top', color='green')
        plt.text(year, forecast_value_all_cars, f'{forecast_value_all_cars:,.0f}', ha='center', va='top', color='green')


plt.title('Time Series Forecasting - EV vs All Types of Cars')
plt.xlabel('Year')
plt.ylabel('Units of Cars')
plt.legend()
plt.show()

#print(test_all_cars['AllCars_Total'])
#print(forecast_values_all_cars.to_numpy())

print('Plotting of "Time Series Forecasting - EV vs All Types of Cars" is complete')
# In[433]:


import pandas as pd

df_final = pd.DataFrame(forecast_index, columns=['Year'])
df_final['EVCars'] = forecast_values.values
df_final['AllCars_Total'] = forecast_values_all_cars.values

df_final['EVCarsPercentage'] = 100*df_final['EVCars']/df_final['AllCars_Total']
df_final.set_index("Year", inplace=True)

df_final = df_final[df_combined.columns]
df_actual_and_predicted = pd.concat([df_combined, df_final])

#print(df_final)
#print(df_combined)
#print(df_actual_and_predicted)


# In[438]:


createActualAndForecastString = """
CREATE TABLE IF NOT EXISTS EU_CARS_ACTUAL_AND_FORECAST (
Year	integer,
EVCars	numeric,
EVCarsPercentage	numeric,
AllCars_Total	numeric
);
"""

readActualAndForecastString = 'select * from EU_CARS_ACTUAL_AND_FORECAST;'


try :
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    #dbCursor.execute("drop table  if exist EU_CARS_ACTUAL_AND_FORECAST;")
    dbCursor.execute(createActualAndForecastString)
    dbCursor.execute(readActualAndForecastString)
    print (dbCursor.fetchall())
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

try:
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    insertStringActualAndForecast = "INSERT INTO EU_CARS_ACTUAL_AND_FORECAST VALUES ({},{},{},{})"

    print(insertStringActualAndForecast)

    for index,row in df_actual_and_predicted.iterrows() :
        #print(row)
        dbCursor.execute(insertStringActualAndForecast.format(index, row['EVCars'], row['EVCarsPercentage'], row['AllCars_Total']))    
    
    dbCursor.close()
    dbConnection.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error:", dbError)
finally:
    if(dbConnection):
        dbConnection.close()
print('Successfully inserted data into table EU_CARS_ACTUAL_AND_FORECAST')

try:
    dbConnection = psycopg2.connect(
        user = ct.psql_user,
        password = ct.psql_password,
        host = ct.psql_host,
        port = ct.psql_port,
        database = ct.psql_database)

    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()

    dbCursor.execute(readActualAndForecastString)    
    print (dbCursor.fetchall())

    dbCursor.close()
    dbConnection.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error:", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print('Successfully read from table EU_CARS_ACTUAL_AND_FORECAST')

print("End of EV Sales")