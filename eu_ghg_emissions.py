#!/usr/bin/env python
# coding: utf-8

import csv
import json
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

csvpath = r'UNFCCC_v26.csv'
jsonpath = r'UNFCCC_v26.json'

def convert_csv_into_json(csvpath, jsonpath):
    array4json = []
      
    #reading the csv file from csv path
    with (open(csvpath, encoding='utf-8') as csvf): 
        csvReader = csv.DictReader(csvf) 

        rowcount = 0
        for row in csvReader: 
            #Read only Car specific and Total data
            if  row['Pollutant_name'] == 'All greenhouse gases - (CO2 equivalent)' and (row['Sector_name'] == '1.A.3.b.i - Cars'  or row['Sector_name'] == 'Total emissions (UNFCCC)') :
                array4json.append(row)
                rowcount += 1
    with open(jsonpath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(array4json, indent=4)
        jsonf.write(jsonString)

#convert_csv_into_json(csvpath, jsonpath)


createStringGreenHouseCO2 = """
CREATE TABLE IF NOT EXISTS GreenHouse_CO2 (
Country_code char(20),
Country	varchar(50),
Format_name	varchar(50),
Pollutant_name	varchar(200),
Sector_code	varchar(50),
Sector_name	varchar(300),
Parent_sector_code	varchar(30),
Unit	varchar(30),
Year	integer,
emissions	numeric, 
Notation varchar(30),
PublicationDate integer,
DataSource varchar(30)
);
"""

readDataGreenHouseCO2 = 'select count(*) from GreenHouse_CO2;'


# In[51]:


try :
    dbConnection = psycopg2.connect(
        user = "dap",
        password = "dap",
        host = "192.168.56.30",
        port = "5432",
        database = "climate")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    #dbCursor.execute("drop table IF EXISTS GreenHouse_CO2;")
    #dbCursor.execute(createStringGreenHouseCO2)
    #dbCursor.execute(readDataGreenHouseCO2)
    print (dbCursor.fetchall())
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print("Successfully created table GreenHouse_CO2")


try:
    dbConnection = psycopg2.connect(
                            user = "dap",
                            password = "dap",
                            host = "192.168.56.30",
                            port = "5432",
                            database = "climate")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    insertString = "INSERT INTO GreenHouse_CO2 VALUES ('{}'," + "'{}',"*8 + "cast(coalesce(nullif('{}',''),'0') as float),'{}','{}','{}')"
    i = 0
    with open('UNFCCC_v26.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header
        for row in reader:
            #print(insertString.format(*row))
            if  row[3] == 'All greenhouse gases - (CO2 equivalent)' and (row[5] == '1.A.3.b.i - Cars'  or row[5] == 'Total emissions (UNFCCC)') and row[8] != '1985-1987'  :
                #dbCursor.execute(insertString.format(*row))
                i += 1
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error:", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print("Successfully Inserted data in to table GreenHouse_CO2")

# In[95]:


readCO2Emissions_cars = '''Select Year,  
    sum(emissions) as CarsEmission
    from GreenHouse_CO2
    where Sector_name = '1.A.3.b.i - Cars' and Country_code = 'EUA'
    group by  Year
'''

readCO2Emissions_ALL = '''Select Year,
    sum(emissions) as TotalEmission
    from GreenHouse_CO2
    where Sector_name = 'Total emissions (UNFCCC)' and Country_code = 'EUA'
    group by  Year
'''


try:
    dbConnection = psycopg2.connect(user = "dap",
    password = "dap",
    host = "192.168.56.30",
    port = "5432",
    database = "climate")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute(readCO2Emissions_cars)
  
    CO2_cars_Only = dbCursor.fetchall(); 

    dbCursor.execute(readCO2Emissions_ALL)
    CO2_all_sector = dbCursor.fetchall(); 

    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print('Successfully read readCO2Emissions_cars and readCO2Emissions_ALL from table GreenHouse_CO2')
# In[437]:



df = pd.DataFrame(CO2_cars_Only, columns =['Year', 'CarsEmission'])
sorted_df_cars = df.sort_values('Year').reindex()

df1 = pd.DataFrame(CO2_all_sector, columns =['Year', 'AllEmission'])
sorted_df_all = df1.sort_values('Year').reindex()

df_combined = pd.merge(sorted_df_cars, sorted_df_all, on='Year')

plt.figure(figsize=(12, 6))
plt.plot(df_combined['Year'], df_combined['CarsEmission'], label = 'Cars Only', marker='o')
plt.plot(df_combined['Year'], df_combined['AllEmission'], label = 'Total CO2 Emissions - All Industries', marker='*')

n = 4  

for i, (year, CarsEmissionDataPoint) in enumerate(zip(df_combined['Year'], df_combined['CarsEmission'])):
    if i % n == 0:
        plt.text(year, CarsEmissionDataPoint, f'{CarsEmissionDataPoint:,.0f}', ha='center', va='top', color='yellow')

for year, allEmissionDataPoint in zip(df_combined['Year'], df_combined['AllEmission']):
    if year % n == 0:
        plt.text(year, allEmissionDataPoint, f'{allEmissionDataPoint:,.0f}', ha='center', va='bottom', color='green')

plt.title("EU27 Countries Greenhouse gases - (CO2 equivalent)")
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()

print('Plotting "EU27 Countries Greenhouse gases - (CO2 equivalent)" is complete')

# In[442]:



readEmissionsActualAndForecastString = '''select * from EU_CARS_ACTUAL_AND_FORECAST;'''


try :
    dbConnection = psycopg2.connect(
        user = "dap",
        password = "dap",
        host = "192.168.56.30",
        port = "5432",
        database = "climate")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute(readEmissionsActualAndForecastString)
    output = dbCursor.fetchall()
    columns = ['Year', 'EVCars', 'EVCarsPercentage', 'AllCars_Total']
    df_EU_CARS_ACTUAL_AND_FORECAST = pd.DataFrame(output, columns=columns)
    #print (df_EU_CARS_ACTUAL_AND_FORECAST)
    df_EU_CARS_ACTUAL_AND_FORECAST.set_index("Year", inplace=True)
    #print (df_EU_CARS_ACTUAL_AND_FORECAST)
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection):
        dbConnection.close()

print('Successfully read data from table EU_CARS_ACTUAL_AND_FORECAST')



# In[1]:


values_for_plot = df_combined['CarsEmission'].values


#2020 and 2021 CO2 emission values are low during Covid hance using 2019 data for forecasting
values_for_forecast = df_combined
print(values_for_forecast)
print(values_for_forecast.set_index("Year", inplace=True))
values_for_forecast.loc[2020] = values_for_forecast.loc[2019] 
values_for_forecast.loc[2021] = values_for_forecast.loc[2019] 

# Fit Exponential Smoothing model
model = ExponentialSmoothing(values_for_forecast['CarsEmission'].astype(float),  trend='add', seasonal=None)
fit_model = model.fit()

# Forecasting
forecast_steps = 11
forecast = fit_model.forecast(steps=forecast_steps)

EU_cars_last_forecast_steps =  df_EU_CARS_ACTUAL_AND_FORECAST.iloc[-forecast_steps:]


custom_index = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
forecast.index = custom_index
EU_cars_last_forecast_steps_pd_series = EU_cars_last_forecast_steps['EVCarsPercentage'].astype(float)

forecast_less_ev_percentage = forecast*(1 - EU_cars_last_forecast_steps_pd_series/100) 

forecast_index = np.arange(df_combined.index.max() +1, df_combined.index.max() + forecast_steps + 1)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_combined.index.values,  values_for_plot, label='Cars CO2 Emission Original Data', marker='o')
plt.plot(forecast_index, forecast, label='Cars CO2 Emission Forecast', color='red', linestyle='--', marker='o')
plt.plot(forecast_index, forecast_less_ev_percentage, label='Cars CO2 Emission Forecast less EV Cars share %', color='green', linestyle='--', marker='*')

#print(forecast_less_ev_percentage)
n = 3  

for year, CarsEmissionDataPoint in zip(df_combined.index.values, values_for_plot):
    if year % n == 0:
        plt.text(year, CarsEmissionDataPoint, f'{CarsEmissionDataPoint:,.0f}', ha='center', va='bottom', color='green')
n=5
for i, (year, ForecastCarsEmissionDataPoint, ForecastCarsEmissionLessEVDataPoint) in enumerate(zip(forecast_index, forecast,forecast_less_ev_percentage)):
    if i % n == 0:
        plt.text(year, ForecastCarsEmissionDataPoint, f'{ForecastCarsEmissionDataPoint:,.0f}', ha='center', va='bottom', color='yellow')
        plt.text(year, ForecastCarsEmissionLessEVDataPoint, f'{ForecastCarsEmissionLessEVDataPoint:,.0f}', ha='center', va='top', color='cyan')

current_ylim = plt.ylim()

new_upper_limit = current_ylim[1] * 1.3 
plt.ylim(bottom=0, top=new_upper_limit)

plt.title('Exponential Smoothing Forecasting - EU27 Countries Greenhouse gases - (For Cars Only)')
plt.xlabel('Year')
plt.ylabel('CO2Emission')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(values_for_plot, fit_model.fittedvalues)
rmse = sqrt(mse)

print('Plotting "Exponential Smoothing Forecasting - EU27 Countries Greenhouse gases - (For Cars Only)" is complete')



print('*******df_combined**********')
print(df_combined)
print('*******forecast**********')
print(forecast)
print('*******forecast_less_ev_percentage**********')
print(forecast_less_ev_percentage)

createEmissionsActualAndForecastString = """
CREATE TABLE IF NOT EXISTS EMISSIONS_CARS_ACTUAL_AND_FORECAST (
Year	integer,
CarsEmission	numeric,
AllEmission	numeric
);
"""

readEmissionsActualAndForecastString = 'select * from EMISSIONS_CARS_ACTUAL_AND_FORECAST;'

try:
    dbConnection = psycopg2.connect(
        user="dap",
        password="dap",
        host="192.168.56.30",
        port="5432",
        database="climate")
    dbConnection.set_isolation_level(0)  # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute("drop table  IF EXISTS EMISSIONS_CARS_ACTUAL_AND_FORECAST;")
    dbCursor.execute(createEmissionsActualAndForecastString)
    dbCursor.execute(readEmissionsActualAndForecastString)
    print(dbCursor.fetchall())
    dbCursor.close()
except (Exception, psycopg2.Error) as dbError:
    print("Error while connecting to PostgreSQL", dbError)
finally:
    if (dbConnection):
        dbConnection.close()

try:
    dbConnection = psycopg2.connect(
        user="dap",
        password="dap",
        host="192.168.56.30",
        port="5432",
        database="climate")
    dbConnection.set_isolation_level(0)  # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    insertStringEmissionsActualAndForecast = "INSERT INTO EMISSIONS_CARS_ACTUAL_AND_FORECAST VALUES ({},{},{})"

    print(insertStringEmissionsActualAndForecast)

    for index, row in df_combined.iterrows():
        # print(row)
        dbCursor.execute(
            insertStringEmissionsActualAndForecast.format(index, row['CarsEmission'], row['AllEmission']))

    for index, row in forecast_less_ev_percentage.items():
        # print(row)
        dbCursor.execute(
            insertStringEmissionsActualAndForecast.format(index, row, 0.0))


    dbCursor.close()
    dbConnection.close()
except (Exception, psycopg2.Error) as dbError:
    print("Error:", dbError)
finally:
    if (dbConnection):
        dbConnection.close()
print('Successfully inserted data into table EMISSIONS_CARS_ACTUAL_AND_FORECAST')

try:
    dbConnection = psycopg2.connect(
        user="dap",
        password="dap",
        host="192.168.56.30",
        port="5432",
        database="climate")
    dbConnection.set_isolation_level(0)  # AUTOCOMMIT
    dbCursor = dbConnection.cursor()

    dbCursor.execute(readEmissionsActualAndForecastString)
    print(dbCursor.fetchall())

    dbCursor.close()
    dbConnection.close()
except (Exception, psycopg2.Error) as dbError:
    print("Error:", dbError)
finally:
    if (dbConnection):
        dbConnection.close()

print('Successfully read from table EMISSIONS_CARS_ACTUAL_AND_FORECAST')

print("End of eu_ghg_emissions")