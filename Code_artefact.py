#!/usr/bin/env python
# coding: utf-8

# In[98]:


from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import pymongo
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2


# In[99]:


# Commit history is in separate ipynb files 


# In[101]:


######################### co2 xml Datasets ########################################
####################################################################################
####################################################################################


# In[159]:


co2_client = pymongo.MongoClient('192.168.56.30', 27017)
co2_db = co2_client.climate
co2_collection = co2_db.co2_emissions_collection

# Read XML as binary data
with open('co2_emissions.xml', 'rb') as co2_xml_file:
    co2_xml_data = co2_xml_file.read()

# Insert into MongoDB
co2_document = {'co2_data': co2_xml_data}
co2_collection.insert_one(co2_document)


# In[160]:


co2_document = co2_collection.find_one()
co2_xml_data = co2_document['co2_data']
# Parse and create element tree 
co2_root = ET.fromstring(co2_xml_data)


# In[161]:


co2_record_elements = co2_root.findall('.//record')
# Find first 'record' element or assign none if nothing found
co2_first_record = co2_record_elements[0] if co2_record_elements else None


# Extract unique column names from the first record 'field' element
co2_headings = [field.attrib['name'] for field in co2_first_record.findall('.//field')] if co2_first_record else []

co2_data = []

# For loop to iterate through each 'record' element and extract the data from 'field'
for record in co2_record_elements:
    #Extract text from 'field' and create row of data
    row_data = [field.text for field in record.findall('.//field')]
    co2_data.append(row_data)

co2_df = pd.DataFrame(co2_data, columns=co2_headings)
co2_df


# In[162]:


co2_df.info()


# In[163]:


co2_df.drop(["Item"], axis=1, inplace=True)
co2_df.rename(columns={'Value': 'C02_Emissions_MTPC'}, inplace=True)


# In[164]:


co2_df.dtypes


# In[165]:


# Renaming columns for data consistency
replacement_dict = {
    'Egypt, Arab Rep.': 'Egypt',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Czechia': 'Czech Republic',
    'Iran, Islamic Rep.': 'Iran',
    "Cote d'Ivoire": 'Ivory Coast',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    "Korea, Dem. People's Rep.": 'South Korea',
    'Syrian Arab Republic': 'Syria',
    'Turkiye': 'Turkey',
    'Viet Nam': 'Vietnam',
    'United States': 'USA',
    'Hong Kong SAR, China': 'Hong Kong SAR'
}

co2_df['Country or Area'] = co2_df['Country or Area'].replace(replacement_dict)


# In[166]:


co2_df['Year'] = co2_df['Year'].astype(int)
co2_df['C02_Emissions_MTPC'] = co2_df['C02_Emissions_MTPC'].apply(lambda x: round(pd.to_numeric(x), 4) if not pd.isna(x) else None)


# In[167]:


co2_df = co2_df[(co2_df['Year'] >= 1990) & (co2_df['Year'] <= 2020)]


# In[168]:


co2_df[co2_df.isnull().any(axis=1)]


# In[169]:


co2_df.dropna(inplace=True)


# In[170]:


co2_output = co2_df


# In[171]:


co2_output.reset_index()


# In[172]:


# Am going to use this table for merging with other tables due to table structure
try:
    co2_engine = create_engine('postgresql://dap:dap@192.168.56.30:5432/climate')

    co2_output.to_sql('co2_emissions_output', co2_engine, index=False, if_exists='replace')

    print('DataFrame uploaded to PostgreSQL successfully.')

except Exception as e:
    print('Error:', e)


# In[173]:


co2_df= co2_df.pivot(index='Country or Area', columns='Year', values='C02_Emissions_MTPC')
co2_df= co2_df.reset_index()
co2_df.columns.name = None
co2_df


# In[174]:


print(co2_df[co2_df.isnull().any(axis=1)])


# In[175]:


# Filling with adjacent value due to temporal nature of data
co2_df[1990] = co2_df[1990].combine_first(co2_df[1991])


# In[176]:


# Upload to postgres
try:
    co2_engine = create_engine('postgresql://dap:dap@192.168.56.30:5432/climate')

    co2_df.to_sql('co2_emissions', co2_engine, index=False, if_exists='replace')

    print('DataFrame uploaded to PostgreSQL successfully.')

except Exception as e:
    print('Error:', e)


# In[177]:


######################### Exploring co2 Dataset ####################################
####################################################################################
####################################################################################


# In[178]:


eu_27_list = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia",
    "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

total_co2_eu_df = co2_df[co2_df['Country or Area'].isin(eu_27_list)]
total_co2_eu_df = total_co2_eu_df.reset_index(drop=True)


# In[179]:


# Selecting year columns 
co2_years = total_co2_eu_df.drop(columns=['Country or Area'])
# Descriptive statistics 
co2_years.describe().round(4)


# In[180]:


#Checking for outliers 

co2_Q1 = co2_years.quantile(0.25)
co2_Q3 = co2_years.quantile(0.75)
co2_IQR = co2_Q3 - co2_Q1

co2_lower_bound = co2_Q1 - 1.5 * co2_IQR
co2_upper_bound = co2_Q3 + 1.5 * co2_IQR

# Identify outliers
co2_outliers = (co2_years < co2_lower_bound) | (co2_years > co2_upper_bound)

# Display outliers
co2_outliers_countries = total_co2_eu_df[['Country or Area']][co2_outliers.any(axis=1)]
co2_outliers_countries


# In[181]:


# Mean emission levels for each country
mean_emissions = co2_years.mean(axis=1)

mean_emissions_df = pd.DataFrame({'Country': total_co2_eu_df['Country or Area'], 'Mean_emissions': mean_emissions})

# Sort by mean pollution levels in descending order
highest_emissions = mean_emissions_df.sort_values(by='Mean_emissions', ascending=False)

# Show top 10
highest_emissions.head(10)


# In[182]:


# Sort ascending order 
lowest_emissions = mean_emissions_df.sort_values(by='Mean_emissions')
# Show bottom 10
lowest_emissions.head(10)


# In[183]:


# Kernel density plot for each year
co2_years.plot(kind='kde', figsize=(12, 8))
plt.title('Distribution of co2 emissions Levels (Kernel Density Plot)')
plt.xlabel('co2 emissions Metric Tonnes Per Capita(MPTC)')
plt.show()


# In[184]:


######################### Air Pollution Datasets ###################################
####################################################################################
####################################################################################


# In[185]:


# Connecting to MongoDB
client = pymongo.MongoClient('192.168.56.30', 27017)
air_db = client.climate
air_collection = air_db.air_pollution_collection_1965_2019

# Reading in the xml file as binary 
with open('air_pollution_1965_2019.xml', 'rb') as air_xml_file:
    air_xml_data = air_xml_file.read()

# Inserting into MongoDB
air_document = {'air_pollution_1965_2019': air_xml_data}
air_collection.insert_one(air_document)


# In[186]:


# Retrieving File from collection in MongoDB
air_document = air_collection.find_one()
air_xml_data = air_document['air_pollution_1965_2019']
# Parse and create element tree 
air_root = ET.fromstring(air_xml_data)


# In[187]:


air_record_elements = air_root.findall('.//record')
# Find first 'record' element or assign none if nothing found
air_first_record = air_record_elements[0] if air_record_elements else None

# Extract unique column names from the first record 'field' element
air_headings = [field.attrib['name'] for field in air_first_record.findall('.//field')] if air_first_record else []

air_data = []

# For loop to iterate through each 'record' element and extract the data from 'field'
for record in air_record_elements:
    #Extract text from 'field' and create row of data
    row_data = [field.text for field in record.findall('.//field')]
    air_data.append(row_data)

air_poll_df_1965_2019 = pd.DataFrame(air_data, columns=air_headings)
air_poll_df_1965_2019


# In[188]:


# Cleaning dataframe, dropping column, converting to numeric and rounding numbers
air_poll_df_1965_2019.drop(['Item'], axis=1, inplace=True)
air_poll_df_1965_2019['Value'] = pd.to_numeric(air_poll_df_1965_2019['Value'])
air_poll_df_1965_2019['Value'] = air_poll_df_1965_2019['Value'].apply(lambda x: round(x, 2) if not pd.isna(x) else None)


# In[189]:


air_poll_df_1965_2019.dtypes


# In[190]:


# Converting year to integer and creating new Dataframe on years 2010-2019
air_poll_df_1965_2019['Year'] = air_poll_df_1965_2019['Year'].astype(int)
air_poll_2010_2019 = air_poll_df_1965_2019[(air_poll_df_1965_2019['Year'] >= 2010) & (air_poll_df_1965_2019['Year'] <= 2019)]


# In[191]:


print(air_poll_2010_2019.columns)
# air_poll_2010_2019.reset_index(inplace=True)
# air_poll_2010_2019.drop(['index'], axis=1, inplace=True)
# air_poll_2010_2019


# In[192]:


#Transposing Dataframe
air_poll_10_19 = air_poll_2010_2019.pivot(index='Country or Area', columns='Year', values='Value')
# Resetting index to make 'Country or Area' a regular column 
air_poll_10_19 = air_poll_10_19.reset_index()
# Remove column name for index (was affecting layout)
air_poll_10_19.columns.name = None
air_poll_10_19


# In[193]:


######################### Webscraping for 2020 - 2022 ##############################
####################################################################################
####################################################################################


# In[194]:


url = 'https://en.wikipedia.org/wiki/List_of_countries_by_air_pollution'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
tables = soup.find_all('table', class_='wikitable sortable')
air_pollution_table = tables[0]
root = ET.Element("table_data")


# In[195]:


# Extract all 'th' table header elements 
headings = air_pollution_table.find_all('th')
# Create XML element 'headings' under root element for table headings
headings_element = ET.SubElement(root, "headings")
# For loop to go through each 'th' element to create corresponding 'heading' element under 'headings'
for heading in headings:
    #Extract text from 'th' removing whitespace
    heading_text = heading.get_text(strip=True)
    # Create new 'heading' element under 'headings' and set its text to extracted heading text
    heading_element = ET.SubElement(headings_element, "heading")
    heading_element.text = heading_text


# In[196]:


# Extract 'tr' (table row) elements and skip headings
rows = air_pollution_table.find_all('tr')[1:]  
# Create XML element to store rows
rows_element = ET.SubElement(root, "rows")

# Same as above for heading but for 'tr'
for row in rows:
    row_element = ET.SubElement(rows_element, "row")
    # Extract all 'td' (table cell) and 'th' (table header) from current row
    cells = row.find_all(['td', 'th'])
    for cell in cells:
        # Extract text from cell removing whitespace
        cell_text = cell.get_text(strip=True)
        cell_element = ET.SubElement(row_element, "cell")
        cell_element.text = cell_text


# In[197]:


tree = ET.ElementTree(root)
# Writes the XML data to file, 'wb' writing in binary mode, data written without modification, incase binary data encoded characters (n)
with open('air_pollution.xml', "wb") as xml_f:
    tree.write(xml_f, xml_declaration=True)


# In[198]:


# Connect to MongoDB
client = pymongo.MongoClient('192.168.56.30', 27017)
db = client.climate
collection = db.air_pollution_collection

# Read in xml in binary mode
with open('air_pollution.xml', 'rb') as xml_f:
    # Store binary xml in a dictionary 
    xml_data = xml_f.read()
    document = {'air_pollution': xml_data}
    # Insert into collection
    collection.insert_one(document)


# In[199]:


# Retrieve document from MongoDB
document = collection.find_one()
xml_data = document['air_pollution']

# Parse xml data
root = ET.fromstring(xml_data)
# Extracting the text content of each 'heading' element nested under the 'headings' element (heading element containing multiple headings) Store each text content in list)
headings = [heading.text for heading in root.findall('.//headings/heading')]


# In[200]:


data = []
# Loop through each 'row' element and extract text of 'cell' elements within.
for row in root.findall('.//rows/row'):
      # Store the text content of each 'cell' element in a list representing a row.
    row_data = [cell.text if cell is not None else None for cell in row.findall('.//cell')]
    data.append(row_data)

air_poll_19_22 = pd.DataFrame(data, columns=headings)
air_poll_19_22


# In[201]:


# Renaming columns so they aren't lost during dataframe merge 
air_poll_10_19['Country or Area'] = air_poll_10_19['Country or Area'].replace(replacement_dict)


# In[202]:


# Merge the dataframes together adding 2020 - 2022 onto larger dataframe 
air_poll_df = air_poll_10_19.merge(air_poll_19_22[['Country/Region', '2020', '2021', '2022']], 
                                 left_on='Country or Area', right_on='Country/Region', how='inner')

air_poll_df = air_poll_df.drop(columns='Country/Region')
air_poll_df


# In[203]:


air_poll_df.dtypes


# In[204]:


print(air_poll_df.columns)


# In[205]:


# Rename columns, improved control flow with variables
# Replace '--' values  with NaN
air_poll_df.replace('--', np.nan, inplace=True)

columns_to_rename = {'2020': 2020, '2021': 2021, '2022': 2022}
air_poll_df.rename(columns=columns_to_rename, inplace=True)
year_columns = [2020, 2021, 2022]
# use apply to use function across entire column
air_poll_df[year_columns] = air_poll_df[year_columns].apply(pd.to_numeric)

# Use backward fill for 2020
air_poll_df[2020] = air_poll_df[2020].combine_first(air_poll_df[2019])

# Use forward fill for 2021
air_poll_df[2021] = air_poll_df[2021].combine_first(air_poll_df[2022])


# In[206]:


# Check for null values in DataFrame
print(air_poll_df[air_poll_df.isnull().any(axis=1)])


# In[207]:


# Improved control flow with variable and if statement
null_rows = air_poll_df[air_poll_df.isnull().any(axis=1)]

if not null_rows.empty:
    # Drop rows with null values
    air_poll_df.drop(null_rows.index, inplace=True)
    print("Null values dropped.")
else:
    print("No rows with null values found")


# In[208]:


# Upload to postgres
try:
    engine = create_engine('postgresql://dap:dap@192.168.56.30:5432/climate')

    air_poll_df.to_sql('air_pollution', engine, index=False, if_exists='replace')

    print('DataFrame uploaded to PostgreSQL successfully.')

except Exception as e:
    print('Error:', e)


# In[209]:


air_eu_df = air_poll_df[air_poll_df['Country or Area'].isin(eu_27_list)]
air_eu_df = air_eu_df.reset_index(drop=True)


# In[210]:


######################### Exploring Air pollution ##################################
####################################################################################
####################################################################################


# In[211]:


# Selecting year columns 
air_poll_years = air_eu_df.drop(columns=['Country or Area'])
air_poll_years.describe().round(4)


# In[212]:


#Checking for outliers 

air_Q1 = air_poll_years.quantile(0.25)
air_Q3 = air_poll_years.quantile(0.75)
air_IQR = air_Q3 - air_Q1

air_lower_bound = air_Q1 - 1.5 * air_IQR
air_upper_bound = air_Q3 + 1.5 * air_IQR

# Identify outliers
air_outliers = (air_poll_years < air_lower_bound) | (air_poll_years > air_upper_bound)

# Display outliers
outliers_countries = air_eu_df[['Country or Area']][air_outliers.any(axis=1)]
outliers_countries


# In[213]:


# Mean pollution levels for each country
mean_pollution = air_poll_years.mean(axis=1)

mean_pollution_df = pd.DataFrame({'Country': air_eu_df['Country or Area'], 'Mean_Pollution': mean_pollution})

# Sort by mean pollution levels in descending order
highest_pollution = mean_pollution_df.sort_values(by='Mean_Pollution', ascending=False)

# Show top 10
highest_pollution.head(10)


# In[214]:


# Sort ascending order 
lowest_emissions = mean_pollution_df.sort_values(by='Mean_Pollution')
lowest_emissions.head(10)


# In[215]:


air_poll_years.hist(bins=20, figsize=(12, 8))
plt.suptitle('Distribution of Air Pollution PM2.5 Levels (Histogram)')
plt.show()


# In[216]:


# Kernel density plot for each year
air_poll_years.plot(kind='kde', figsize=(12, 8))
plt.title('Distribution of Air Pollution PM2.5 Levels (Kernel Density Plot)')
plt.xlabel('Air Pollution PM2.5 Levels')
plt.show()


# In[217]:


yearly_average = air_eu_df[list(range(2010, 2023))].mean()


plt.figure(figsize=(10, 6))

sns.lineplot(x=range(2010, 2023), y=yearly_average.values, label='EU Average', marker='o')


plt.xlabel('Year')
plt.ylabel('Average (PM2.5) in micrograms per cubic meter')
plt.title('Average Year by Year')


plt.legend()
plt.show()


# In[218]:


air_arima_df = air_eu_df.melt(id_vars='Country or Area', var_name='Year', value_name='Air_Pollution')
air_arima_df = air_arima_df.groupby('Year').mean().reset_index()


# In[219]:


# Insufficient datapoints for arima model, used SMA instead 

window_size = 3
sma = air_arima_df['Air_Pollution'].rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))
plt.plot(air_arima_df['Year'], air_arima_df['Air_Pollution'], label='Original Data')
plt.plot(air_arima_df['Year'], sma, label=f'SMA ({window_size}-point)', color='red')
plt.title('Simple Moving Average')
plt.xlabel('Year')
plt.ylabel('Air Pollution PM2.5')
plt.legend()
plt.xticks(air_arima_df['Year'])
plt.show()


# In[220]:


# Creating Output table for EU countries
EV_df = pd.read_csv("IEA Global EV Data 2023.csv")
ev_eu_df = EV_df.loc[(EV_df['region'] == 'EU27') & (EV_df['category'] == 'Historical') & (EV_df['parameter'] == 'EV sales') & (EV_df['mode'] == 'Cars') ].groupby(["year"]).agg(EVSales=('value', 'sum'))


# In[221]:


ev_eu_df.reset_index(inplace=True)
ev_eu_df['year'] = ev_eu_df['year'].astype(int)
output_df = pd.merge(air_arima_df, ev_eu_df, left_on='Year', right_on='year', how='left')


# In[222]:


co2_eu_df = co2_df[co2_df['Country or Area'].isin(eu_27_list)]
co2_eu_df = co2_eu_df.reset_index(drop=True)


# In[223]:


co2_eu_df = co2_eu_df.melt(id_vars='Country or Area', var_name='Year', value_name='Total_co2_emissions_MTPC')
co2_eu_df = co2_eu_df.groupby('Year').mean().reset_index()


# In[224]:


output_df = output_df.drop(columns='year', errors='ignore')
co2_eu_df['Year'] = co2_eu_df['Year'].astype(int)
output_df = pd.merge(output_df, co2_eu_df, on='Year', how='left')


# In[225]:


co2_car_df = pd.read_csv('UNFCCC_v26.csv')


# In[226]:


pollutant = co2_car_df['Pollutant_name'] == 'All greenhouse gases - (CO2 equivalent)'
sector = co2_car_df['Sector_name'] == '1.A.3.b.i - Cars'
country = co2_car_df['Country'] == 'EU-27'


co2_car_df = co2_car_df[pollutant & sector & country]


# In[227]:


co2_car_df['Year'] = co2_car_df['Year'].astype(int)
co2_car_df.drop(['Country','Country_code','Sector_name', 'Format_name', 'Pollutant_name', 'Sector_code','Parent_sector_code', 'Unit', 'Notation','PublicationDate', 'DataSource'], axis=1, inplace=True)
co2_car_df = co2_car_df.sort_values(by='Year')
co2_car_df = co2_car_df.reset_index(drop=True)


# In[228]:


output_df = pd.merge(output_df, co2_car_df, on='Year', how='left')


# In[229]:


correlation_columns = output_df[['Air_Pollution', 'EVSales', 'Total_co2_emissions_MTPC', 'emissions']]
correlation_matrix = correlation_columns.corr()
print(correlation_matrix)


# In[230]:


######################################### Creating output table for LM #################################################################


# In[231]:


lm_eu_df = pd.read_csv("IEA Global EV Data 2023.csv")
lm_eu_df = lm_eu_df.loc[(lm_eu_df['region'].isin(eu_27_list)) & (lm_eu_df['category'] == 'Historical') & (lm_eu_df['parameter'] == 'EV sales') & (lm_eu_df['mode'] == 'Cars')]


# In[232]:


lm_eu_df.reset_index(inplace=True)
lm_eu_df['year'] = lm_eu_df['year'].astype(int)


# In[233]:


lm_eu_df.drop(['index', 'category', 'parameter', 'mode', 'powertrain', 'unit'], axis=1, inplace=True)
ev_df_1 = lm_eu_df


# In[234]:


lm_air_df = air_eu_df.melt(id_vars='Country or Area', var_name='Year', value_name='Air_Pollution')


# In[235]:


lm_output_df = pd.merge(lm_air_df, ev_df_1, left_on=['Country or Area', 'Year'], right_on=['region', 'year'], how='inner')


# In[236]:


lm_output_df.drop(['region'], axis=1, inplace=True)


# In[237]:


# Multiple values in value column so aggregated by summing the 'value' column and using 'first' for 'Air_Pollution' for each combination of 'Country or Area' and 'Year'
lm_output_df = lm_output_df.groupby(['Country or Area', 'Year'], as_index=False).agg({
    'Air_Pollution': 'first',  
    'value': 'sum'
})


# In[238]:


lm_co2_eu_df = co2_df['Country or Area'].isin(eu_27_list)
lm_co2_eu_df = co2_df.reset_index(drop=True)


# In[239]:


lm_co2_eu_df = lm_co2_eu_df.melt(id_vars='Country or Area', var_name='Year', value_name='Total_co2_emissions_MTPC')


# In[240]:


lm_co2_eu_df['Year'] = lm_co2_eu_df['Year'].astype(int)
lm_output_df = pd.merge(lm_output_df, lm_co2_eu_df, on= ['Country or Area', 'Year'], how='inner')


# In[241]:


lm_co2_car_df = pd.read_csv('UNFCCC_v26.csv')


# In[242]:


pollutant = lm_co2_car_df['Pollutant_name'] == 'All greenhouse gases - (CO2 equivalent)'
sector = lm_co2_car_df['Sector_name'] == '1.A.3.b.i - Cars'
country = lm_co2_car_df['Country'].isin(eu_27_list)
lm_co2_car_df = lm_co2_car_df[pollutant & sector & country]


# In[243]:


# Problems with strings and integers in year column values like (1987 - 1990)
years_to_keep = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
lm_co2_car_df = lm_co2_car_df[lm_co2_car_df['Year'].isin(years_to_keep)]


# In[244]:


lm_co2_car_df['Year'] = lm_co2_car_df['Year'].astype(int)
lm_co2_car_df.drop(['Country','Country_code','Sector_name', 'Format_name', 'Pollutant_name', 'Sector_code','Parent_sector_code', 'Unit', 'Notation','PublicationDate', 'DataSource'], axis=1, inplace=True)
lm_co2_car_df = lm_co2_car_df.sort_values(by='Year')
lm_co2_car_df = lm_co2_car_df.reset_index(drop=True)


# In[245]:


lm_output_df = pd.merge(lm_output_df,lm_co2_car_df, on='Year', how='inner')


# In[246]:


# Group by 'Country or Area' and 'Year', aggregating 'value' and 'emissions'
lm_output_df = lm_output_df.groupby(['Country or Area', 'Year'], as_index=False).agg({
    'Air_Pollution': 'first',
    'value': 'sum',
    'Total_co2_emissions_MTPC': 'first',
    'emissions': 'sum'
})


# In[247]:


lm_output_df.rename(columns={'value': 'EVSales'}, inplace=True)


# In[248]:


# Change dependent Variable?
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# LM formula
formula = 'EVSales ~ Air_Pollution + Total_co2_emissions_MTPC + emissions'
model = smf.ols(formula=formula, data=lm_output_df)

results = model.fit()

# Perform ANOVA
anova_table = anova_lm(results)
print(anova_table)


# In[249]:


print(results.summary())

