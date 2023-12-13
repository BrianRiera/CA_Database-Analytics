This code base uses 5 different datasets of XML and CSV formats, loads them in MongoDB and PostgreSQL Databases and then process them to analyze Electric Vehicles correlation on Greenhouse Gas emissions, CO2 and Air Pollution.

Main.py is the main to file to be run.

Prior to executing Main.py here are prerequisites

**Install the following packages**
pandas, matplotlib, psycopg2, csv, numpy, statsmodels, json, sklearn, math, bs4, requests, xml, pymongo, sqlalchemy, seaborn

**Databases Connection details**
Update constants.py with MongoDb and PostgreSQL DB details

**Dataset File names and paths**
Update constants.py with csv, xml file names and paths if they are not in root folder 
