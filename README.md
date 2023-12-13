This code base uses 5 different datasets of XML and CSV formats, loads them into MongoDB and PostgreSQL Databases, and then processes them to analyze Electric Vehicles' correlation with Greenhouse Gas emissions, CO2, and Air Pollution.

Main.py is the main file to be run.

Before executing Main.py here are the prerequisites

**Install the following packages**
pandas, matplotlib, psycopg2, csv, numpy, statsmodels, json, sklearn, math, bs4, requests, xml, pymongo, sqlalchemy, seaborn

**Databases Connection details**
Update constants.py with MongoDb and PostgreSQL DB details

**Dataset File names and paths**
Update constants.py with csv, xml file names, and paths if they are not in the root folder 



**If you choose to run the files separately the order is listed as:**
**(1)** Constants
**(2)** ev_sales
**(3)** eu_ghg_emissions
**(4)** co2_emissions (combined with air_pollution as Code_artefact)
**(5)** air_pollution (combined with co2_emissions as Code_artefact)
**(6)** TimeSeriesPlots
