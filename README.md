This code base uses 5 different datasets of XML and CSV formats, loads them in MongoDB and PostgreSQL Databases and then process them to analyze Electric Vehicles correlation on Greenhouse Gas emissions, CO2 and Air Pollution.

Main.py is the main to file to be run.

Prior to executing Main.py here are prerequisites

**Install the following packages**
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import csv
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import pymongo
from sqlalchemy import create_engine
import seaborn as sns

**Databases Connection details**
Update constants.py with MongoDb and PostgreSQL DB details

**Dataset Files names and paths**
Update constants.py with csv, xml file names and paths if they are not in root folder 
