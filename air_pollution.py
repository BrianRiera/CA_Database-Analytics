from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import pymongo
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_countries_by_air_pollution'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

tables = soup.find_all('table', class_='wikitable sortable')
air_pollution_table = tables[0]
root = ET.Element("table_data")

headings = air_pollution_table.find_all('th')
headings_element = ET.SubElement(root, "headings")
for heading in headings:
    heading_text = heading.get_text(strip=True)
    heading_element = ET.SubElement(headings_element, "heading")
    heading_element.text = heading_text

rows = air_pollution_table.find_all('tr')[1:]  # Skip headings
rows_element = ET.SubElement(root, "rows")

for row in rows:
    row_element = ET.SubElement(rows_element, "row")
    cells = row.find_all(['td', 'th'])
    for cell in cells:
        cell_text = cell.get_text(strip=True)
        cell_element = ET.SubElement(row_element, "cell")
        cell_element.text = cell_text

tree = ET.ElementTree(root)

with open('air_pollution.xml', "wb") as xml_f:
    tree.write(xml_f, xml_declaration=True)

client = pymongo.MongoClient('192.168.56.30', 27017)
db = client.climate
collection = db.air_pollution_collection

with open('air_pollution.xml', 'rb') as xml_f:
    xml_data = xml_f.read()
    document = {'air_pollution': xml_data}
    collection.insert_one(document)

document = collection.find_one()

xml_data = document['air_pollution']

root = ET.fromstring(xml_data)

headings = [heading.text for heading in root.findall('.//headings/heading')]

data_list = []
for row in root.findall('.//rows/row'):
    row_data = [cell.text for cell in row.findall('.//cell')]
    data_list.append(dict(zip(headings, row_data)))

df = pd.DataFrame(data_list)
df

#Average air pollution by year