import pandas as pd
import numpy as np

print('Start of file1')
data = {
"calories": [420, 380, 390],
"duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)
print('End of file1')