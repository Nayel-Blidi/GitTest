
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import os 
import requests


# %% 
"""
Chinese reading program project in three steps :
- Learning handwritten characters (dataset NN)
- Identifying characters on a picture (? to be thought)
- Translating the visualized text (dictionnary dataset to be found (pleco ?))

Optional :
- Voice synthesis of the translated text
- API (html ?)

Test run with chinese numbers, lighter
"""

# %% Utilities

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

# %%

#raw_data = pd.read_csv("D:\Machine Learning\Chinese handwritten characters dataset")

url = "https://translate.google.com/?hl=fr"
url = "https://www.google.com/"

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
#     "Accept-Encoding": "*",
#     "Connection": "keep-alive"
# }

# headers = {
#     'Content-Type': 'application/json',
#     'accept': 'application/json',
# }

data = {
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3'
}

# response = requests.get(url=url)
response = requests.post(url, data)

print(0)
# %%
