# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

parent_directory = os.path.dirname(current_dir)
sys.path.append(parent_directory + "\ML_classes")

import ML_classes as MLc

# %%

listing_df = pd.DataFrame(pd.read_csv("listings.csv"))
listing_columns = listing_df.columns.values

listing_df.head()

neighbourhoods_df = pd.DataFrame(pd.read_csv("neighbourhoods.csv"))

reviews_df = pd.DataFrame(pd.read_csv("reviews.csv"))
reviews_columns = reviews_df.columns.values

print(listing_columns, reviews_columns)






# %%
