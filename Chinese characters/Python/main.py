
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import os 
import requests

import imageHandler


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

# %%
if __name__ == "__main__":
    imageClass = imageHandler()
    