# %%

import ML_classes as MPc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

dataset = pd.read_csv("C:\Git\GitTest\iris.csv")
dataarray = dataset.values

data = dataarray[:,:-1]
labels = dataarray[:, -1]
datacolomns = dataset.columns.values.tolist()

array = MPc.Dataset(data, labels, datacolomns)
metadata = array.returnMetadata()

theta = array.descentGradient()

test_prediction = array.autoTestGradient()
test_prediction, fig1 = array.visualizeGradient(data, datacolomns[0], datacolomns[1])
test_prediction, fig2 = array.visualizeGradient(data, datacolomns[1], datacolomns[2])


# test_prediction = array.visualizeGradient(data, datacolomns[2], datacolomns[3])

# %%

# array.visualizeData2D()
# array.visualizeData2D("petal_length", "petal_width", plot_type="scatter")

# array.visualizeData3D()
# array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")

# array.visualizeData2D("sepal_length", "sepal_width", plot_type="scatter")
# array.normalizeData()
# array.visualizeData2D("sepal_length", "sepal_width", plot_type="scatter")
# array.denormalizeData()
# array.visualizeData2D("sepal_length", "sepal_width", plot_type="scatter")
# plt.close('all')

# array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")
# array.normalizeData()
# array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")
# array.denormalizeData()
# array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")

plt.show()

