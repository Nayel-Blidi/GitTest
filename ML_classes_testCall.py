# %%
import ML_classes as MLc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensuring the working directory is correct
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Loading example dataset
dataset = pd.read_csv("iris.csv")
dataarray = dataset.values
print(dataset[["species"]])

# Separation of the datavalues and labels
data = dataarray[:,:-1]
labels = dataarray[:, -1]

# Names of the colomns (keys)
datacolomns = dataset.columns.values.tolist()

array = MLc.Dataset(data, labels, datacolomns)
metadata = array.returnMetadata()
array.returnData()
print(metadata)

# Data plots
array.visualizeData2D("petal_length", "petal_width", plot_type="scatter")
array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")

# Standard gradient descent computation
theta = array.descentGradient()

# Gradient evaluation
test_prediction = array.autoTestGradient()
test_prediction, fig1 = array.visualizeGradient2D(data, datacolomns[0], datacolomns[1])
test_prediction, fig1 = array.visualizeGradient3D(data, datacolomns[0], datacolomns[1], datacolomns[2])

# 1vsAll gradient classification
all_theta = array.oneVsAll()
test_prediction = array.predictOneVsAll(data, labels)

array.descentGradient()
test_prediction = array.visualizeGradient2D(data, datacolomns[2], datacolomns[3])
test_prediction = array.visualizeGradient3D(data)


# array.visualizeData2D()
# array.visualizeData2D("petal_length", "petal_width", plot_type="scatter")


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

# plt.show()
plt.close('all')

