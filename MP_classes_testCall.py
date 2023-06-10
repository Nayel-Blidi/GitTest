
import MP_classes as MPc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\Git\GitTest\iris.csv")
dataarray =dataset.values
datalabels = dataset.columns.values

array = MPc.Dataset(dataarray, datalabels)
print(datalabels)
array.visualizeData2D("sepal_length", "sepal_width", plot_type="scatter")
array.visualizeData2D("petal_length", "petal_width", plot_type="scatter")

array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="scatter")
array.visualizeData3D("petal_length", "petal_width", "sepal_length", plot_type="default")
