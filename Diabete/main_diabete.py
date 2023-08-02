# %%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

parent_directory = os.path.dirname(current_dir)
sys.path.append(parent_directory + "\ML_classes")

import ML_classes as MLc

# %%
# Loading dataset
datasheet = pd.read_csv("diabetes_prediction_dataset.csv")
dataframe = pd.DataFrame(datasheet)

# Remove the 18 cases of "other" genders, to simplify the learning process
dataframe = dataframe.drop(dataframe[dataframe["gender"] == "Other"].index)
# If the gender happens to be an important factor, then we'll run additional studies for these kind of cases

# Removing the occurences of lacking smoking data 
dataframe = dataframe.drop(dataframe[dataframe["smoking_history"] == "No Info"].index)

data_colomns = dataframe.columns

data_train = dataframe.values[0::2,:-1] 
data_test = dataframe.values[1::2,:-1] 
data_train_labels = dataframe[["diabetes"]].values[0::2] 
data_test_labels = dataframe[["diabetes"]].values[1::2]

array = MLc.Dataset(data_train, data_train_labels, data_colomns)
array.returnMetadata()

# %% Data edition
# Original dataset
fig, ax = plt.subplots()
ax.table(cellText=dataframe.head(20).values, loc='center')
# plt.show()

# Changing "gender" to int: 0 "Male", 1 "Female"
array.replaceData("Male", 0)
array.replaceData("Female", 1)

# Changing "smoking_history"  
array.replaceData("never", 0)
array.replaceData("ever", 1)
array.replaceData("current", 1)
array.replaceData("former", 0.5)
array.replaceData("not current", 1)

# Normalizing data
array.normalizeData()

# Final dataset
fig, ax = plt.subplots()
ax.table(cellText=array.returnData()[0][0:20], loc='center')
# plt.show()

# %%
import tensorflow as tf

# Model creation
model = tf.keras.models.Sequential()

loss = "mse" # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

input_layer =           tf.keras.layers.Dense(128, input_dim=8, activation='relu', name="input_layer")
processing_layer_1 =    tf.keras.layers.Dense(64, activation='relu', name="processing_layer_1")
processing_layer_2 =    tf.keras.layers.Dense(16, activation='relu', name="processing_layer_2")
processing_layer_3 =    tf.keras.layers.Dense(2, activation='relu', name="processing_layer_3")
output_layer =          tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")

layers = [input_layer, processing_layer_1, processing_layer_2, processing_layer_3, output_layer]
for layer in layers:
    model.add(layer)

model.build()
model.compile(loss=loss,  optimizer=optimizer, metrics=metrics)
print(model.summary())

# Data conversion to tensor objects
tf_data_train, tf_data_test = array.returnData()[0][0::2], array.returnData()[0][1::2]
tf_data_train_labels, tf_data_test_labels = array.returnData()[1][0::2], array.returnData()[1][1::2]
tf_data_train = tf.convert_to_tensor(tf_data_train, dtype=tf.float32)
tf_data_test = tf.convert_to_tensor(tf_data_test, dtype=tf.float32)

# Model computation
model.fit(tf_data_train, tf_data_train_labels, epochs=10, verbose=2)

# %% Model Evaluation
prediction = np.round(model.predict(tf_data_test))
error_rate = np.sum(prediction[prediction != tf_data_test_labels]) / len(data_test_labels)
print("Diabete detection accuracy on test data =", round(1 - error_rate, 4)*100, "%")

# %%
plt.close('all')
plt.show()

