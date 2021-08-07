# Processing of Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from catboost import CatBoostClassifier

# Model building
import tensorflow as tf
from tensorflow import keras

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, Conv1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# reading in the csv data file
raw_data = pd.read_csv("water_potability.csv")
water_q_data = pd.read_csv("water_potability.csv")

# examining the data
print(raw_data.head(-1))

# checking to see how much null data we have
print("Null in B:\n", raw_data.isnull().sum())

# instead of removing NaN rows replace them with the average value
# print(raw_data.mean()
raw_data.fillna(value=raw_data.mean(),inplace=True)
raw_data = raw_data.sample(frac=1,random_state=81)
water_q_data = water_q_data[water_q_data.isna().any(axis=1)]

water_q_data.fillna(value=water_q_data.mean(),inplace=True)

print(raw_data.mean())
# remove features to reduce complexity
# removing features decreased the accuracy of the model
f_drop = ['Organic_carbon','Turbidity']
raw_data.drop(f_drop,axis=1,inplace=True)
water_q_data.drop(f_drop,axis=1,inplace=True)

# print(raw_data.head(-1))

# remove NaN rows
raw_data.dropna(inplace=True)


# resetting index
raw_data.reset_index(inplace=True)

# number of classification
print(raw_data["Potability"].value_counts())


# split the data into X and y
X = raw_data.iloc[:, 1:8]
y = raw_data.iloc[:, 8]


X2 = water_q_data.iloc[:, 0:7]
y2 = water_q_data.iloc[:, 7]
print(X2,y2)

# scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_data2 = scaler.fit_transform(X2)

# convert the array to a dataframe
scaled_data = pd.DataFrame(scaled_data)

scaled_data2 = pd.DataFrame(scaled_data2)

# creation of polynomial features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(scaled_data)
X_poly = pd.DataFrame(X_poly)

X_poly2 = poly.fit_transform(scaled_data2)
X_poly2 = pd.DataFrame(X_poly2)

poly_features_remove =[17, 18, 27]
scaled_data = X_poly.drop(poly_features_remove, axis=1)

scaled_data2 = X_poly2.drop(poly_features_remove, axis=1)

# split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

# using logistic regression to classify the data
model = svm.SVC(C=1.5,kernel='rbf')
#model = CatBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Performance
print("Mean Sq Err: %.2f" % mean_squared_error(y_test, y_pred))
print("accuracy: ", model.score(X_train, y_train))
print("accuracy: ", accuracy_score(y_test,y_pred))

# # convert the dataframes to arrays
# X_train = np.asarray(X_train)
# y_train = np.asarray(y_train)
# X_test = np.asarray(X_test)
# y_test = np.asarray(y_test)
#
# # reshape the arrays to pass into neural network
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
# y_train = y_train.reshape((y_train.shape[0]), 1)
# y_test = y_test.reshape((y_test.shape[0]), 1)
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
#
# # Shape of the Training and Testing Data
# print("X train:", X_train.shape)
# print("Y train:", y_train.shape)
# print("X test:", X_test.shape)
# print("Y test:", y_test.shape)
#
# # creating model
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, input_shape=(9, 1, 1),
#                  activation='relu'))
# # model.add(MaxPooling2D(pool_size=2))
# # model.add(Dropout(0.2))
# # #
# # model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
# # model.add(MaxPooling2D(pool_size=2))
# #
# # model.add(GlobalAveragePooling2D())
# model.add(Dense(2, activation='softmax'))
#
# model.summary()
#
# # compile model
# adam = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# # fit the model
# print(X_train)
# model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=2)

# manual testing
# n=373
# print(y2)
#
# data = scaled_data2.iloc[n:n+1,:]
# print(data)
# print(water_q_data.iloc[n:n+1,:])
# print("Prediction:",model.predict(data), "Actuality", y2.iloc[n])

y_nan_pred = model.predict(scaled_data2)
print("NaN accuracy: ", accuracy_score(y2,y_nan_pred))