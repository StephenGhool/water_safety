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
# _____________________________________________________________________________________________
# RESULTS
# 0   ph : 0.6109550561797753
# 1   Chloramines : 0.6688741721854304
# 2   Hardness : 0.6854304635761589
# 3   Solids : 0.6672185430463576
# 4   Sulfate : 0.6289308176100629
# 5   Organic_carbon : 0.7185430463576159
# 6   Trihalomethanes : 0.694488188976378
# 7   Conductivity : 0.6920529801324503
# 8   Turbidity : 0.7152317880794702
# reading in the csv data file
# _______________________________________________________________________________________________
test_data = pd.read_csv("water_potability.csv")
water_q_data = pd.read_csv("water_potability.csv")

# remove features to reduce complexity
# removing features decreased the accuracy of the model
f_drop = ['ph', 'Chloramines', 'Hardness', 'Solids', 'Sulfate', 'Organic_carbon', 'Trihalomethanes', 'Conductivity',
          'Turbidity']
for i in range(len(f_drop)):
    raw_data = test_data.drop(f_drop[i], axis=1)

    # remove NaN rows
    raw_data.dropna(inplace=True)

    # resetting index
    raw_data.reset_index(inplace=True)

    # split the data into X and y
    X = raw_data.iloc[:, 1:9]
    y = raw_data.iloc[:, 9]

    # scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # convert the array to a dataframe
    scaled_data = pd.DataFrame(scaled_data)

    # creation of polynomial features
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(scaled_data)
    scaled_data = X_poly

    # split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

    # using logistic regression to classify the data
    model = svm.SVC()
    # model = CatBoostClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    # print("Mean Sq Err: %.2f" % mean_squared_error(y_test, y_pred))
    print(i, " ", f_drop[i], ":", accuracy_score(y_test, y_pred))
