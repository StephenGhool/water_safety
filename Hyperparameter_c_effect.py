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

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# best value for C is 1.5
# reading in the csv data file
raw_data = pd.read_csv("water_potability.csv")
water_q_data = pd.read_csv("water_potability.csv")

# examining the data
print(raw_data.head(-1))

# checking to see how much null data we have
print("Null in B:\n", raw_data.isnull().sum())

f_drop = ['Organic_carbon', 'Turbidity']
raw_data.drop(f_drop, axis=1, inplace=True)

# remove NaN rows
raw_data.dropna(inplace=True)

# resetting index
raw_data.reset_index(inplace=True)

# split the data into X and y
X = raw_data.iloc[:, 1:8]
y = raw_data.iloc[:, 8]

# scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# convert the array to a dataframe
scaled_data = pd.DataFrame(scaled_data)

# creation of polynomial features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(scaled_data)
X_poly = pd.DataFrame(X_poly)
scaled_data = X_poly

poly_features_remove =[17, 18, 27]
scaled_data = X_poly.drop(poly_features_remove, axis=1)

# split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

#c_values = [0.001,0.1,1,1.5,5,10,100]
c_values = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]

for i in range(0, len(c_values)):
    # using logistic regression to classify the data
    model = svm.SVC(C=c_values[i])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    print("C = ",c_values[i]," ",accuracy_score(y_test, y_pred))

