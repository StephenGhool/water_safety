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

# From the following we could observe that removing features 17,18 and 27 would slighly increase the performance of the sys
# feature 0   0.7168874172185431
# feature 1   0.7152317880794702
# feature 2   0.7152317880794702
# feature 3   0.7152317880794702
# feature 4   0.7135761589403974
# feature 5   0.7152317880794702
# feature 6   0.7185430463576159
# feature 7   0.7168874172185431
# feature 8   0.7102649006622517
# feature 9   0.7135761589403974
# feature 10   0.7185430463576159
# feature 11   0.7152317880794702
# feature 12   0.7152317880794702
# feature 13   0.7135761589403974
# feature 14   0.7168874172185431
# feature 15   0.7135761589403974
# feature 16   0.7152317880794702
# feature 17   0.7235099337748344
# feature 18   0.7201986754966887
# feature 19   0.7152317880794702
# feature 20   0.7185430463576159
# feature 21   0.7185430463576159
# feature 22   0.7119205298013245
# feature 23   0.7152317880794702
# feature 24   0.7152317880794702
# feature 25   0.7218543046357616
# feature 26   0.7168874172185431
# feature 27   0.7201986754966887
# feature 28   0.7135761589403974
# feature 29   0.7152317880794702
# feature 30   0.7152317880794702
# feature 31   0.7119205298013245
# feature 32   0.7168874172185431
# feature 33   0.7168874172185431
# feature 34   0.7168874172185431
# feature 35   0.7135761589403974

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

for i in range(0, len(scaled_data.columns)):
    scaled_data = X_poly.drop(i, axis=1)

    # split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

    # using logistic regression to classify the data
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    print("feature",i," ",accuracy_score(y_test, y_pred))

