import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error, accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# reading in the csv data file
raw_data = pd.read_csv("water_potability.csv")
water_q_data = pd.read_csv("water_potability.csv")

# removing features decreased the accuracy of the model
f_drop = ['Organic_carbon','Turbidity']
raw_data.drop(f_drop,axis=1,inplace=True)

# remove NaN rows
raw_data.dropna(inplace=True)

raw_data = raw_data.sample(frac=1)

# resetting index
raw_data.reset_index(inplace=True)

test_data = raw_data

# size of dataset
size = [20, 100, 500, 1000, 1500, 2000,2010]

mean_sq_err =[]
mod_train_acc =[]
mod_test_acc =[]

for i in range(len(size)):
    raw_data = test_data.iloc[0:size[i]]

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

    poly_features_remove = [17, 18, 27]
    scaled_data = X_poly.drop(poly_features_remove, axis=1)

    # split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

    # using logistic regression to classify the data
    model = svm.SVC(C=1.5, kernel='rbf')
    # model = CatBoostClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mean_sq_err.append(mean_squared_error(y_test, y_pred))
    mod_train_acc.append(model.score(X_train, y_train))
    mod_test_acc.append(accuracy_score(y_test, y_pred))
    print(i, " ",size[i]," ", accuracy_score(y_test, y_pred) )
print(y_pred)
# plot graphs

# plt.figure(1)
# plt.plot(size,mean_sq_err)
# plt.title('Mean Sq Err')
#
# plt.figure(2)
# plt.plot(size,mod_train_acc)
# plt.title('Model Training Acc')
#
# plt.figure(3)
# plt.plot(size,mod_test_acc)
# plt.title('Model Test Acc')
#
# plt.show()
