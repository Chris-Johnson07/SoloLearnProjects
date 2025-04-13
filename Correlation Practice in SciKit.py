from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)
housing_df['MEDV'] = housing.target

column_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'MEDV']

corr_matrix = housing_df[column_names].corr().round(2)
print(corr_matrix)

##################

model = LinearRegression()
print(model)
X = housing_df['AveRooms']
X = pd.DataFrame(X)
Y = housing_df['MEDV']
Y = pd.DataFrame(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
model.fit(X_train, Y_train)

print(model.intercept_.round(2))
print(model.coef_.round(2))

new_RM = np.array([6.5]).reshape(-1,1)
print(model.predict(new_RM))