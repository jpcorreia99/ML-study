import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

housing_prepared = np.load("housing_prepared.npy")
housing_labels = np.load("housing_labels.npy")
print(housing_prepared)
print(housing_labels)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing_prepared[:5]
some_labels = housing_labels[:5]
predictions = lin_reg.predict(some_data)

print(predictions)
print(some_labels)

from sklearn.metrics import mean_squared_error
error= mean_squared_error(predictions, some_labels)
print(np.sqrt(error))
'''strat_test_set = pd.read_pickle("strat_test_set.pickle")
print(strat_test_set.head())'''

from sklearn.tree import DecisionTreeRegressor

tree_reg =DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

scores = np.sqrt(-scores)
print("Scores: ", scores)
print("Mean: ", scores.mean())

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

scores = np.sqrt(-scores)
print("Scores: ", scores)
print("Mean: ", scores.mean())