import numpy as np
import scipy.stats as stats

housing_prepared = np.load("housing_prepared.npy")
housing_labels = np.load("housing_labels.npy")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

forest_reg = RandomForestRegressor()
#lista de parametros do RandomForesClassifier
#2 dicionários, cada um com combinações de parametros, a grid search vai testar
# 3*4=12 + 2*3 =6 = 18 possiveis combinações de parametros
param_grid = [
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]

grid_search = GridSearchCV(forest_reg, param_grid,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#random grid search
from sklearn.model_selection import RandomizedSearchCV
param_dist = [{'n_estimators': range(1,100), 'max_features': range(1,8)}]

random_search = RandomizedSearchCV(forest_reg, param_dist, n_iter=30,
                                   scoring="neg_mean_squared_error",
                                   return_train_score=True)

'''random_search.fit(housing_prepared, housing_labels)
print(random_search.best_params_)'''


#ENSEMBLE METHODS
feature_importances = grid_search.best_estimator_.feature_importances_
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', "Inland"]
print(sorted(zip(feature_importances, features), reverse=True))
#disto vemos que só importa se for INLAND (depois alterei para só ter inland/n inland)

#Evaluating on the test set
final_model = grid_search.best_estimator_
housing_test_prepared = np.load("housing_test_prepared.npy")
housing_test_labels = np.load("housing_test_labels.npy")

print(housing_test_prepared[:5])
print(housing_test_labels[:5])

final_predictions = final_model.predict(housing_test_prepared)
from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(final_predictions, housing_test_labels)
print("Error: ",np.sqrt(final_mse))

#generalizar o erro ao intervalo de confiança de 95% para saber melhor por onde andará o erro

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions-housing_test_labels)**2
interval = stats.t.interval(confidence , len(squared_errors)-1,
                             loc = squared_errors.mean(),
                             scale = stats.sem(squared_errors)) #sem- standard error of the mean
print(np.sqrt(interval))

import joblib
joblib.dump(final_model,"final_model.pickle")
#later model = joblid.load("final_model.pickle")
