import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(500, 1) - 0.5
Y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(500)

X_train, X_test, Y_train, Y_test = X[:400], X[400:], Y[:400], Y[400:]
plt.scatter(X, Y)
plt.show()

gb_rg = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.1)
gb_rg.fit(X_train, Y_train)
print(gb_rg.score(X_test, Y_test))

# finding the optimal number of predictors
# staged_predict()- returns an iterator over the accuraccy of the predictor with one, two,... trees

errors = [mean_squared_error(Y_test, Y_pred)
          for Y_pred in gb_rg.staged_predict(X_test)]

best_number_of_estimators = np.argmin(errors) + 1  # because it will return an index
print(best_number_of_estimators, "estimators")

gb_rg_optimized = GradientBoostingRegressor(max_depth=2, n_estimators=best_number_of_estimators, learning_rate=0.1)
gb_rg_optimized.fit(X_train, Y_train)
print(gb_rg_optimized.score(X_test, Y_test))

# early stopping- stops if we don't improve for 5 straight epochs
# warm_start = True, sklearn keeps existing trees when fit() is called again, reducing trainig time
# because it doesn't need to train it all again.

gb_rg_es = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float("inf")
error_going_up = 0
best_number_of_estimators = 0
for i in range(1, 400):
    gb_rg_es.n_estimators = i
    gb_rg_es.fit(X_train, Y_train)
    error = mean_squared_error(Y_test, gb_rg_es.predict(X_test))
    if error < min_val_error:
        min_val_error = error
        error_going_up = 0
        best_number_of_estimators = i
    else:
        error_going_up += 1
        if error_going_up == 5:
            break
print("Best number of estimators(early_stopping)", best_number_of_estimators)
gb_rg_es.warm_start=False
gb_rg_es.n_estimators = best_number_of_estimators
gb_rg_es.fit(X_train, Y_train)
print("Score: ", gb_rg_es.score(X_test, Y_test))
