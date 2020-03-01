import numpy as np
import matplotlib.pyplot as plt
import xgboost

np.random.seed(42)
X = np.random.rand(500, 1) - 0.5
Y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(500)

X_train, X_test, Y_train, Y_test = X[:400], X[400:], Y[:400], Y[400:]
plt.scatter(X, Y)
plt.show()

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, Y_train)
print(xgb_reg.score(X_test, Y_test))

#can handle early stopping by itself

xgb_reg = xgboost.XGBRegressor(n_estimators=200)
xgb_reg.fit(X_train, Y_train,
    eval_set=[(X_test,Y_test)],early_stopping_rounds=5)
print(xgb_reg.score(X_test, Y_test))
