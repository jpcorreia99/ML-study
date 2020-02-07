from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X,y = make_regression(100,1, noise=10, bias=3.6)
plt.scatter(X,y)
plt.show()

linear_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", LinearSVR(epsilon=1.5)) #epsilon- tolerância das margens
])

linear_svr.fit(X,y)
print(linear_svr.score(X,y))
from sklearn.metrics import mean_squared_error
predictions = linear_svr.predict(X)
print("MSE: ", mean_squared_error(y, predictions))

#Non linear-data
import  numpy as np
import random

def square(x):
    y=[]
    for elem in x:
        noise = random.uniform(-2,2)
        y.append(elem*elem+noise)
    y = np.array(y)
    return y


x = np.arange(-5,5, step= 0.1).reshape(-1,1)
y = square(x)
plt.scatter(x,y)
plt.show()
print(x.shape, y.shape)

from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(x,y)
print(mean_squared_error(y, svm_poly_reg.predict(x)))
print(svm_poly_reg.score(x,y))