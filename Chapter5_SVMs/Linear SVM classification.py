import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data'][:, (2,3)] #petal length, petal width
y = (iris['target']==2).astype(np.float64) # if it is iris virginica

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")) #quanto mais baixo o C, mais outliers ele permite dentro das margens
])

svm_clf.fit(X,y)
print((svm_clf.score(X,y)))
