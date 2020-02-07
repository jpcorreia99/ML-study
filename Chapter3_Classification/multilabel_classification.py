from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1) #devove um dict
print(mnist.keys())
X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_large = (y_train >= 7) # fica 1 se o numero for 7 ou maior e 0 se o resto
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
print(y_multilabel[0])

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
#knn_clf.fit(X_train, y_multilabel)
#print(knn_clf.predict([X_train[0]])) # o 5 não é >=7 e é impar

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
y_train_knn_cross_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv = 3)
print("Knn f1 score: ", f1_score(y_multilabel, y_train_knn_cross_pred, average="macro")) #macro- assume que todas as labels são igualmente importantes , caso contrário usar weighted

