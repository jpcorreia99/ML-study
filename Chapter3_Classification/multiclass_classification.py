from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1) #devove um dict
print(mnist.keys())
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#One versus the rest(OvR), treinar 10 classificadores binários, um para tentar descobrir cada uma das classes vs não ser essa classe
#OvO one versus one, treinar classificador para deterar 0vs1, 0vs2, 1vs 2, etc N*(N-1)/2
#As SVM's apenas conseguem fazer binary classification, mas podemos passar-lhes dados multiclass e ela vai fazer Ovo por trás

from sklearn.svm import SVC #support vector machine classifier
svm_clf = SVC()
'''svm_clf.fit(X_train, y_train)
print("Accuracy: ",svm_clf.score(X_test, y_test))
'''
#se quisesse forçar OvR
from sklearn.multiclass import OneVsRestClassifier
ovr_svc_clf = OneVsRestClassifier(SVC())
'''ovr_svc_clf.fit(X_train, y_train)'''

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
sgd_clf = SGDClassifier(random_state=42) #dá shufle aos dados com seed 42
X_train = StandardScaler().fit_transform(X_train.astype(np.float64))
#sgd_clf.fit(X_train,y_train)
#print(sgd_clf.predict([X_train[0]]))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv = 3)
conf_matrix = confusion_matrix(y_train, y_train_pred)
print(conf_matrix)

row_sums = conf_matrix.sum(axis=1, keepdims=True)
norm_conf_mx = confusion_matrix/row_sums #normaliza a matriz
np.fill_diagonal(norm_conf_mx, 0) # não nos interessa a diagonal para entender os erros
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()