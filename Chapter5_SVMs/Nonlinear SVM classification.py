from sklearn.datasets import make_moons
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm  import LinearSVC
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=1000, noise=0.15) # cria um dataset em forma de duas luas (ver pág 158)

plt.scatter(X[:,0], X[:, 1],c=y, cmap="jet")
plt.show()

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)), #um polinomial feature de nivel 2 adiciona ao dataset x, um feature que é o valor x² (ver página 157)
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(X,y)

from sklearn.metrics import confusion_matrix, accuracy_score
predictions = polynomial_svm_clf.predict(X)
print(confusion_matrix(predictions, y))

#Polinomial kernel
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X,y)
print(poly_kernel_svm_clf.score(X,y))

#RBF kernel- similarity features
rbf_kernel_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C = 1000))
])

rbf_kernel_clf.fit(X[:900],y[:900])
predictions = rbf_kernel_clf.predict(X[900:])
print(confusion_matrix(y[900:], predictions))
print(confusion_matrix(y[900:], predictions))
print("score: ", accuracy_score(y[900:], predictions))


plt.scatter(X[:,1], y)
plt.show()