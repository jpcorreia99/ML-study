from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

X, Y = make_moons(n_samples=5000, noise=0.25)
plt.scatter(X[:, 0], X[:, 1], cmap='jet', c=Y)
plt.show()
X_train, Y_train, X_test, Y_test = X[:4000], Y[:4000], X[:4000], Y[:4000]

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # the one that comes default7
    n_estimators=200,
    algorithm="SAMME.R",  # algorithm that can only be used if the base estimator outputs class_probabilities
    learning_rate=0.5)

ada_clf.fit(X_train,Y_train)
print(ada_clf.score(X_test, Y_test))

