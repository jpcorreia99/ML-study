from sklearn.ensemble import  BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X,y = make_moons(n_samples=500, noise=0.15)
plt.scatter(X[:,0], X[:,1],c=y, cmap="jet")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#500 arvores, com 100 samples, paraleliza o máximo que puder
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=0.6, bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train,y_train)
print(bag_clf.score(X_test, y_test))

#Out of bag evaluation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(X,y)
print(bag_clf.oob_score_)

