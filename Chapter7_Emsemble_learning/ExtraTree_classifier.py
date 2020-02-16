import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

X,y = make_moons(n_samples=500, noise=0.15)
X_train, X_test, Y_train, Y_test = train_test_split(X,y)
plt.scatter(X[:,0],X[:,1], c=y, cmap='jet')
plt.show()

rnd_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, oob_score=True, bootstrap=True, warm_start=True)
rnd_clf.fit(X_train,Y_train)
print(rnd_clf.score(X_test,Y_test))