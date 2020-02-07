import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

mnist = load_digits()
X = mnist["images"]
y = mnist["target"]

n_samples, nx,ny = X.shape
X =X.reshape(n_samples,nx*ny)

print(X.shape)
plt.imshow(X[0].reshape(8,8), cmap="gray")
plt.show()

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),n_estimators=500,
    bootstrap=True, max_samples=0.6,
    bootstrap_features=True, max_features=0.6,
    oob_score=True
)

bag_clf.fit(X,y)
print(bag_clf.oob_score_)