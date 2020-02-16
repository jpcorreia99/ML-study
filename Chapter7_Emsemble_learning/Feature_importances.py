from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()

X = iris['data']
y = iris['target']

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X,y)
for name,score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)


from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(mnist["data"], mnist["target"])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot,
               interpolation="nearest")
    plt.axis("off")



plot_digit(rnd_clf.feature_importances_)
plt.show()

