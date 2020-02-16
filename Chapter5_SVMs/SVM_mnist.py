from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

mnist = load_digits()
x = mnist["images"]
y = mnist["target"]

plt.imshow(x[0], cmap="gray")
plt.show()

#imagens 8x8
print(x.shape)
nsamples, nx, ny = x.shape
x = x.reshape((nsamples,nx*ny))
print(x.shape)
print(x[0])

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


train_data, test_data, train_labels , test_labels = train_test_split(x,y,test_size=0.1)
svc_clf = Pipeline([
    ("standardizer", StandardScaler()),
    #("svm_clf", SVC(kernel="rbf", C= .001, gamma=100))
    ("svm_clf", SVC(kernel="poly", degree=3))
])

svc_clf.fit(train_data, train_labels)
print("Score: ",svc_clf.score(test_data, test_labels))