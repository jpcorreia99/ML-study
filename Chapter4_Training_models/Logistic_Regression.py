from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(list(iris.keys()))


X = iris['data'][:, 3:] #petal Width
y = (iris['target'] == 2).astype(np.int) #1 if iris virginica, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0,3,1000). reshape(-1,1) #lista com 1000 valores entre 0 e 3, sempre crescendo. são depois reordenados para uma lista de listas, em que cada target é uma lista
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label = "Iris Indica") #green
plt.plot(X_new, y_proba[:,0], "b--", label = "Not Iris Indica") #blue
plt.xlabel("Petal_width(cm)")
plt.ylabel("Probability")
plt.show()

#Agora treinar um classificador multiclasse
X = iris["data"][:, (2,3)] #petal length, petal width
y = iris["target"]

#Iria fazer treino OvR (One versus Rest), mas como indicamos "multinomial" vai usar a funçã softmax de decisão
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C = 10)
softmax_reg.fit(X,y)

print(softmax_reg.predict([[5,2]]))