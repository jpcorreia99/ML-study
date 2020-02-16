from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1) #devove um dict
print(mnist.keys())
X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)
some_digit = X[0]
plt.imshow(some_digit.reshape(28,28), cmap="binary")
plt.show()

y = y.astype(np.uint8) #a label tinha os números em strings

#este mnist já vem ordenado de maneira a o test set ser representativo

X_train, X_test, y_train, y_test = X[:60000], X[:60000], y[:60000], y[:60000]

#simplificar para um exemplo onde só tem de descobrir se é 5 ou não
y_train_5 = (y_train == 5) #True para os 5's, false para o resto
y_test_5 = (y_test == 5)
print(y_train_5.shape)


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42) #dá shufle aos dados com seed 42
#o SGDClassifier é um classifier qualquer que treina com stochastic gradient descent
#neste caso é um classificador que calcula uma função de decisão e os digitos que estiverem acima de um threshold são classificados como 5
sgd_clf.fit(X_train,y_train_5)
print(sgd_clf.predict([X[0]]))

#Cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def cross_validation(X_train, X_test, y_train_5, y_test_5):
    skfolds = StratifiedKFold(n_splits=3)
    for train_indexes, test_indexes in skfolds.split(X_train, y_train_5): #Nota, o train e o test index s
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_indexes]
        y_train_5_folds = y_train_5[train_indexes]
        X_test_folds = X_test[test_indexes]
        y_test_5_folds = y_test_5[test_indexes]

        clone_clf.fit(X_train_folds, y_train_5_folds)
        print(clone_clf.score(X_test_folds,y_test_5_folds))

#cross_validation(X_train,X_test,y_train_5, y_test_5)

from sklearn.model_selection import cross_val_score
#print(cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy'))

#confusion matrix

from sklearn.model_selection import cross_val_predict #em vez de retornar os scores, retorna as previsões

y_train_pred = cross_val_predict(sgd_clf, X_train,y_train_5, cv = 5)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_train_5, y_train_pred)
print(matrix)

#precision and recall
# precision = TP/(TP+FP),  pag 91/92
#recall or sensitivity= TP/(TP+FN)

from sklearn.metrics import precision_score, recall_score
print("Precisiopn score: ",precision_score(y_train_5, y_train_pred))
print("Recall score: ", recall_score(y_train_5,y_train_pred))

#estas métricas são combinadas numa métrica chamada f1_score
#F1 = 2 * (precision*recall)/(precision+ recall), é a média harmónica das métricas
# a média harmónica dá mais peso a valores baixos portanto o F1_score só será bom e tanto a precsion como o recall forem muito bons

from sklearn.metrics import f1_score
print("F1_score: ",f1_score(y_train_5, y_train_pred))
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function') # devolve os valores resultantes na decision function

from sklearn.metrics import precision_recall_curve

precisions, recalls, tresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.xlabel("Precision")
plt.show()

#ROC Curce
#reciever operating characteristic
# curva da true positive rate contra a false posotive rate
# TPR = TP/TP+FN # falsos negativos são 5's que ele classificou como não sendo
# FPR = 1 - true negative rate(specitivity): true negative rate = TN/TN + FP
#fpr - probabilidade de rejeitar a hipotese nula, ou seja, dar falso alarme
# logo a ROC curve grafa sensitibity(recall) contra 1- specitivity

from sklearn.metrics import roc_curve
fpr, tpr, tresholds = roc_curve(y_train_5, y_scores) #fpr false positive rate, tpr -true positive rate

#m bom classificador tenta aproximar-se do canto esquerdo superior
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#uma maneira de comparar classificadores é com a área por baixo da ROC, quanto melhor -> area =1
from sklearn.metrics import roc_auc_score
print("Area under curve: ", roc_auc_score(y_train_5, y_scores))



#Using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier()
#este classificador não retorna o valor do treshold, em vez disse retorna a probabilidade que ele acha de um objeto pertencer à classe
y_probs = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method="predict_proba")
# a curva ROC  está à espera das labels e dos scores mas podemos dar as probabilidades de classes

y_scores_forest= y_probs[:, 1] #score = proba of the positive class
fpr_forest, tpr_forest, tresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label ="SGD")
plt.plot(fpr_forest, tpr_forest, label = "Random fores")
plt.xlabel("False Positive Rate")
plt.ylabel("True positive rate")
plt.legend("lower_right")
plt.show()


