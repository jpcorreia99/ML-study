from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()
print(wine.keys())

X = wine['data']
y = wine['target']
print(wine['feature_names'])
print(wine['DESCR'])

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)
print(tree_clf.score(X,y))

from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file="wine_tree.dot",
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    rounded=True,
    filled=True
)

import os
os.system("dot -Tpng wine_tree.dot -o wine_tree.png")