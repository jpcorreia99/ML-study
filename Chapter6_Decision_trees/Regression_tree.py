import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_regressor = DecisionTreeRegressor(max_depth=2)
tree_regressor.fit(X,y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_regressor,
    out_file="tree_reg.dot",
    rounded=True,
    filled=True
)

import os
os.system("dot -Tpng tree_reg.dot -o tree_teg.png")
