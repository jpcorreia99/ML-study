import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

strat_train_set = pd.read_pickle("strat_train_set.pickle")
strat_test_set = pd.read_pickle("strat_test_set.pickle")


housing = strat_train_set.drop("median_house_value",axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()



#2 maneiras de tratar de NA's
#1 dropar os valores
 #housing.dropna(subset = "total_bedrooms")
#2 substiuir os NA's por algum valor (0, média, mediana)
 #median =
# ["total_bedrooms"].median()
 #housing.fillna(median, inplace=True)

#outra forma de fazer o #2
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
#since the median only takes numeric values, drop ocean proximity
housing_num = housing.drop("ocean_proximity", axis = 1)
imputer.fit(housing_num)
print(imputer.statistics_) #mediana de cada atriburo

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)

#text and categorical encoding
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head())

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)

#PIPELINES

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scalar", StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr[:10])
#custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
class ProcessOceanProximity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_to_keep = "INLAND"
        self.categories_to_replace = ["<1H OCEAN","NEAR OCEAN","NEAR BAY", "ISLAND"]
        self.new_category_name ="NOT_INLAND"
    def fit(self, X, y= None):
        return self
    def transform(self, X, y=None):
        X.replace(to_replace=self.categories_to_replace, value=self.new_category_name, inplace=True)

#full pipeline withou separation
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num) #lista das categorias numéricas
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OrdinalEncoder(), cat_attribs)
    #adicionar o encoding
])

housing["ocean_proximity"].replace(to_replace= ["<1H OCEAN","NEAR OCEAN","NEAR BAY", "ISLAND"], value="NOT_INLAND", inplace=True)
housing_prepared = full_pipeline.fit_transform(housing)
#Nota: concatenou a matriz numérica com a matriz dos one-hot



housing_test = strat_test_set.drop("median_house_value",axis = 1)
housing_test["ocean_proximity"].replace(to_replace= ["<1H OCEAN","NEAR OCEAN","NEAR BAY", "ISLAND"], value="NOT_INLAND", inplace=True)
housing_test_prepared = full_pipeline.transform(housing_test)
housing_test_labels = strat_test_set["median_house_value"].copy()


np.save("housing_test_prepared", housing_test_prepared)
np.save("housing_test_labels", housing_test_labels.to_numpy())

np.save("housing_prepared", housing_prepared)
np.save("housing_labels", housing_labels.to_numpy())
print(list(strat_train_set))
print(housing[:5])
print(housing_prepared[:5])