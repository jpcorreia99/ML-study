import os
import tarfile
from urllib import request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    request.urlretrieve(housing_url, tgz_path) # vai buscar ao url e guarda no tgz_path, com o nome indicado na linha em cima
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)

#fetch_housing_data()

housing = load_housing_data()
print(housing.head())
print(housing.info())
#print(housing[['ocean_proximity']])
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

#housing.hist(bins=50, figsize=(20,15)) #50 barras
#plt.show()

#creating test set

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # np.random.permutation(10): array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] # iloc, interger location, basicamente indexing

#housing = housing.reset_index() # adds an index collumn
train_data, test_data = split_train_test(housing, 0.2)
print(train_data.head())

#mas nós queremos manter a representatividade dos dados, isto é se o dataset fosse 60%male e 40% female, o test e o train devem seguir estas percentagens

#separa os median_incomes em 5 categorias, consoante o intervalo em,que estão, relembrar que os mediam incomes são *10000$
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])
#housing["income_cat"].hist()
#plt.show()

#com isto podemos fazer sampling baseado em strats

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #parte em 2, 20%, random seed 42

strat_train_set = None # para no IDE não estar a dar erro de não estar definido
strat_test_set = None
#na verdade retorna um gerador que só vai sher chamado uma vez porque train index e test index são listas de indices
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#as percentagens mantêm-se
print(housing["income_cat"].value_counts()/len(housing))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

#dropar a income_cat
strat_train_set.drop("income_cat",axis=1, inplace = True) #inplace = True, altera diretamente os dados, False, retorna uma cópia
strat_test_set.drop("income_cat", axis = 1, inplace = True)
print(strat_train_set.info())

#visualizing
housing = strat_train_set.copy()
#housing.plot(kind = "scatter", x="longitude", y = "latitude", alpha=0.1) # alfa permite ver melhor a densidade de pontos
#plt.show()

housing.plot(kind= "scatter", x = "longitude", y = "latitude",
             s = housing["population"]/100, label = "population", figsize = (10,7), # cada ponto é um circulo que aumenta de tamanho consoante a densidade populacional
             c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True) #a colorização do ponto depende o median house value
#c - argumento possivel de usar no scatter para saber como atribuir cor a cada ponto
plt.legend()
#vermelho - mais caro, circulos grandes- maior densidade populacional
plt.show()

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
#vemos que o median income é o que mais influência tem

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes]) #nota, a diagonal, em vez de ser plotted contra si própria, mostra um simples histograma da classe
plt.show()
housing.plot(kind="scatter", x ="median_income", y ="median_house_value", alpha=0.1) #mostra uma correlação linear com um cap nos 500000$
plt.show()

#Attribute Combinations

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"] #ambos estes valores vêm sobre a forma do total no distrito
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

def pass_data():
    return strat_train_set, strat_test_set


strat_train_set.to_pickle("./strat_train_set.pickle")
strat_test_set.to_pickle("./strat_test_set.pickle")
## Preparing the Data
housing = strat_train_set.drop("median_house_value",axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

