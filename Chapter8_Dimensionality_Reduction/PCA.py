import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()


train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

nsamples, nx, ny = train_data.shape
reshaped_train_data = train_data.reshape((nsamples,nx*ny))

nsamples, nx, ny = test_data.shape
reshaped_test_data = test_data.reshape((nsamples,nx*ny))

#pca = PCA(n_components= 0.95)
pca = PCA(n_components=154, svd_solver = "randomized") #randomized, uma um método para tentar convergir poara os valores, é mais rápido
train_data_reduced = pca.fit_transform(reshaped_train_data)
test_data_reduced = pca.transform(reshaped_test_data) #este segundpo já não é fit pois ele encontraria PCA's diferentes e teríamos informações diferentes no dataset de treino e no de teste

print(train_data_reduced.shape)


'''model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=train_data_reduced[0].shape),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ['acc']
)
print(model.summary())

model.fit(train_data_reduced,
          train_labels,
          epochs = 10,
          validation_data=(test_data_reduced, test_labels))
'''

#cumsum = np.cumsum(pca.explained_variance_ratio_)

#res = pca.inverse_transform(train_data_reduced)


'''for i in range(5):
    plt.imshow(train_data[i], cmap="binary")
    plt.show()
    plt.imshow(res[i].reshape(28,28),cmap="binary")
    plt.show()

print(train_data_reduced.shape)

plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot(cumsum, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()'''

4
