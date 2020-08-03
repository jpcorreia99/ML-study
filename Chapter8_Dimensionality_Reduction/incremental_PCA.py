from tensorflow import keras
from sklearn.decomposition import IncrementalPCA
import numpy as np
import os

#usar quando vou carregando os meus dados aos poucos

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

nsamples, nx, ny = train_data.shape
train_data = train_data.reshape((nsamples,nx*ny))

nsamples, nx, ny = test_data.shape
test_data = test_data.reshape((nsamples,nx*ny))

#np.save('train_data', train_data)

'''n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(train_data, n_batches):
    inc_pca.partial_fit(x_batch)'''

#se o ficheiro estiver em memória, o numpy consegue mapear e acedemos aos poucos

x_mm =np.memmap('chapter_8/train_data.npy', mode="readonly", shape=(nsamples,nx*ny))
inc_pca = IncrementalPCA(n_components=154, batch_size=256)
inc_pca.fit(x_mm)

print(np.cumsum(inc_pca.explained_variance_ratio_))

train_data= inc_pca.fit_transform(train_data)
test_data = inc_pca.transform(test_data)

print(train_data.shape)
print(test_data.shape)

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=train_data[0].shape),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ['acc']
)
print(model.summary())

model.fit(train_data,
          train_labels,
          epochs = 10,
          validation_data=(test_data, test_labels))
