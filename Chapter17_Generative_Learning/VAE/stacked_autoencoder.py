from tensorflow import keras
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)



stacked_encoder= keras.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape = x_train[0].shape),
    keras.layers.Dense(30, activation="relu")
])

stacked_decoder = keras.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=[30]),
    keras.layers.Dense(784, activation="sigmoid")
])

autoencoder = keras.Sequential([stacked_encoder,stacked_decoder])

print(autoencoder.summary())

autoencoder.compile(optimizer = keras.optimizers.SGD(lr=1.5),
                    loss = 'binary_crossentropy',
                    metrics=['acc'] )

autoencoder.fit(x_train,
              x_train,
              epochs=10,
              batch_size=256,
              validation_data=(x_test, x_test),
              shuffle=True)

predictions = autoencoder.predict(x_test)

x_test = x_test.reshape(10000,28,28)
predictions = predictions.reshape(10000,28,28)

for i in range(5):
    plt.imshow(x_test[i], cmap="binary")
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i],cmap="binary")
    plt.title("predicted")
    plt.show()