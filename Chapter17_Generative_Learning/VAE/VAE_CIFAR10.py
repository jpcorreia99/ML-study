import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


print(tf.version)


class Sampling(keras.layers.Layer):
    def __call__(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

print(tf.version)

(X_train,_), (X_test,_) = keras.datasets.cifar10.load_data()
X_train= X_train/255
X_test = X_test/255

codings_size = 500


inputs = keras.layers.Input(shape=X_train[0].shape)

z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(512, activation="selu")(z)
z = keras.layers.Dense(256, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.models.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(512, activation="selu")(decoder_inputs)
x = keras.layers.Dense(1024, activation="selu")(x)
x = keras.layers.Dense(32 * 32 * 3, activation="sigmoid")(x)
outputs = keras.layers.Reshape(X_train[0].shape)(x)
variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

print(variational_ae.summary())

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)
variational_ae.add_loss(K.mean(latent_loss) / 1024.)
variational_ae.compile(loss="binary_crossentropy", optimizer="adam", metrics=[rounded_accuracy])
history = variational_ae.fit(X_train, X_train, epochs=50, batch_size=256)

variational_ae.save("VAE_CIFAR10.h5")

print(images)
codings = tf.random.normal(shape = [12, codings_size])
images = variational_decoder(codings)
for i in range(12):
    plt.imshow(images[i])
    plt.show()

