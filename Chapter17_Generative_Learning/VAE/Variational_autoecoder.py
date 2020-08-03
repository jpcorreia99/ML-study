import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


# samples a coding vector from the normal distribution with mean  and standart deviation
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


(train_data, _), (test_data, _) = keras.datasets.mnist.load_data()
codings_size = 10
print(train_data[0].shape[0])

encoder_inputs = keras.layers.Input(shape=train_data[0].shape)
z = keras.layers.Flatten()(encoder_inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])

variational_encoder = keras.models.Model([encoder_inputs],[codings_mean, codings_log_var, codings])

decoder_inputs = keras.layers.Input(shape =[codings_size])
x = keras.layers.Dense(100, activation = "selu")(decoder_inputs)
x = keras.layers.Dense(150, activation = "selu")(x)
x = keras.layers.Dense((train_data[0].shape)[0]*(train_data[0].shape)[1], activation = "sigmoid")(x) #28*28
outputs = keras.layers.Reshape(train_data[0].shape)(x)

variational_decoder = keras.models.Model(inputs = [decoder_inputs], outputs = [outputs])

_, _ , codings = variational_encoder(encoder_inputs)
reconstruction = variational_decoder(codings)

variational_autoencoder = keras.models.Model(inputs = [encoder_inputs], outputs = [reconstruction])



print(variational_encoder.summary())

print(variational_decoder.summary())

print(variational_autoencoder.summary())

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)

variational_autoencoder.add_loss(K.mean(latent_loss) / 784.)

variational_autoencoder.compile(loss="binary_crossentropy", optimizer="rmsprop")
history = variational_autoencoder.fit(train_data, train_data, epochs=25, batch_size=128)


codings = tf.random.normal(shape = [12, codings_size])
images = variational_decoder(codings).numpy()
plt.imshow(images[0], cmap = "binary")
plt.show()