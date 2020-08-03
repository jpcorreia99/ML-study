import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

codings_size = 30

(X_train, Y_train) , (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32')
X_train = X_train/255.



generator = keras.models.Sequential([
    keras.layers.Dense(100, activation = "selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation = "selu"),
    keras.layers.Dense(28*28, activation="selu"),
    keras.layers.Reshape((28,28))
])

discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(150,activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator,discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False

gan.compile(loss ="binary_crossentropy", optimizer = "rmsprop")

batch_size = 32

dataset = tf.data.Dataset.from_tensor_slices(X_train) # cria um dataset em que cada tensor é uma das imagens
dataset.shuffle(buffer_size=60000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1) #acelera o fetching, ver pag 421

y = tf.constant([0.] * batch_size + [1.] * batch_size)  # criação das labels, 0- falsas, 1-verdadeiras


def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(epoch)
        for X_batch in dataset:
            print("passeou")
            #phase 1 - training the discriminator
            noise = tf.random.normal(shape=(batch_size, codings_size)) # retira noise de uma distribuição normal
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images,X_batch], axis=0)
            y = tf.constant([[0.]]*batch_size + [[1.]]*batch_size) # criação das labels, 0- falsas, 1-verdadeiras
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real,y)

            #phase 2 - trainind the generator
            noise = tf.random.normal(shape=(batch_size, codings_size))  # retira noise de uma distribuição normal
            y = tf.constant([[1.]]*batch_size) # deve gerar imagens que são todas "verdadeiras"
            discriminator.trainable = False
            gan.train_on_batch(noise,y)

train_gan(gan,dataset,batch_size,codings_size,n_epochs=1)

noise = tf.random.normal(shape=(10,codings_size)) # cria 10 tensores aliatórios
generated_images = generator(noise)

print(generated_images.shape)
for i in range(10):
    plt.imshow(generated_images[i], cmap="binary")
    plt.show()
