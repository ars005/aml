# B. Implementation of Generative Adversarial Network (GAN)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize data
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# Parameters
latent_dim = 100
batch_size = 128
epochs = 10

# Generator
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(negative_slope=0.2),
    Dense(256),
    LeakyReLU(negative_slope=0.2),
    Dense(28 * 28, activation="tanh"),
    Reshape((28, 28, 1))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256),
    LeakyReLU(negative_slope=0.2),
    Dense(128),
    LeakyReLU(negative_slope=0.2),
    Dense(1, activation="sigmoid")
])

# Compile Discriminator
discriminator.compile(
    optimizer=Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Combined GAN
discriminator.trainable = False

gan = Sequential([generator, discriminator])

gan.compile(
    optimizer=Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)

# Training
for epoch in range(epochs):

    # Real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]

    # Fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise, verbose=0)

    # Train discriminator
    discriminator.trainable = True
    discriminator.train_on_batch(
        real_images, np.ones((batch_size, 1))
    )
    discriminator.train_on_batch(
        fake_images, np.zeros((batch_size, 1))
    )

    # Train generator
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gan.train_on_batch(
        noise, np.ones((batch_size, 1))
    )

    if (epoch + 1) % 2 == 0:
        print("Epoch:", epoch + 1, "completed")

# Generate synthetic images
noise = np.random.normal(0, 1, (16, latent_dim))
images = generator.predict(noise, verbose=0)
images = (images + 1) / 2

# Display generated images
plt.figure(figsize=(8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.suptitle("Synthetic Handwritten Digits using GAN")
plt.show()