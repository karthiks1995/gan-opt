import tensorflow as tf
from tensorflow.keras import models, layers, datasets, activations

def get_discriminator(in_shape = (32, 32, 3)):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (3,3), padding='same', input_shape=(32, 32, 3)))
  model.add(layers.LeakyReLU(alpha=0.2))
  #Hidden
  model.add(layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
  model.add(layers.LeakyReLU(alpha=0.2))
  #Hidden
  model.add(layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
  model.add(layers.LeakyReLU(alpha=0.2))
  #Hidden
  model.add(layers.Conv2D(256, (3, 3), padding='same', strides=(2, 2)))
  model.add(layers.LeakyReLU(alpha=0.2))
  #Classifier
  model.add(layers.Flatten())
  model.add(layers.Dropout(0.4))
  model.add(layers.Dense(1, activation='sigmoid'))
  #Model Compile
  opt = get_g_optimizer()
  loss = 'binary_crossentropy'
  model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
  return model

def get_generator(latent_dim):
  model = tf.keras.Sequential()
  #Dense
  model.add(layers.Dense(256 * 4 * 4, input_dim = latent_dim))
  model.add(layers.LeakyReLU(alpha = 0.2))
  model.add(layers.Reshape((4, 4, 256)))
  #Unsample to 8X8
  model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.LeakyReLU(alpha = 0.2))
  #Unsample to 16X16
  model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.LeakyReLU(alpha = 0.2))
  #Unsample to 32X32
  model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.LeakyReLU(alpha = 0.2))
  #output
  model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
  return model

def def_gan(generator, discriminator):
  discriminator.trainable = False
  model = tf.keras.Sequential()
  model.add(generator)
  model.add(discriminator)
  opt = get_g_optimizer()
  pt = get_g_optimizer()
  loss = 'binary_crossentropy'
  model.compile(optimizer=opt, loss=loss)
  return model

def generator_loss(fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  return cross_entropy(tf.ones_like(fake_output), fake_output)

def disriminator_loss(real_output, fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def get_d_optimizer():
  return tf.keras.optimizers.Adam(2e-4)

def get_g_optimizer():
  return tf.keras.optimizers.Adam(2e-4)