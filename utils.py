import tensorflow as tf
from tensorflow.keras import models, layers, datasets, activations
import matplotlib.pyplot as plt
import models


def load_real_samples():
    (X_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X = X_train.astype('float32')
    X = (X - 127.5) / 127.5
    return X


def generate_latent_points(latent_dim, num_samples):
    x_latent = tf.keras.backend.random_normal((num_samples, latent_dim))
    return x_latent


def generate_fake_samples(gen_model, latent_dim, num_samples):
    x_latent = generate_latent_points(latent_dim, num_samples)
    X = gen_model(x_latent)
    y = tf.keras.backend.zeros((num_samples, 1))
    return X, y


def generate_real_samples(dataset, num_samples):
    ix = tf.random.uniform((1, num_samples), minval=0, maxval=dataset.shape[0], dtype=tf.dtypes.int32, seed=None, name=None)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = tf.ones((num_samples, 1))
    return X, y


def plot_generate_fake_samples():
    latent_dim = 100
    # define the discriminator model
    model = models.get_generator(latent_dim)
    # generate samples
    num_samples = 49
    X, _ = generate_fake_samples(model, latent_dim, num_samples)
    X = X.numpy()
    generate_and_save_images(model, 1, generate_latent_points(latent_dim, num_samples))
    # scale pixel values from [-1,1] to [0,1]
    for i in range(X.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def plot(num_samples, latent_dim):
    for i in range(num_samples):
        generator = models.get_generator(latent_dim)
        X, _ = generate_fake_samples(generator, latent_dim, num_samples)
        # define subplot
        plt.subplot(7, 7, 1 + i)
        # turn off axis labels
        plt.axis('off')
        # plot single image
        plt.imshow(X[i])
    # show the figure
    plt.show()


latent_dim = 100
generator = models.get_generator(latent_dim)
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
# print("gene:", generated_image)
# generated_image = np.array(generated_image)
plt.imshow(generated_image[0] * 127.5 + 127.5, cmap='gray')
