import tensorflow as tf
import time
import matplotlib.pyplot as plt
import models
import utils


def train_discriminator(model, dataset, n_iter=20, n_batch=25, latent_dim=100):
    half_batch = int(n_batch / 2)
    for i in range(n_iter):
        X_real, y_real = utils.generate_real_samples(dataset=dataset, num_samples=half_batch)
        _, real_acc = model.train_on_batch(X_real, y_real)
        X_fake, y_fake = utils.generate_fake_samples(model, latent_dim=latent_dim, num_samples=half_batch)
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        print('>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))


def train_gan(gan_model, latent_dim, n_epochs=20, n_batch=25):
    for i in range(n_epochs):
        x_gan = utils.generate_latent_points(latent_dim, n_batch)
        y_gan = tf.ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)


def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = utils.generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = utils.generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(dataset, generator, discriminator, gan_model, latent_dim=100, n_epochs=20, n_batch=25):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            start_time = time.time()
            # get randomly selected 'real' samples
            X_real, y_real = utils.generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = discriminator.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = utils.generate_fake_samples(generator, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = utils.generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = tf.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            time_taken = time.time() - start_time
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f Time Taken:%.2f seconds' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss, time_taken))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, generator, discriminator, dataset, latent_dim)


latent_dim = 100
discriminator = models.get_discriminator()
generator = models.get_generator(latent_dim)
gan_model = models.def_gan(generator, discriminator)
dataset = utils.load_real_samples()
train(dataset, generator, discriminator, gan_model, latent_dim)
