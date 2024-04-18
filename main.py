import time
import glob
import tensorflow as tf
from keras import layers, Model
import os

image_folder = 'propoganda' # folder with soviet propaganda for training
batch_size = 32
img_height = 128
img_width = 128

# image normalization
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_height, img_width])
    image = (image - 0.5) * 2.0
    return image

image_paths = tf.data.Dataset.list_files(glob.glob(image_folder+'/*.jpg'), shuffle=False)

images_dataset = image_paths.map(load_and_preprocess_image)

# batch, shuffle, and prefetch the dataset for training
images_dataset = images_dataset.batch(32).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)

def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),  # Adjusted for 8x8 starting size
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),  # start with 8x8 feature map
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 16x16
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 32x32
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 64x64
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 128x128
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')  # output layer to match the image shape
    ])
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
        # Adjust to match resolution of the preprocessed images
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model



# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()



output_folder = 'output13040224_2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def save_good_images(model, test_input, epoch, threshold=0.5):
    predictions = model(test_input, training=False)
    decision = discriminator(predictions, training=False)

    for i in tf.range(predictions.shape[0]):
        if decision[i] > threshold:
            image = tf.cast((predictions[i] * 127.5) + 127.5, tf.uint8)
            epoch_str = tf.strings.as_string(epoch, width=4, fill='0')
            i_str = tf.strings.as_string(i)
            image_path = tf.strings.join([output_folder, "/image_at_epoch_", epoch_str, "_", i_str, ".png"])
            tf.io.write_file(image_path, tf.io.encode_png(image))


noise_dim = 100
num_examples_to_generate=128
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images, epoch):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # saving good images
    if epoch % 10 == 0:
        save_good_images(generator, seed, epoch)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, epoch)

        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch + 1, save_path))

        if epoch % 10 == 0 or epoch == epochs - 1:
            save_good_images(generator, seed, epoch + 1)
        print(f"Time for epoch {epoch + 1} is {time.time() - start:.2f} sec")


import os

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

train(images_dataset, 1000)
