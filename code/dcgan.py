import tensorflow as tf
from tensorflow.keras import layers 
import matplotlib.pyplot as plt 
import time 
import cv2 
from skimage import io
import os
import numpy as np
import random 
from PIL import ImageFile
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

size = 64

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_pokemon_data():
    images = []
    
    folder = "../data/pokemon"

    for filename in os.listdir(folder):
        img = np.array(Image.open(os.path.join(folder,filename)).convert('RGBA'))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (size, size)) 

        if img is not None:
            images.append(img)

    return images

train_images = np.array(load_pokemon_data())
  
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

#hyperparameters

BUFFER_SIZE = 814
BATCH_SIZE = 20

EPOCHS = 5000
noise_dim = 100
num_examples_to_generate = 16


train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def make_newgenerator_model():
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(size/8)*int(size/8)*512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Reshape((int(size/8), int(size/8), 512)))
    assert model.output_shape == (None, int(size/8), int(size/8), 512)  # Note: None is the batch size
    
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(size/8), int(size/8), 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(size/4), int(size/4), 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(size/4), int(size/4), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(size/2), int(size/2), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, size, size, 3)

    return model

def make_newdiscriminator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
    model.add(layers.experimental.preprocessing.RandomRotation(0.1))
    model.add(layers.experimental.preprocessing.RandomTranslation(0.1, 0.1))
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[size, size, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_generator_model():
    model = tf.keras.Sequential()
  
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    
    ones_like = tf.convert_to_tensor(np.random.uniform(0.9,1,len(real_output)))
    zeros_like = tf.convert_to_tensor(np.random.uniform(0,0.1,len(fake_output)))
        
    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_loss
    
    real_loss = cross_entropy(ones_like, real_output)
    fake_loss = cross_entropy(zeros_like, fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss
    

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      
      tf.print(gen_loss)
      
  
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)
      
    if epoch % 40 == 0:
        generate_and_save_images(generator,
                             epoch + 1,
                             seed)
        
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  generate_and_save_images(generator,
                           epochs,
                           seed)
   
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(np.array(predictions[i] * 127.5 + 127.5).astype(int))
      plt.axis('off')


  plt.savefig('../data/generated/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()
  

#generator

generator = make_newgenerator_model()

#discriminator

discriminator = make_newdiscriminator_model()

seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)
