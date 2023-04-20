import os
import tensorflow as tf
from config import *

def load_files():
  files = tf.data.Dataset.list_files("C:/Users/nikkv/source/repos/NeuralNetworks/Lab5/Data" + "/*/*.jpg")
  return files


def load_image(file):
  class_matches = tf.strings.split(file, os.path.sep)[-2] == CLASSES
  image_class = tf.argmax(class_matches)
  image = tf.io.read_file(file)
  image = tf.io.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [HEIGHT, WIDTH])
  return image, image_class


def load_images(files):
  return files.map(load_image).batch(batch_size=BATCH_SIZE)


def prepare_data():
  files = load_files()
  images = load_images(files)
  count = len(images)

  train_size = int(TRAIN_SIZE * count)
  validation_size = int(VALIDATION_SIZE * count)
  test_size = int(TEST_SIZE * count)

  train_ds = images.take(train_size)
  validation_ds = images.skip(train_size).take(validation_size)
  test_ds = images.skip(train_size + validation_size).take(test_size)

  return train_ds, validation_ds, test_ds
