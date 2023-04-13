import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
from config import *

def load_cifar10(train_size, test_size, validation_size):
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  validation_images = train_images[train_size:(train_size + validation_size)]
  validation_labels = train_labels[train_size:(train_size + validation_size)]

  train_images = train_images[:train_size]
  train_labels = train_labels[:train_size]

  test_images = test_images[:test_size]
  test_labels = test_labels[:test_size]

  return (train_images, train_labels, test_images, test_labels, validation_images, validation_labels)

def process_images(image, label):
  # Normalize images to have a mean of 0 and standard deviation of 1
  image = tf.image.per_image_standardization(image)
  # Resize images from 32x32 to 227x227
  image = tf.image.resize(image, (227,227))
  return image, label

def prepare_dataset(images, labels):
  ds = tf.data.Dataset.from_tensor_slices((images, labels))
  ds_size = tf.data.experimental.cardinality(ds).numpy()
  ds = (ds
        .map(process_images)
        .shuffle(buffer_size=ds_size)
        .batch(batch_size=32, drop_remainder=True))
  return ds, ds_size

def show_results(model, test_ds, test_images):
  for images, true_classes in test_ds.take(1):
    predictions = model.predict(images)
    true_classes = true_classes.numpy().squeeze(axis=1)
    classes = np.argmax(predictions, axis=1)
    print(true_classes)
    print(classes)
    for index, img in enumerate(images):
        print('Actual label: ', CLASS_NAMES[true_classes[index]])
        print('Predicted label: ', CLASS_NAMES[classes[index]])
        img = img.numpy()
        plt.imshow(test_images[index])
        plt.show()
