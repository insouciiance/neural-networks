from tensorflow import keras
from model import inception_v3
from config import *
from loading import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
model = inception_v3((WIDTH,HEIGHT,3), len(CLASSES))

learning_rate = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=100000, decay_rate=0.96)

model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.SGD(learning_rate=learning_rate))

train_ds, validation_ds, test_ds = prepare_data()

# model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds, verbose=1)

# model.save(MODEL_PATH)

model = keras.models.load_model(MODEL_PATH)

def show_results(model, test_ds):
  for images, true_classes in test_ds.take(1):
    predictions = model.predict(images)[1]
    true_classes = true_classes.numpy()
    classes = np.argmax(predictions, axis=1)
    print(true_classes)
    print(classes)
    for index, img in enumerate(images):
        print('Actual label: ', CLASSES[true_classes[index]])
        print('Predicted label: ', CLASSES[classes[index]])
        img = img.numpy()
        plt.imshow(images[index].numpy().astype("uint8"))
        plt.show()

show_results(model, test_ds)
