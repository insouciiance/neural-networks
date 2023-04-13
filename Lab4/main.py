import tensorflow as tf
from helpers import *
from config import *
from model import alexnet

train_images, \
train_labels, \
test_images, \
test_labels, \
validation_images, \
validation_labels = load_cifar10(TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE)

train_ds, train_ds_size = prepare_dataset(train_images, train_labels)
test_ds, test_ds_size = prepare_dataset(test_images, test_labels)
validation_ds, validation_ds_size = prepare_dataset(validation_images, validation_labels)

print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

model = alexnet()

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(train_ds,
          epochs=EPOCHS,
          validation_data=validation_ds,
          validation_freq=1)

model.save(MODEL_PATH)

# model = keras.models.load_model(MODEL_PATH)

model.evaluate(test_ds)

show_results(model, test_ds, test_images)
