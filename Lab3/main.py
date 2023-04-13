import tensorflow as tf
import matplotlib.pyplot as plt
from helpers import *

EPOCHS = 10
NUMBERS = 10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

model = create_model()
compile_model(model)

model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

numbers = [y.argmax() for y in model.predict(X_test[:NUMBERS])]

for i in range(NUMBERS):
  print(f'Actual number: {y_test[i]}')
  print(f'Predicted number: {numbers[i]}')
  plt.figure(figsize=(1, 1))
  plt.imshow(X_test[i])
  plt.show()
