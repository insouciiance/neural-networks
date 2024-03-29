from tensorflow import keras
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
z = np.array([0, 1, 1, 0])

if __name__ == '__main__':
  model = keras.Sequential()
  model.add(keras.layers.Dense(15, input_dim=2, activation="sigmoid"))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(x, y, epochs=3000, verbose=0)

  model.evaluate(x, z)
  print(model.predict(x).round())
