import numpy as np
import func as func
import helpers as helpers
from models.feed_forward import *
from models.cascade_forward import *
from models.elman import *

EPOCHS = 500
N = 1500
TRAIN_RATIO = 0.75

n_train = int(N * TRAIN_RATIO)
X = np.random.randint(10, size=(N, 2))
print(X)
Y = np.array([func.func(x, y) for x, y in X])
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

def test_model(model):
  helpers.compile_model(model)
  history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X_test, Y_test))
  helpers.graph(history.history['loss'], history.history['val_loss'])

def main():
  test_model(feed_forward([10]))
  test_model(feed_forward([20]))
  test_model(cascade_forward([20]))
  test_model(cascade_forward([10, 10]))
  test_model(elman([15]))
  test_model(elman([5, 5, 5]))

if __name__ == "__main__":
    main()
