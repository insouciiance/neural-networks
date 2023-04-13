import tensorflow as tf
import matplotlib.pyplot as plt

def create_model(hidden_layers = [50, 50]):
  layers = [tf.keras.layers.Flatten()]

  for n in hidden_layers:
    layers.append(tf.keras.layers.Dense(n, activation='relu'))

  layers.append(tf.keras.layers.Dense(10, activation='softmax'))

  model = tf.keras.models.Sequential(layers=layers)
  return model

def compile_model(model, initial_learning_rate = 0.001, decay_steps = 100, decay_rate = 0.99, loss = 'sparse_categorical_crossentropy'):
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

  model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
    loss=loss,
    metrics=['accuracy'])
