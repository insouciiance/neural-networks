import tensorflow as tf

def cascade_forward(neurons, input_size = 2):
  input = tf.keras.layers.Input(input_size)
  layers = input

  for n in neurons:
    x = tf.keras.layers.Dense(n, activation='relu')(layers)
    layers = tf.keras.layers.concatenate([layers, x])
  
  output = tf.keras.layers.Dense(1)(layers)
  model = tf.keras.Model(input, output)
  return model
