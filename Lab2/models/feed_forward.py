import tensorflow as tf

def feed_forward(neurons, input_size = 2):
  input = tf.keras.layers.Input(input_size)
  layers = input

  for n in neurons:
    layers = tf.keras.layers.Dense(n, activation='relu')(layers)
  
  output = tf.keras.layers.Dense(1)(layers)
  model = tf.keras.Model(input, output)
  return model
