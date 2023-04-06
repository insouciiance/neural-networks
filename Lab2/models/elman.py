import tensorflow as tf

def elman(neurons, input_size = 2):
  input = tf.keras.layers.Input(input_size)
  layers = tf.expand_dims(input, axis = 1)
  layers = tf.keras.layers.SimpleRNN(neurons[0])(layers)

  for n in neurons[1:]:
    layers = tf.expand_dims(layers, axis = 1)
    layers = tf.keras.layers.SimpleRNN(n, activation='relu')(layers)
  
  output = tf.keras.layers.Dense(1)(layers)
  model = tf.keras.Model(input, output)
  return model
