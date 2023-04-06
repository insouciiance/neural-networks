import tensorflow as tf
import matplotlib.pyplot as plt

def compile_model(model, initial_learning_rate = 0.001, decay_steps = 100, decay_rate = 0.99, loss = "log_cosh"):
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

  model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
    loss=loss)
  
  model.summary()

def graph(train_loss, test_loss):
  plt.title('Training loss (log_cosh)')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.plot(train_loss, label='Train loss')
  plt.plot(test_loss, label='Test loss')
  plt.grid()
  plt.legend()
  plt.show()
