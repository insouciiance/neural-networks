import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

# (train_dataset, test_dataset), info = tfds.load(
#     'yelp_polarity_reviews',
#     split=['train[:50000]', 'test[:5000]'],
#     with_info=True,
#     as_supervised=True)

# BUFFER_SIZE = 10000
# BATCH_SIZE = 16

# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# VOCAB_SIZE = 1000
# encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
# encoder.adapt(train_dataset.map(lambda text, label: text))

# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])

# model.fit(train_dataset, epochs=5,
#           validation_data=test_dataset,
#           validation_steps=30)

# model.save('model.tf', save_format='tf')

# test_loss, test_acc = model.evaluate(test_dataset)

# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

model = tf.keras.models.load_model('model.tf')

inputs = [
  'The movie was not good. It was pure trash, complete garbage. I would not recommend anyone to spend their time watching it.',
  'I really liked watching the movie. There were some weird things but the good completely overwrites all the bad there is.',
  'The movie was cool. The animation and the graphics were out of this world. I would recommend this movie'
  ]

for input in inputs:
  predictions = model.predict(np.array([input]))
  print(input)
  print(predictions)
