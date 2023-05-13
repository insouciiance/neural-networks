import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from video_recognition import recognize_video

xception = tf.keras.applications.Xception(
            input_shape=(150, 150, 3),
            include_top=False, 
            classes=2,
            pooling='avg')
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(xception.output)
model = tf.keras.models.Model(inputs=xception.input, outputs=outputs)

# for layer in model.layers[:40]:
#     layer.trainable = False

model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
  loss='binary_crossentropy',
  metrics=['accuracy'])

train_dir = "train"
test_dir = "test"

epochs = 10
batch_size = 2

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(150, 150), batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=(150, 150), batch_size=batch_size)

model.fit(train_ds, validation_data=test_ds, epochs=epochs)

# model.save('model.keras')

# model = tf.keras.models.load_model('model.keras')

model.summary()

def show_results(model, test_ds):
  for images, true_classes in test_ds.take(5):
    predictions = model.predict(images) * 2
    true_classes = true_classes.numpy()
    print(true_classes)
    print(predictions)
    for index, img in enumerate(images):
        print('Actual label: ', true_classes[index])
        print('Predicted: ', predictions[index])
        img = img.numpy()
        plt.imshow(images[index].numpy().astype("uint8"))
        plt.show()

# show_results(model, test_ds)
recognize_video(model)
