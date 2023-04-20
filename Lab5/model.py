from keras.models import Model
from keras.layers.merging import concatenate
from keras.layers import Input, MaxPooling2D, Conv2D, GlobalAveragePooling2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization

def conv_batch_normalization(prev_layer, filter_size, kernel_size, strides=(1,1), padding="same"):
  x = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding)(prev_layer)
  x = BatchNormalization(axis=3)(x)
  x = Activation(activation="relu")(x)
  return x


def stem_block(prev_layer):
  x = conv_batch_normalization(prev_layer, filter_size=32, kernel_size=(3,3), strides=(2,2))
  x = conv_batch_normalization(x, filter_size=32, kernel_size=(3,3))
  x = conv_batch_normalization(x, filter_size=64, kernel_size=(3,3))
  x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
  x = conv_batch_normalization(x, filter_size=80, kernel_size=(1,1))
  x = conv_batch_normalization(x, filter_size=192, kernel_size=(3,3))
  x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
  return x


def inception_block_a(prev_layer, filter_size):
  branch1 = conv_batch_normalization(prev_layer, filter_size=64, kernel_size=(1,1))
  branch1 = conv_batch_normalization(branch1, filter_size=96, kernel_size=(3,3))
  branch1 = conv_batch_normalization(branch1, filter_size=96, kernel_size=(3,3))

  branch2 = conv_batch_normalization(prev_layer, filter_size=48, kernel_size=(1,1))
  branch2 = conv_batch_normalization(branch2, filter_size=64, kernel_size=(3,3))

  branch3 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding="same")(prev_layer)
  branch3 = conv_batch_normalization(branch3, filter_size=filter_size, kernel_size=(1,1))

  branch4 = conv_batch_normalization(prev_layer, filter_size=64, kernel_size=(1,1))

  output = concatenate([branch1, branch2, branch3, branch4], axis=3)
  return output


def inception_block_b(prev_layer, filter_size):
  branch1 = conv_batch_normalization(prev_layer, filter_size=filter_size, kernel_size=(1,1))
  branch1 = conv_batch_normalization(branch1, filter_size=filter_size, kernel_size=(7,1))
  branch1 = conv_batch_normalization(branch1, filter_size=filter_size, kernel_size=(1,7))
  branch1 = conv_batch_normalization(branch1, filter_size=filter_size, kernel_size=(7,1))    
  branch1 = conv_batch_normalization(branch1, filter_size=192, kernel_size=(1,7))
  
  branch2 = conv_batch_normalization(prev_layer, filter_size=filter_size, kernel_size=(1,1))
  branch2 = conv_batch_normalization(branch2, filter_size=filter_size, kernel_size=(1,7))
  branch2 = conv_batch_normalization(branch2, filter_size=192, kernel_size=(7,1))
  
  branch3 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding="same")(prev_layer)
  branch3 = conv_batch_normalization(branch3, filter_size=192, kernel_size=(1,1))
  
  branch4 = conv_batch_normalization(prev_layer, filter_size=192, kernel_size=(1,1))
  
  output = concatenate([branch1, branch2, branch3, branch4], axis=3)
  return output


def inception_block_c(prev_layer):
  branch1 = conv_batch_normalization(prev_layer, filter_size=448, kernel_size=(1,1))
  branch1 = conv_batch_normalization(branch1, filter_size=384, kernel_size=(3,3))
  branch1_1 = conv_batch_normalization(branch1, filter_size=384, kernel_size=(1,3))    
  branch1_2 = conv_batch_normalization(branch1, filter_size=384, kernel_size=(3,1))
  branch1 = concatenate([branch1_1, branch1_2], axis = 3)
  
  branch2 = conv_batch_normalization(prev_layer, filter_size=384, kernel_size=(1,1))
  branch2_1 = conv_batch_normalization(branch2, filter_size=384, kernel_size=(1,3))
  branch2_2 = conv_batch_normalization(branch2, filter_size=384, kernel_size=(3,1))
  branch2 = concatenate([branch2_1 , branch2_2], axis=3)
  
  branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding="same")(prev_layer)
  branch3 = conv_batch_normalization(branch3, filter_size=192, kernel_size=(1,1))
  
  branch4 = conv_batch_normalization(prev_layer, filter_size=320, kernel_size=(1,1))
  
  output = concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
  return output


def reduction_block_a(prev_layer):
  branch1 = conv_batch_normalization(prev_layer, filter_size=64, kernel_size=(1,1))
  branch1 = conv_batch_normalization(branch1, filter_size=96, kernel_size=(3,3))
  branch1 = conv_batch_normalization(branch1, filter_size=96, kernel_size=(3,3) , strides=(2,2))
  
  branch2 = conv_batch_normalization(prev_layer, filter_size=384, kernel_size=(3,3), strides=(2,2))
  
  branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(prev_layer)
  
  output = concatenate([branch1, branch2, branch3], axis=3)
  return output


def reduction_block_b(prev_layer):
  branch1 = conv_batch_normalization(prev_layer, filter_size=192, kernel_size=(1,1))
  branch1 = conv_batch_normalization(branch1, filter_size=192, kernel_size=(1,7))
  branch1 = conv_batch_normalization(branch1, filter_size=192, kernel_size=(7,1))
  branch1 = conv_batch_normalization(branch1, filter_size=192, kernel_size=(3,3), strides=(2,2), padding="valid")
  
  branch2 = conv_batch_normalization(prev_layer, filter_size=192, kernel_size=(1,1) )
  branch2 = conv_batch_normalization(branch2, filter_size=320, kernel_size=(3,3), strides=(2,2), padding="valid")
  
  branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(prev_layer)
  
  output = concatenate([branch1, branch2, branch3], axis=3)
  return output


def aux_classifier(prev_layer, output_size):
  x = AveragePooling2D(pool_size=(5,5), strides=(3,3))(prev_layer)
  x = conv_batch_normalization(x, filter_size=128, kernel_size=(1,1))
  x = Flatten()(x)
  x = Dense(units=768, activation="relu")(x)
  x = Dropout(rate=0.2)(x)
  x = Dense(units=output_size, activation="softmax")(x)
  return x


def inception_v3(input_size, output_size):
  input_layer = Input(shape=input_size)

  x = stem_block(input_layer)

  x = inception_block_a(prev_layer=x, filter_size=32)
  x = inception_block_a(prev_layer=x, filter_size=64)
  x = inception_block_a(prev_layer=x, filter_size=64)

  x = reduction_block_a(prev_layer=x)

  x = inception_block_b(prev_layer=x, filter_size=128)
  x = inception_block_b(prev_layer=x, filter_size=160)
  x = inception_block_b(prev_layer=x, filter_size=160)
  x = inception_block_b(prev_layer=x, filter_size=192)

  aux = aux_classifier(prev_layer=x, output_size=output_size)

  x = reduction_block_b(prev_layer=x)

  x = inception_block_c(prev_layer=x)
  x = inception_block_c(prev_layer=x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(units=2048, activation="relu")(x)
  x = Dropout(rate=0.2)(x)
  x = Dense(units=output_size, activation="softmax")(x)
    
  model = Model(inputs=input_layer, outputs=[x, aux], name="Inception-V3")    
  return model
