from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Embedding, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from tensorboardcolab import *
from numpy import array
import numpy as np
import glob
import os

NP_FILE_PATH = "/content/gdrive/My Drive/tensorflow/data/"
FILE_COUNT = 200

embedding_size = 64
verbose, epochs, batch_size = 1, 30, 256

x_data = np.load(NP_FILE_PATH + 'x'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
y_data = np.load(NP_FILE_PATH + 'y'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
voca_arr = np.load(NP_FILE_PATH + 'v'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
label_names = np.load(NP_FILE_PATH + 'l'+ str(FILE_COUNT)+'.npy', allow_pickle=True)

# x_data = sequence.pad_sequences(x_data, dtype='int32', padding='post', truncating='post', value=0)
y_data = np_utils.to_categorical(y_data)

voca_length, input_length, label_length = len(voca_arr) + 1, len(x_data[0]), len(label_names)

model = Sequential()
model.add(Embedding(voca_length, embedding_size, input_length=input_length))
print(model.output_shape)
model.add(Reshape((input_length, embedding_size, 1)))
model.add(Conv2D(filters=32,
                 kernel_size=(10, embedding_size),
                 activation='relu',
                 input_shape=x_data.shape,
                 padding='valid',
                 strides=1))
model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(Dropout(0.25))
print(model.output_shape)

model.add(Reshape((59, 32, 1)))
model.add(Conv2D(filters=64,
                 kernel_size=(10, 32),
                 activation='relu',
                 padding='valid',
                 strides=1))
model.add(MaxPooling2D(pool_size=(4, 1)))
model.add(Dropout(0.5))
print(model.output_shape)

model.add(Flatten())
print(model.output_shape)

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(label_length, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tbc=TensorBoardColab()
model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True, validation_split=0.1, callbacks=[TensorBoardColabCallback(tbc)])
tbc.close()

print("TRAINING COMPLETED!")

