from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Embedding, GlobalMaxPooling2D, BatchNormalization
from keras.layers.merge import *
from tensorboardcolab import *
from numpy import array
from keras.utils import np_utils
import keras
import numpy as np
import glob
import os

NP_FILE_PATH = "/content/gdrive/My Drive/tensorflow/data/"
FILE_COUNT = 200

embedding_size = 64
verbose, epochs, batch_size = 1, 100, 1024
filters = 32

x_data = np.load(NP_FILE_PATH + 'x'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
y_data = np.load(NP_FILE_PATH + 'y'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
voca_arr = np.load(NP_FILE_PATH + 'v'+ str(FILE_COUNT)+'.npy', allow_pickle=True)
label_names = np.load(NP_FILE_PATH + 'l'+ str(FILE_COUNT)+'.npy', allow_pickle=True)

# x_data = sequence.pad_sequences(x_data, dtype='int32', padding='post', truncating='post', value=0)
y_data = np_utils.to_categorical(y_data)

voca_length, input_length, label_length = len(voca_arr) + 1, len(x_data[0]), len(label_names)



# input layer
main_input = Input(shape=(128, ), dtype='int32')
embedded_input = Embedding(voca_length, embedding_size, input_length=(128,))(main_input)
reshaped_input = Reshape((input_length, embedding_size, 1))(embedded_input)

# first feature extractor
conv1 = Conv2D(filters, kernel_size=(8, embedding_size), activation='relu')(reshaped_input)
batc1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 1))(batc1)
dout1 = Dropout(0.25)(pool1)
flat1 = Flatten()(pool1)

# second feature extractor
conv2 = Conv2D(filters, kernel_size=(16, embedding_size), activation='relu')(reshaped_input)
batc2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 1))(batc2)
dout2 = Dropout(0.25)(pool2)
flat2 = Flatten()(pool2)

# third feature extractor
conv3 = Conv2D(filters, kernel_size=(32, embedding_size), activation='relu')(reshaped_input)
batc3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 1))(batc3)
dout3 = Dropout(0.25)(pool3)
flat3 = Flatten()(dout3)

# merge feature extractors
merge = concatenate([flat1, flat2, flat3])

# interpretation layer
# prediction output
dens1 = Dense(256, activation='relu')(merge)
batc5 = BatchNormalization()(dens1)
dout5 = Dropout(0.25)(batc5)
output = Dense(label_length, activation='softmax')(dout5)
model = Model(inputs=main_input, outputs=output)

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize layers
# print(model.summary())

# fit
tbc=TensorBoardColab()
model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True, validation_split=0.1, callbacks=[TensorBoardColabCallback(tbc)])
tbc.close()


