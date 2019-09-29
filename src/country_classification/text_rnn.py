from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Reshape, Dense, LSTM, Embedding, SpatialDropout1D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.constraints import max_norm
from numpy import array
from tensorboardcolab import *
import numpy as np
import glob
import os

NP_FILE_PATH = "/content/gdrive/My Drive/tensorflow/data/"
FILE_COUNT = 500

embedding_size = 32
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
model.add(SpatialDropout1D(0.2))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(label_length, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tbc=TensorBoardColab()
model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True, validation_split=0.1, callbacks=[TensorBoardColabCallback(tbc)])
tbc.close()

print("TRAINING COMPLETED!")

