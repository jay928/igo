import os
import glob
import numpy as np
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
import random
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pickle

maxlen = 100 # 100개 단어 이후는 버려
max_words = 500  # 빈도 높은 500개의 단어만 사용
text_dir = "..\\raw"
os.chdir(text_dir)
dir_list = os.listdir(text_dir)
# 레이블(국가명)
countries = []
label = {}
cnt = 0
for i in range(0, len(dir_list)) :
    if dir_list[i].split(" ")[0] not in countries :
        countries.append(dir_list[i].split(" ")[0])
        label[dir_list[i].split(" ")[0]] = cnt
        cnt += 1
print(label)

# 데이터 로드
okt=Okt()
contents_list = []
x_list = []
for country in countries:  # 레이블 선택
    print("country name = " + country)
    for dir_element in  dir_list :
        if country in dir_element : # 해당 국가와 관련된 폴더 선택
            print("dir name = " + dir_element)
            for file in glob.glob(dir_element+"/blog.naver.com/*.txt"): # 전체 파일 read
                print("file name = " + file)
                targetFile = open(file, mode='rt', encoding='utf-8')
                contents = targetFile.read()
                contents_list.append([label[country], okt.nouns(contents)])    # 한 행씩 담기

print("shuffling....")
random.shuffle(contents_list)   # 셔플링
npContents = np.array(contents_list)
print("Tokenizing....")
tokenizer = Tokenizer(num_words=max_words)  # 빈도높은 수까지만

print("FitOnTexts....")
tokenizer.fit_on_texts(npContents[:,1])

# saving
with open('..\\result\\tokenizer_v03.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
#with open('tokenizer.pickle', 'rb') as handle:
#    tokenizer = pickle.load(handle)

print("Texts To Sequences....")
x_data_ToSequence = tokenizer.texts_to_sequences(npContents[:,1])

print("Pad Sequences....")
x_data = pad_sequences(x_data_ToSequence,maxlen=maxlen)
y_data = np_utils.to_categorical(npContents[:,0])

# train / test 셋 생성
x_train = x_data[:int(len(x_data) * 0.8)]
x_test = x_data[int(len(x_data) * 0.8):]

y_train = y_data[:int(len(y_data) * 0.8)]
y_test = y_data[int(len(y_data) * 0.8):]

# 모델 학습하기
model = Sequential()
model.add(Embedding(max_words+1, 120))  # +1은 패딩해준 데이터가 들어갈 공간
model.add(LSTM(120))
model.add(Dense(10, activation='softmax'))  # 10개 나라....

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test,y_test))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

model.save('..\\result\\country_classification_v03.h5')