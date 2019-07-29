import os
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle
import glob
from konlpy.tag import Okt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

maxlen = 200 # 200개 단어 이후는 버려
max_words = 1000  # 빈도 높은 1,000개의 단어만 사용
text = ""
model = load_model('..\\result\\country_classification_v01.h5')

# loading
with open('..\\result\\tokenizer_v01.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

okt=Okt()
contents_list = []
for file in glob.glob("..\\예측데이터\\태국\\*.txt"): # 전체 파일 read
    print("file name = " + file)
    targetFile = open(file, mode='rt', encoding='utf-8')
    contents = targetFile.read()
    contents_list.append(okt.nouns(contents))  # 한 행씩 담기

for i in range(0, len(contents_list)) :
    print(contents_list[i])
x_data_ToSequence = tokenizer.texts_to_sequences(contents_list)
npContents = np.array(pad_sequences(x_data_ToSequence, maxlen=maxlen))
yhat = model.predict_classes(npContents)
print(yhat)