from numpy import array
import numpy as np
import glob
import os

NP_FILE_PATH = "/content/gdrive/My Drive/tensorflow/data/"
VOCA_PATH = "/content/gdrive/My Drive/tensorflow/data/allvocaset.csv"
DATA_PATH = "/content/gdrive/My Drive/tensorflow/data/raw_all"
DETAIL_PATH = '/blog.naver.com/*.txt'

FILE_COUNT = 100

def npIndex(np_array, object, plus):
    result = np.where(np_array==object)
    return np.array(result).tolist()[0][0] + plus


def findFolders(path):
    folders = []
    for r, d, f in os.walk(path):
        for folder in d:
            folderName = os.path.join(r, folder)
            if "blog.naver.com" in folderName:
                continue

            folders.append(folderName)

    return folders


def findLabels(folders):
    labels = []
    for folder in folders:
        label = folder.split("/")[-1].split(" ")[0]
        labels.append(label)

    return array(list(set(labels)))


def findY(y_array, file):
    for y in y_array:
        if y in file:
            return y

    return None


def loadData(folders, labels, voca):
    x_data = []
    y_data = []

    for i, folder in enumerate(folders):
        files = [f for f in glob.glob(folder + DETAIL_PATH, recursive=True)]

        y = findY(labels, files[0])
        y_as_int = npIndex(labels, y, 0)

        x_line = []

        for j, file in enumerate(files):
            if j >= FILE_COUNT:
                break

            print("(", j+1, "/", str(FILE_COUNT), ", ", i+1, "/", len(folders), ") Importing file : ", file)

            with open(file, 'r', encoding='UTF-8') as f:
                lines = f.readlines()

                for line in lines:
                    for index in range(0, len(line)):
                        char = line[index]
                        if char not in voca:
                            continue

                        x_line.append(npIndex(voca, char, 1))

        for ii in range(0, 128-len(x_line)%128):
            x_line.append(0) #add pedding

        x_line_array = array(x_line)

        reshaped = x_line_array.reshape(-1, 128)

        for x in reshaped:
            x_data.append(x.tolist())
            y_data.append(y_as_int)

    return x_data, y_data


voca_arr = np.loadtxt(VOCA_PATH, dtype=np.str, encoding='UTF-8')
folder_names = findFolders(DATA_PATH)
label_names = findLabels(folder_names)

x_data, y_data = loadData(folder_names, label_names, voca_arr)


np.save(NP_FILE_PATH + 'x' + str(FILE_COUNT), x_data, allow_pickle=True)
np.save(NP_FILE_PATH + 'y' + str(FILE_COUNT), y_data, allow_pickle=True)
np.save(NP_FILE_PATH + 'v' + str(FILE_COUNT), voca_arr, allow_pickle=True)
np.save(NP_FILE_PATH + 'l' + str(FILE_COUNT), label_names, allow_pickle=True)
