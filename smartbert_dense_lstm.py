from tensorflow import keras
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
argv = sys.argv[1:]


UNIT = 128
SEQ = None
TKN = 512
DIM = 768
VOC = 50264
PAD = 0.0
BATCH = 500
BATCH_SIZE = 20
EPOCH = 50
MODEL_PATH = './models/smartbert_dense_lstm'
DROP = 0.2


def buildModel():
    model = keras.Sequential()
    model.add(keras.layers.Masking(
        mask_value=PAD, input_shape=(SEQ, TKN, DIM)))
    model.add(keras.layers.Reshape(-1))
    model.add(keras.layers.Dense(DIM))
    model.add(keras.layers.LSTM(UNIT, return_sequences=False))
    model.add(keras.layers.Dropout(DROP))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.summary()
    return model


def loadModel():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(e)
        return buildModel()


def summary():
    model = loadModel()
    model.summary()


def pad(xs):
    arr = []
    # find max length of sequence
    max = 0
    for x in xs:
        if (len(x) > max):
            max = len(x)
    # pad sequence to max
    for x in xs:
        while (len(x) < max):
            x.append([PAD]*DIM)
        arr.append(x)
    return arr




if (argv[0] == 'summary'):
    summary()
if (argv[0] == 'train'):
    train(batch=int(argv[1]), start=int(argv[2]),
          batch_size=int(argv[3]), epoch=int(argv[4]))
