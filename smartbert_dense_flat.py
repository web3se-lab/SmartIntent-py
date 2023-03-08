import os
import tensorflow as tf
from tensorflow import keras
import dataset as db
import sys
argv = sys.argv[1:]

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


DIM = 768
PAD = 0.0
BATCH = 100
BATCH_SIZE = 100
EPOCH = 100
MAX_SEQ = 256
DROP = 0.5  # best is 0.5

MODEL_PATH = './models/smartbert_dense'


def buildModel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        input_shape=(MAX_SEQ, DIM), units=768, activation='relu'))
    model.add(keras.layers.Dropout(DROP))
    model.add(keras.layers.Dense(units=768, activation='relu'))
    # model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dropout(DROP))
    model.add(keras.layers.Flatten())
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
    for x in xs:
        while (len(x) < MAX_SEQ):
            x.append([PAD]*DIM)
        arr.append(x[:MAX_SEQ])
    return arr


def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])
    id = start
    print("Batch:", batch)
    print("Batch Size:", batch_size)
    print("Total:", batch*batch_size)
    while (batch > 0):
        xs = []
        ys = []
        print("Current Batch:", batch)
        print("Current Id:", id)
        while (len(xs) < batch_size):
            data = db.getXY2(id)
            id = id+1
            if (data == None):
                continue
            xs.append(data['x'])
            ys.append(data['y'])
        tx = tf.convert_to_tensor(pad(xs))
        ty = tf.convert_to_tensor(ys)
        print(tx)
        print(ty)
        model.fit(tx, ty, batch_size=batch_size,
                  epochs=epoch, shuffle=True)
        model.save(MODEL_PATH)
        batch = batch-1


if (argv[0] == 'summary'):
    summary()
if (argv[0] == 'train'):
    train(batch=int(argv[1]), start=int(argv[2]),
          batch_size=int(argv[3]), epoch=int(argv[4]))
