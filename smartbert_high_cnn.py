import tensorflow as tf
from tensorflow import keras
from highlight import compute
import dataset as db
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
argv = sys.argv[1:]


DIM = 768
VOC = 50264
PAD = 0.0
BATCH = 100
BATCH_SIZE = 100
EPOCH = 100
MAX_SEQ = 256
DROP = 0.5  # best is 0.5

DIST = 0.015
SCALE = 2

MODEL_PATH = './models/smartbert_high_cnn'


def buildModel():
    inputs = keras.layers.Input((MAX_SEQ, DIM))
    conv1 = keras.layers.Conv1D(
        filters=DIM, kernel_size=3, padding='same', activation='relu')(inputs)
    pool1 = keras.layers.MaxPool1D(pool_size=(8))(conv1)
    conv2 = keras.layers.Conv1D(
        filters=DIM, kernel_size=4, padding='same', activation='relu')(inputs)
    pool2 = keras.layers.MaxPool1D(pool_size=(8))(conv2)
    conv3 = keras.layers.Conv1D(
        filters=DIM, kernel_size=5, padding='same', activation='relu')(inputs)
    pool3 = keras.layers.MaxPool1D(pool_size=(8))(conv3)
    concat = keras.layers.Concatenate(axis=-1)([pool1, pool2, pool3])
    flat = keras.layers.Flatten()(concat)
    drop = keras.layers.Dropout(DROP)(flat)
    outputs = keras.layers.Dense(10, activation='sigmoid')(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)
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


def highlight(xs):
    arr = []
    for x in xs:
        if (compute(x) >= DIST):
            x = [SCALE*i for i in x]
        arr.append(x)
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
            xs.append(highlight(data['x']))
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
