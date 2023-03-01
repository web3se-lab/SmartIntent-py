import tensorflow as tf
from tensorflow import keras
import dataset as db
import sys
import config
import os
argv = sys.argv[1:]
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"


DIM = 768
VOC = 50264
PAD = 1
PAD_TKN = 512
BATCH = 100
BATCH_SIZE = 100
EPOCH = 100
MAX_SEQ = 256
DROP = 0.5  # best is 0.5

MODEL_PATH = './models/cnn'


def buildModel():
    inputs = keras.layers.Input((MAX_SEQ, PAD_TKN))
    embedding = keras.layers.Embedding(
        input_dim=VOC, output_dim=DIM)(inputs)
    conv1 = keras.layers.Conv2D(
        filters=DIM, kernel_size=3, padding='same', activation='relu')(embedding)
    pool1 = keras.layers.MaxPool2D(pool_size=(8, 8))(conv1)
    conv2 = keras.layers.Conv2D(
        filters=DIM, kernel_size=4, padding='same', activation='relu')(embedding)
    pool2 = keras.layers.MaxPool2D(pool_size=(8, 8))(conv2)
    conv3 = keras.layers.Conv2D(
        filters=DIM, kernel_size=5, padding='same', activation='relu')(embedding)
    pool3 = keras.layers.MaxPool2D(pool_size=(8, 8))(conv3)
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
            x.append([1]*PAD_TKN)
        arr.append(x[:MAX_SEQ])
    return arr


def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    gpu = config.multi_gpu()
    with gpu.scope():
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
    print("GPU:", gpu.num_replicas_in_sync)
    while (batch > 0):
        print("Current Batch:", batch)
        print("Current Id:", id)
        xs = []
        ys = []
        while (len(xs) < batch_size):
            data = db.getXY(id)
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
