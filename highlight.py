import json
import dataset as db
from scipy.spatial import distance
import sys
argv = sys.argv[1:]

path = './models/kmeans-model.json'

with open(path) as fp:
    data = json.load(fp)
    centroids = data['centroids']


def compute(vec):
    min = 1
    for c in centroids:
        dis = distance.cosine(c, vec)
        if (dis < min):
            min = dis
    return min


def predict(id):
    data = db.getXY2(id)
    xs = data['x']
    pt = []
    for x in xs:
        pt.append(compute(x))
    pt.sort(reverse=True)
    print(pt)


if (argv[0] == 'predict'):
    predict(int(argv[1]))
