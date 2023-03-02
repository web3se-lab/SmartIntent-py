import json
from scipy.spatial import distance

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
