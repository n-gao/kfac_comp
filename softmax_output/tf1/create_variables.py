
import numpy as np
import pickle as pk
from utils import save_pk

np.random.seed(123)
n_dim_in = 10
n_dim_out = 5
n_dists = 5
n_points = 69
locs = np.random.uniform(-1, 1, size=(n_dim_out,))

labels = []
points = []
for i in range(n_points):
    dist = np.random.choice(n_dists, 1)
    mean = locs[dist]
    data = np.random.normal(mean, scale=0.1, size=(1, n_dim_in))
    points.append(data)
    labels.append(dist)

data = np.concatenate(points, axis=0)
labels = np.concatenate(labels, axis=0)

weights = np.random.uniform(-1., 1., (n_dim_in, n_dim_out))

save_pk(weights, 'weights.pk')
save_pk(data, 'data.pk')
save_pk(labels, 'labels.pk')