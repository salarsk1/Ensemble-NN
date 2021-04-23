import sys
sys.path.insert(0, '../')
from src import *
import csv
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(12345)
random.seed(12345)
d = 4
nx = 15
n_total = int(nx ** 4)
n_data  = 2_000
lx = 1.5
x1 = np.linspace(-lx, lx, nx)
x2 = np.linspace(-lx, lx, nx)
x3 = np.linspace(-lx, lx, nx)
x4 = np.linspace(-lx, lx, nx)
X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4)
x_total = np.stack([X1, X2, X3, X4], axis=4).reshape(-1,d)
y_total = 10.0 * d
for i in range(d):
    y_total += x_total[:,i]**2 - 10.0 * np.cos(2.0 * np.pi * x_total[:,i])
indices_train = np.sort(random.sample(range(0, n_total), n_data))
set_total = set(range(0, n_total))
set_train = set(indices_train)
indices_test  = np.array(sorted(list(set_total.difference(set_train))))
x = x_total[indices_train]
y = y_total[indices_train]
y = tf.constant(y, shape=(n_data, 1), dtype = 'float32')

n_test = n_total - n_data
x_test = x_total[indices_test]
y_test = y_total[indices_test]
y_test = tf.constant(y_test, shape=(n_test, 1), dtype = 'float32')

networks = [(38, "sigmoid"), (38, "tanh"), (37, "sigmoid"), (37, "sigmoid"),
			(39, "sigmoid"), (39, "tanh"), (40, "sigmoid"), (40, "tanh"), 
			(41, "sigmoid"), (41, "tanh")]

models = [Linear([networks[i][0]], 1, activation = networks[i][1]) for i in range(len(networks))]
n_epochs = 20_000
lam_reg = 0.05
model = Ensemble(models, n_data, n_total, x, y, 
                x_test, y_test, x_total, y_total,
                d, n_epochs, lam_reg, 0.0, 0.99)
model.ensemble()
model.save_data("../results/test3/", networks)
