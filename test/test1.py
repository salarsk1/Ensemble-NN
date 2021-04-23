import sys
sys.path.insert(0, '../')
from src import *
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(12345)

d = 1
n_total = 1000
n_data  = 38
x_total = np.linspace(-4, 4, n_total).reshape(-1, 1)
y_total  = x_total * np.sin(x_total**2)
indices_train = np.array(np.arange(0, 1000, 27), dtype='int')
set_total = set(range(0, n_total))
set_train = set(indices_train)
indices_test  = np.array(sorted(list(set_total.difference(set_train))))
x = x_total[indices_train]
y =  y_total[indices_train]
y = tf.constant(y, shape=(n_data, 1), dtype = 'float32')

n_test = n_total - n_data
x_test = x_total[indices_test]
y_test = y_total[indices_test]
y_test = tf.constant(y_test, shape=(n_test, 1), dtype = 'float32')
networks = [(9, "tanh"), (11, "sigmoid"), (11, "softplus"), (9, "tanh"),
           (11, "sigmoid"), (12, "sigmoid")]
models = [Linear([networks[i][0]], 1, activation = networks[i][1]) for i in range(len(networks))]

n_epochs = 20_000
lam_reg = 0.002
model = Ensemble(models, n_data, n_total, x, y, 
                 x_test, y_test, x_total, y_total,
                 d, n_epochs, lam_reg, 0.0, 0.99)
model.ensemble()
model.save_data("../results/test1/", networks)
model.plot_aggregate("../results/test1/")
