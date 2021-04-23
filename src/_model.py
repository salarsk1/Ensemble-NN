import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import tensorflow_probability as tfp
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

__all__ = ["Linear", "Utility", "buildFunctions"]

class Linear(tf.keras.Model):
    def __init__(self, hidden_layers, output_size, activation = 'sigmoid'):

        super(Linear, self).__init__()
        self.hidden_layers = hidden_layers
        self.layers_list = []
        self.n_layers = len(hidden_layers)
        for i in range(self.n_layers):
            self.layers_list.append(keras.layers.Dense(hidden_layers[i], activation = activation))
        self.layers_list.append(keras.layers.Dense(output_size))

    def call(self, input_tensor, training = True):
        x = self.layers_list[0](input_tensor)
        for i in range(1, self.n_layers):
            x = self.layers_list[i](x)
        return self.layers_list[self.n_layers](x)
class Utility(object):

    def __init__(self):
        pass

    def custom_loss(self, M_k, lam):
        def loss(y_obs, y_pred):
            zb = self.zero_bias(y_pred, y_obs)
            mf_zb = self.misfit(zb, y_obs)
            mf = self.misfit(y_pred, y_obs)
            # constr = tf.maximum(tf.reduce_mean(tf.multiply(mf, M_k)), 0.0)
            constr = tf.square(tf.reduce_mean(tf.multiply(mf_zb, M_k)))
            # logit = [tf.reduce_mean(30.0 * tf.multiply(mf, M_k)), tf.constant(0.0)]
            # constr = tf.reduce_max(tf.nn.softmax(logit))
            # constr = tf.math.log(tf.math.exp(tf.reduce_mean(tf.multiply(mf, M_k))) + tf.math.exp(0.0))
            return tf.reduce_mean(tf.square(mf)) - tf.square(tf.reduce_mean(mf)) + lam * constr
        return loss

    def misfit(self, y_pred, y_obs):
        y_pred = tf.reshape(y_pred, [-1, 1])
        y_obs  = tf.reshape(y_obs , [-1, 1])
        return y_pred - y_obs

    def zero_bias(self, y_pred, y_obs):
        y_pred = tf.reshape(y_pred, [-1, 1])
        y_obs  = tf.reshape(y_obs , [-1, 1])
        return y_pred - tf.reduce_mean(y_pred) + tf.reduce_mean(y_obs)

    def shift_model(self, y_pred, y_pred_train, y_obs):
        return y_pred - tf.reduce_mean(y_pred_train) + tf.reduce_mean(y_obs)

    def get_beta(self, y_pred, y_obs, M_k):
        # zero_bias_nn = self.zero_bias(y_pred, y_obs)
        mf    = self.misfit(y_pred, y_obs)
        term1 = tf.reduce_mean(tf.square(mf))
        term2 = tf.reduce_mean(tf.multiply(M_k, mf))
        term3 = tf.reduce_mean(tf.square(M_k - mf))
        return (term1 - term2) / term3

    def cum_misfit(self, y_pred, y_obs, beta, M_k):
        # zero_bias_nn = self.zero_bias(y_pred, y_obs)
        mf = self.misfit(y_pred, y_obs)
        return beta * M_k + (1.0 - beta) * mf

    def corr(self, y_pred, y_obs, M_k):
        mf = self.misfit(y_pred, y_obs)
        return tf.reduce_mean(tf.multiply(mf, M_k))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

"""The function "buildFunctions" is taken from the code
   written by Pi-Yueh Chuang <pychuang@gwu.edu>
   https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
"""
def buildFunctions(model, loss, train_x, train_y, lam):

    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    count = 0
    idx = []
    part = []

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    @tf.function
    def f(params_1d):

        with tf.GradientTape() as tape:
            assign_new_model_parameters(params_1d)
            loss_value = loss(model(train_x, training=True), train_y) + lam * tf.reduce_mean(tf.square(params_1d))

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        f.iter.assign_add(1)
        if f.iter % 50000 == 0:
            tf.print("Iter:", f.iter, "loss:", loss_value)

        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    return f