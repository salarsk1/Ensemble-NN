import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_probability as tfp

from ._model import *
import csv
import os

__all__ = ["Ensemble"]

class Ensemble(object):
    def __init__(self, models, n_data, n_total, x, y, x_test, y_test, x_total, y_total,
                 d, n_epochs, lam_reg, beta_low=0.0, beta_up=0.95):
        self.models = models
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.x_total = x_total
        self.y_total = y_total
        self.d = d
        self.n_epochs = n_epochs
        self.lam_reg = lam_reg
        self.beta_low = beta_low
        self.beta_up = beta_up
        self.betas = [0.0]
        self.M_ks  = []
        self.agg_train = []
        self.M_ks_test = []
        self.train_errors = []
        self.agg_biases = []
        self.n_networks = len(models)
        self.n_data = n_data
        self.n_total = n_total
        self.n_test = self.n_total - self.n_data

    def ensemble(self):
        util = Utility()
        print("Training network number {}".format(1))
        self.models[0].build((None, self.d))

        loss_fun = util.custom_loss(tf.zeros(self.y.shape), 0)
        func = buildFunctions(self.models[0], loss_fun, self.x, self.y, self.lam_reg)
        init_params = tf.dynamic_stitch(func.idx, self.models[0].trainable_variables)
        nodes = self.models[0].hidden_layers[0]
        init_params = tf.zeros(init_params.shape)
        params = np.random.uniform(-1., 1.,size=init_params.shape)
        init_params += params

        def cond(t1, t2):
          return False

        results = tfp.optimizer.bfgs_minimize(value_and_gradients_function = func, 
                                               initial_position = init_params, 
                                               max_iterations = self.n_epochs, 
                                               stopping_condition=None)
        for c in range(20):
            init_params = results.position + np.random.uniform(1, 2, size=init_params.shape)*\
                          np.random.randint(-1, 1, init_params.shape) * 0.05

            results = tfp.optimizer.bfgs_minimize(value_and_gradients_function = func, 
                                                   initial_position = init_params, 
                                                   max_iterations = self.n_epochs, 
                                                   stopping_condition=None)
            func.assign_new_model_parameters(results.position)
        func.assign_new_model_parameters(results.position)
        
        zero_bias_nn = util.zero_bias(self.models[0].predict(self.x), self.y)
        pred_test = util.shift_model(self.models[0].predict(self.x_test), self.models[0].predict(self.x), self.y)
        print('MSE of model 1 on training set: {}'.format(tf.reduce_mean(tf.square(zero_bias_nn - self.y))))
        print('MSE of model 1 on test set: {}'.format(tf.reduce_mean(tf.square(pred_test - self.y_test))))
        print('BIAS AG on test: {}'.format(tf.reduce_mean(pred_test - self.y_test)))
        self.M_ks.append(util.misfit(zero_bias_nn, self.y))
        self.agg_train.append(tf.reduce_mean(tf.square(self.M_ks[-1])).numpy())
        self.M_ks_test.append(tf.reduce_mean(tf.square(pred_test - self.y_test)).numpy())
        self.train_errors.append(tf.reduce_mean(tf.square(zero_bias_nn - self.y)).numpy())
        self.agg_biases.append(tf.reduce_mean(pred_test - self.y_test).numpy())
        print("******************************************************************************")
        for i in range(1, self.n_networks):
            lam = 4.0

            print("Training network number {}".format(i+1))
            counter = 1
            self.models[i].build((None, self.d))
            while(True):

                loss_fun = util.custom_loss(self.M_ks[-1], lam)
                func = buildFunctions(self.models[i], loss_fun, self.x, self.y, self.lam_reg)
                init_params = tf.dynamic_stitch(func.idx, self.models[i].trainable_variables)
                nodes = self.models[i].hidden_layers[0]
                init_params = tf.zeros(init_params.shape)
                init_params += np.random.uniform(-1, 1, size=init_params.shape)
                results = tfp.optimizer.bfgs_minimize(value_and_gradients_function = func, 
                                                      initial_position = init_params, 
                                                      max_iterations = self.n_epochs)
                for c in range(20):
                    temp = np.random.randint(-1, 1, init_params.shape) * 0.0005
                    init_params = results.position + np.random.uniform(1, 2, size=init_params.shape)*temp
                    results = tfp.optimizer.bfgs_minimize(value_and_gradients_function = func, 
                                                          initial_position = init_params, 
                                                          max_iterations = self.n_epochs)
                    func.assign_new_model_parameters(results.position)
                func.assign_new_model_parameters(results.position)
                zero_bias_nn = util.zero_bias(self.models[i].predict(self.x), self.y)
                check = util.get_beta(zero_bias_nn, self.y, self.M_ks[-1]).numpy()
                if check >= self.beta_low  and check <= self.beta_up:
                    print('MSE of model {} on training set: {}'.format\
                         (i+1, tf.reduce_mean(tf.square(zero_bias_nn - self.y))))
                    pred_test = util.shift_model(self.models[i].predict(self.x_test), \
                                                 self.models[i].predict(self.x), self.y)
                    print('MSE of model {} on test set: {}'.format\
                         (i+1, tf.reduce_mean(tf.square(pred_test - self.y_test))))
                    self.betas.append(util.get_beta(zero_bias_nn, self.y, self.M_ks[-1]).numpy())
                    temp = util.cum_misfit(zero_bias_nn, self.y, self.betas[-1], self.M_ks[-1])
                    self.M_ks.append(temp)
                    print("MSE of aggregated model on training set up to network {}: {}".format\
                         (i+1, tf.reduce_mean(tf.square(self.M_ks[-1]))))
                    alphas = []
                    alphas = [(1 - self.betas[p]) * np.prod(self.betas[p+1:]) for p in range(i)]
                    alphas.append(1 - self.betas[-1])
                    final_model_test = tf.zeros((self.n_test, 1))
                    for p in range(i+1):
                        final_model_test = tf.add(final_model_test, alphas[p] * \
                                           util.shift_model(self.models[p].predict(self.x_test), \
                                                            self.models[p].predict(self.x), self.y))

                    print('MSE of aggregated model on test set: {}:'.format\
                         (tf.reduce_mean(tf.square(final_model_test - self.y_test))))
                    print('BIAS AG on test: {}:'.format(tf.reduce_mean(final_model_test - self.y_test)))
                    print("Correlation for model {}: {}".format\
                         (i+1, util.corr(zero_bias_nn, self.y, self.M_ks[-2])))
                    print('Lambda: {}'.format(lam))
                    print('beta {}: {}'.format(i+1, self.betas[-1]))
                    self.agg_train.append(tf.reduce_mean(tf.square(self.M_ks[-1])).numpy())
                    self.M_ks_test.append(tf.reduce_mean(tf.square(final_model_test - self.y_test)).numpy())
                    self.train_errors.append(tf.reduce_mean(tf.square(zero_bias_nn - self.y)).numpy())
                    self.agg_biases.append(tf.reduce_mean(pred_test - self.y_test).numpy())
                    break
                else:
                    lam = min(lam * 2, 1024)
                    print("Iteration: {}, lambda = {}, correlation = {}, beta = {}".format\
                         (counter+1, lam, util.corr(zero_bias_nn, self.y, self.M_ks[-1]), check))
                    counter += 1
                if counter > 9:
                    self.betas.append(1.0)
                    self.M_ks.append(self.M_ks[-1])
                    self.M_ks_test.append(0)
                    self.train_errors.append(0)
                    self.agg_train.append(0)
                    self.agg_biases.append(0)

                    print("The correlation in this model did not reach negative value.")
                    break
            print("******************************************************************************")
        # self.final_model_train = tf.zeros((self.n_data, 1))
        # for i in range(self.n_networks):
        #     self.final_model_train = tf.add(self.final_model_train, alphas[i] * \
        #                         util.shift_model(self.models[i].predict(self.x), 
        #                                          self.models[i].predict(self.x), self.y))
        self.final_model_test = final_model_test
        self.alphas = alphas

    
    def save_data(self, directory, networks):
        if not os.path.exists(directory):
            os.mkdir(directory)
        util = Utility()
        alphas = [(1 -self. betas[i]) * np.prod(self.betas[i+1:]) for i in range(self.n_networks-1)]
        alphas.append(1 - self.betas[-1])
        # print('alpha values: ', alphas)
        # print('beta values', self.betas)
        # print("agg MSE on train", self.agg_train)
        # print("agg MSE on test", self.M_ks_test)
        # print("models train error", self.train_errors)
        # print("Agg biases", self.agg_biases)
        header = ["ANN", "Nodes", "Type", "MSE", "BETA", "AG. MSE", "AG. MSE/TE", "Coeff."]
        data = zip(self.train_errors, self.betas, self.agg_train, 
                   self.M_ks_test, self.agg_biases, alphas)
        dump = []
        for i in range(len(alphas)):
            dump.append([i+1,
                         networks[i][0],
                         networks[i][1],
                         float("{:.5f}".format(self.train_errors[i])),
                         float("{:.5f}".format(self.betas[i])),
                         float("{:.5f}".format(self.agg_train[i])),
                         float("{:.5f}".format(self.M_ks_test[i])),
                         float("{:.5f}".format(alphas[i]))])
        with open(directory+"data.csv", 'w') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(header)
            writer.writerows(dump)
        

    def plot_aggregate(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        util = Utility()

        for i in range(self.n_networks):
            alphas = []
            alphas = [(1 - self.betas[p]) * np.prod(self.betas[p+1:i+1]) for p in range(i)]
            alphas.append(1 - self.betas[i])
            plt.figure()
            final_model_total = tf.zeros((self.n_total, 1))
            for j in range(i+1):
                final_model_total = tf.add(final_model_total, alphas[j] * \
                                    util.shift_model(self.models[j].predict(self.x_total), 
                                    self.models[j].predict(self.x), self.y))
            plt.plot(self.x_total, final_model_total, color = 'red', label="Aggregate # "+str(i+1), lw=2)
            plt.plot(self.x_total, self.y_total, color='green', label="True function", lw=2)
            plt.xlabel('x')
            plt.legend()
            plt.savefig(directory+"agg_"+str(i+1)+"_learned.png", dpi=300)
            plt.close()
            plt.figure()
            current = util.shift_model(self.models[i].predict(self.x_total),
                                       self.models[i].predict(self.x), self.y)
            plt.plot(self.x_total, current, color = 'blue', label="Member net. # "+str(i+1), lw=2)
            plt.plot(self.x_total, self.y_total, color='green', label="True function", lw=2)
            plt.xlabel('x')
            plt.legend()
            plt.savefig(directory +"model_"+str(i+1)+"_learned.png", dpi=300)
            plt.close()