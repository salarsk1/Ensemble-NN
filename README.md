# Ensemble-NN
This is the code for ensemble neural networks for eliminating multi-colinearity issues.
# Runing the code for test cases
To run the code for test cases, switch to the test repository, and run each test case. The results will be stored in the test directory.
# Running the code for new data set
- Creat a list of neural network models and store them in the "models" list.
- Store your "n_total" number of data set with dimensionality "d" in "x_total" and "y_total" arrays.
- Creat an observaton set of "n_data" point, "x" and "y" out of and store them in numpy arrays.
- Creat a test set of "n_test" points, "x_test" and "y_test" and store them in numpy arrays.
- Set number of epochs "n_epochs", for BFGS optimizaiton.
- Set L2 regularization coefficient "lam_reg", for the optimization of loss function.
- Set "beta_low" and "beta_up" for stopping condition in the blending stage.

# Requirements
tensorflow 2.4 and tensorflow_probability

To install requirements

pip install -r requirements.txt
