from check_grad import check_grad
from logistic_regression_template import *
from utils import *
from logistic import *
from plot_digits import show_pane

#SCRIPT TO RUN LOGISTIC REGRESSION

#Hyperparameters for Logistic Regression
hyperparameters = {
                'learning_rate': 0.15,
                'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                'num_iterations': 350,
                'weight_decay': 0 # related to standard deviation of weight prior 
                }

#Run Logistic Regression on mnist_train
num_runs = 1
logging = np.zeros((hyperparameters['num_iterations'], 5))
for i in xrange(num_runs):
    logging += run_logistic_regression(hyperparameters)
logging /= num_runs

#Run Logistic Regression on mnist_train_small
#num_runs1 = 1
#logging1 = np.zeros((hyperparameters['num_iterations'], 5))
#for i in xrange(num_runs1):
#    logging1 += run_logistic_regression_small(hyperparameters)
#logging1 /= num_runs1


#Hyperparameters for Regularized Logistic Regression
hyperparameters_r = {
                'learning_rate': 0.13,
                'weight_regularization': True, # boolean, True for using Gaussian prior on weights
                'num_iterations': 320,
                'weight_decay': 0.1 # related to standard deviation of weight prior 
                }

#Run Regularized Logistic Regression on mnist_train
num_runs_r = 1
logging_r = np.zeros((hyperparameters_r['num_iterations'], 5))
for i in xrange(num_runs_r):
    logging_r += run_logistic_regression(hyperparameters_r)
logging_r /= num_runs_r

#Run Regularized Logistic Regression on mnist_train_small
#num_runs_r1 = 1
#logging_r1 = np.zeros((hyperparameters_r['num_iterations'], 5))
#for i in xrange(num_runs_r1):
#    logging_r1 += run_logistic_regression_small(hyperparameters_r)
#logging_r1 /= num_runs_r1
