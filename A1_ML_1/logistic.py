""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function   
    k = np.ones((data.shape[0],1))
    data_n = np.concatenate((data, k), axis=1)
    y = sigmoid(data_n.dot(weights))
    print y
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    #print (y)
    ce = - np.sum((1-targets).T.dot(np.log(y))) - np.sum((targets).T.dot(np.log(1 - y)));
    num = 0;
    for i in range(0,targets.shape[0]):
        k = (y[i,0] >= 0.5).astype(np.int)
        if targets[i,0] == 1-k:
            num = num + 1;
    
    frac_correct = (num * 1.0)/targets.size;

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        k = np.ones((data.shape[0],1))
        x= np.concatenate((data, k), axis=1)
        # TODO: compute f and df without regularization
        f = - (np.sum(targets.T.dot(np.log(1-y)))) - (np.sum((1 - targets).T.dot(np.log(y))))
        df = (x.T.dot(targets-(1-y)))
    return f, df, y

def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """
 
    # TODO: Finish this function
    y = logistic_predict(weights, data)
    k = np.ones((data.shape[0],1))
    x= np.concatenate((data, k), axis=1)
    w = np.copy(weights)
    w[data.shape[1]] = 0;
    f = - (np.sum(targets.T.dot(np.log(1-y)))) - (np.sum((1 - targets).T.dot(np.log(y)))) + np.sum((hyperparameters['weight_decay']/2)* w.T.dot(w)) - (((w.shape[0]+1)/2)*np.log(hyperparameters['weight_decay']/(2*np.pi)))
    df = (x.T.dot(targets-(1-y)))+ (hyperparameters['weight_decay']*w)
    return f, df
