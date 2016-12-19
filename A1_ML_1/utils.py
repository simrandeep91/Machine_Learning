import numpy as np
from numpy import genfromtxt

def sigmoid(x):
    """Computes the element wise logistic sigmoid of x.

    Inputs:
        x: Either a row vector or a column vector.
    """
    return 1.0 / (1.0 + np.exp(-x))

def load_indian():
    with open('pima_indians_diabetes.data','rb') as f:
        #train_set = np.loadtxt(f)
        train_set = genfromtxt('pima_indians_diabetes.data', delimiter=',')
        train_input = train_set[:500,:8]
        train_data = train_set[:500,8:]
    return train_input,train_data


def load_indian1():
    with open('pima_indians_diabetes.data','rb') as f:
        #train_set = np.loadtxt(f)
        train_set = genfromtxt('pima_indians_diabetes.data', delimiter=',')
        train_input = train_set[500:,:8]
        train_data = train_set[500:,8:]
    return train_input,train_data

def load_voice():
    with open('voice.csv','rb') as f:
        #train_set = np.loadtxt(f)
        #train_set = genfromtxt('voice.csv', delimiter=',')
        #train_input = train_set[1:2700,:16]
        #y = train_set[1:2700,16:]
        train_set = genfromtxt('voice.csv', delimiter=',')[:,:20]
        #print train_set
        #train_input = train_set[2700:,:20]
        train_input = train_set[1:2700,:20]
        #y = train_set[1:2700,16:]
        y = genfromtxt('voice.csv', delimiter=',',dtype=("|S10"))[1:2700,20:]
        train_data = np.zeros(y.shape);
        for i in range(0,y.shape[0]):
            if y[i,0] == '"male"':
                train_data[i,0] = 1;
            else:
                train_data[i,0] = 0;
        np.set_printoptions(threshold='nan')
        #print y
        #print train_data
    return train_input,train_data


def load_voice1():
    with open('voice.csv','rb') as f:
        #train_set = np.loadtxt(f)
        train_set = genfromtxt('voice.csv', delimiter=',')[:,:20]
        train_input = train_set[2700:,:20]
        y = genfromtxt('voice.csv', delimiter=',',dtype=("|S10"))[2700:,20:]
        #y = train_set[2700:,16:]
        train_data = np.zeros(y.shape);
        for i in range(0,y.shape[0]):
            if y[i,0] == '"male"':
                train_data[i,0] = 1;
            else:
                train_data[i,0] = 0;
    return train_input,train_data

def load_train():
    """Loads training data."""
    with open('mnist_train.npz', 'rb') as f:
        train_set = np.load(f)
        train_inputs = train_set['train_inputs']
        train_targets = train_set['train_targets']
    return train_inputs, train_targets 

def load_train_small():
    """Loads small training data."""
    with open('mnist_train_small.npz', 'rb') as f:
        train_set_small = np.load(f)
        train_inputs_small = train_set_small['train_inputs_small']
        train_targets_small = train_set_small['train_targets_small']
    return train_inputs_small, train_targets_small

def load_valid():
    """Loads validation data."""
    with open('mnist_valid.npz', 'rb') as f:
        valid_set = np.load(f)
        valid_inputs = valid_set['valid_inputs']
        valid_targets = valid_set['valid_targets']
    
    return valid_inputs, valid_targets 

def load_test():
    """Loads test data."""
    with open('mnist_test.npz', 'rb') as f:
        test_set = np.load(f)
        test_inputs = test_set['test_inputs']
        test_targets = test_set['test_targets']

    return test_inputs, test_targets 
