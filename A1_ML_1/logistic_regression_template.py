import matplotlib.pyplot as plt
from check_grad import check_grad
from utils import *
from logistic import *
from plot_digits import show_pane


def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_voice()

    valid_inputs, valid_targets = load_voice1()

    #PASS test_inputs AND test_targets IN PLACE OF VALIDATION DATASET TO RUN LOGISTIC REGRESSION ON TEST DATASET.
    #test_inputs, test_targets = load_test()
    
    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape
    # Logistic regression weights
    # TODO:Initialize to random weights here.

    #Seed same value
    #np.random.seed(1)

    weights = np.random.randn(M+1,1)
    weights = weights * 0.1
    print weights
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    iterations = np.zeros((hyperparameters['num_iterations']))
    ce_training = np.zeros((hyperparameters['num_iterations']))
    ce_validation = np.zeros((hyperparameters['num_iterations']))
    f_value = np.zeros((hyperparameters['num_iterations']))
    for t in xrange(hyperparameters['num_iterations']): 

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        #
        #For TEST DATASET, pass test_inputs instead of valid_inputs and test_targets instead of valid_targets.
        #

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)
        np.set_printoptions(threshold='nan')
        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
        iterations[t] = t;
        ce_training[t] = cross_entropy_train;
        ce_validation[t] = cross_entropy_valid;
        f_value[t] = f;
    plt.figure(1)
    plt.plot(iterations, ce_training, marker='+', label="Training Set")
    plt.plot(iterations, ce_validation, marker='o', label="Validation Set")
    plt.legend(loc=2)
    if(hyperparameters['weight_regularization'] == False):
        plt.title('Logistic Regression - mnist_train')
    else:
        plt.title('Regularized Logistic Regression - mnist_train')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy')
    plt.show()
    return logging


def run_logistic_regression_small(hyperparameters):
    # TODO specify training data
    train_inputs_1, train_targets_1 = load_train_small()

    valid_inputs, valid_targets = load_valid()
    
    #PASS test_inputs AND test_targets IN PLACE OF VALIDATION DATASET TO RUN LOGISTIC REGRESSION ON TEST DATASET.
    test_inputs, test_targets = load_test()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs_1.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    
    #Seed same value
    #np.random.seed(1)
    
    weights = np.random.randn(M+1,1)
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    iterations = np.zeros((hyperparameters['num_iterations']))
    ce_training_1 = np.zeros((hyperparameters['num_iterations']))
    ce_validation = np.zeros((hyperparameters['num_iterations']))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs_1, train_targets_1, hyperparameters)
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets_1, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N
        
        #
        #For TEST DATASET, pass test_inputs instead of valid_inputs and test_targets instead of valid_targets.
        #
        
        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)
        np.set_printoptions(threshold='nan')
        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
        iterations[t] = t;
        ce_training_1[t] = cross_entropy_train;
        ce_validation[t] = cross_entropy_valid;
    plt.figure(1)
    plt.plot(iterations, ce_training_1, marker='+', label="Training Set")
    plt.plot(iterations, ce_validation, marker='o', label="Validation Set")
    plt.legend(loc=2)
    if(hyperparameters['weight_regularization'] == False):
        plt.title('Logistic Regression - mnist_train_small')
    else:
        plt.title('Regularized Logistic Regression - mnist_train_small')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy')
    plt.show()
    return logging

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.12,
                    'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 500,
                    'weight_decay': 0.1 # related to standard deviation of weight prior 
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots