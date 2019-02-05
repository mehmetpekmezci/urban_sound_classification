import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from generate_signals import *
from load_dataset import *
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
from compute_cost import *
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

def model(X_train, Y_train, X_test, Y_test, percent_training, learning_rate = 0.0001, num_epochs = 500, minibatch_size = 32, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(0)                            
    seed = 3                                         
    (n_x, m) = X_train.shape                          #n_x: input size. m: number of examples in the train set.
    n_y = n_x							
    costs = []                                        # To keep track of the cost

    m_train = m
    m_test = X_test.shape[1]
    
    '''Create placeholders for X, Y, and the dropout probability keep_prob'''
    X, Y, keep_prob = create_placeholders(n_x, n_y)

    '''Initialize parameters for problem, i.e., the W's and b's '''
    parameters = initialize_parameters(n_x)

    '''Implement forward propagation'''
    Z_L = forward_propagation(X, keep_prob, parameters)

    '''Compute cost function, which is a function of final layer and supposed output.'''
    cost = compute_cost(Z_L, Y)

    '''Pick optimizer.'''
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    '''Initialize global variables'''
    init = tf.global_variables_initializer()


    '''Begin Tensorflow Session.'''
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1

            '''Implement Minibatch Gradient Descent'''
            for i in range(num_minibatches):
                minibatch_X = X_train[:,i*minibatch_size:(i+1)*minibatch_size]
                minibatch_Y = Y_train[:,i*minibatch_size:(i+1)*minibatch_size]

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: .9})

                epoch_cost += minibatch_cost/num_minibatches

            '''Print out cost function every 50 epochs, and append cost function to vector (for use in plotting) every 5 epochs.'''
            if print_cost == True and epoch % 50 == 0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if print_cost == True and epoch % 10 == 0:
                costs.append(epoch_cost)
        
        '''Plot the cost.'''
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(np.squeeze(costs))
        plt.ylabel('Cost Function')
        plt.xlabel('Iterations (per tens)')
        plt.show()

        '''Save the parameters in a variable.'''
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        '''Test accuracy on training and test data'''
        accuracy = tf.reduce_mean(tf.norm(tf.cast(Z_L-Y, 'float'))) #removed tf.reduce_mean()

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob:1.0})/(percent_training*m_train))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob:1.0})/((1-percent_training)*m_test))

        return parameters