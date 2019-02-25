import tensorflow as tf
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator


def generate_signals(num_X = 10, max_bandwidth = 10, max_num_sinusoids = 100):
	np.random.seed(1)
	tt = np.linspace(0, 1, 100) ## 0 ile 1 arasinda 100 sayi, 0.01, 0.02 .... 0.99
	tt_length = tt.shape[0]
	X = []
	Y = []
	for i in range(num_X):
		num_sinusoids = np.random.randint(1, max_num_sinusoids)
		freqs = []
		signal = 0
		for j in range(num_sinusoids):
			rand_freq = max_bandwidth * np.random.rand()
			freqs.append(rand_freq)
			rand_offset = 10*np.random.randn()
			sine_or_cosine = np.random.randint(2) #1/2 chance for sine, 1/2 chance for cosine
			noise_or_nonoise = np.random.randint(20) #1/20 of the signals have noise
			if sine_or_cosine == 1:
				signal += np.sin(2 * np.pi * rand_freq * tt)
			else:
				signal += np.cos(2 * np.pi * rand_freq * tt)
			if noise_or_nonoise == 1:
				signal += np.random.randn(tt_length)
		X = np.hstack((X,signal))
	X = X.reshape(num_X, tt_length)
	data = dict()
	data['t'] = tt 
	data['X'] = X
	data['FFT_X'] = np.real(sc.linalg.dft(tt_length).dot(X.T).T)
	return data
	


def load_dataset(data):
	np.random.seed(1) #maintain consistent results
	X = data['X']
	Y = data['FFT_X']
	t = data['t']
	indices = np.random.permutation(Y.shape[0])
	percent_training = .9
	num_training = int(percent_training * Y.shape[0])
	training_idx, test_idx = indices[:num_training], indices[num_training:]
	X_train = X[training_idx,:].T
	Y_train = Y[training_idx,:].T
	X_test = X[test_idx, :].T
	Y_test = Y[test_idx, :].T
	return (X_train, Y_train, X_test, Y_test, t, percent_training)


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
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    keep_prob = tf.placeholder(tf.float32) #used in dropout
    '''Initialize parameters for problem, i.e., the W's and b's '''
    parameters = initialize_parameters(n_x)
    tf.set_random_seed(0)
    W1 = tf.get_variable("W1", [int(len_example//6),len_example], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1 = tf.get_variable("b1", [int(len_example//6),1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [int(len_example//6),int(len_example//6)], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b2 = tf.get_variable("b2", [int(len_example//6),1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [len_example,int(len_example//6)], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b3 = tf.get_variable("b3", [len_example,1], initializer = tf.zeros_initializer())	
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
    '''Implement forward propagation'''
    Z1 = tf.add(tf.matmul(W1, X),b1)
    A1 = tf.nn.dropout(Z1, keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.dropout(Z2, keep_prob)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    Z_L=Z3
    '''Compute cost function, which is a function of final layer and supposed output.'''
    cost = tf.norm(Z_L, Y)

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



#######################################
##### MAIN
#######################################

'''
Load Data
X has dimensions (time indices, number of examples)
Y has dimensions (number of examples,)
'''
num_examples = 30000
max_bandwidth = 10
max_num_sinusoids = 10

SavingDataToFile = False

if SavingDataToFile:
	print('Making dataset.')
	data = generate_signals(num_examples, max_bandwidth, max_num_sinusoids)
	data_filename = '/tmp/DFT_data_m_%s_BW_%s_maxSinusoids_%s'  % (num_examples, max_bandwidth, max_num_sinusoids)
	np.save(data_filename, data)
else:
	data = np.load('/tmp/DFT_data_m_30000_BW_10_maxSinusoids_10.npy').item()
(X_train, Y_train, X_test, Y_test, t, percent_training) = load_dataset(data)

learning_rate = 0.001
num_epochs = 20000
minibatch_size = 250
print_cost = True

parameters = model(X_train, Y_train, X_test, Y_test, percent_training, learning_rate, num_epochs, minibatch_size, print_cost)


'''
Timing
'''

num_examples = X_test.shape[1]


t_naive_DFT_start = time.time()
DFT_X_REAL = np.real(sc.linalg.dft(100).dot(X_test))
t_naive_DFT = time.time() - t_naive_DFT_start


t_fft_start = time.time()
FFT_X_REAL = np.real(np.fft.fft(X_test))
t_fft = time.time() - t_fft_start


W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']

t_nn_start = time.time()

Z1 = np.dot(W1, X_test) + b1
Z2 = np.dot(W2, Z1) + b2
Z3 = np.dot(W3, Z2) + b3

t_nn_DFT = time.time() - t_nn_start


rand_idx = np.random.randint(num_examples)


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

Z_L = np.fft.fftshift(Z3[:,rand_idx])
Y = np.fft.fftshift(DFT_X_REAL[:, rand_idx])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

f, (ax1, ax2) = plt.subplots(2,1,sharey=True)

ax1.plot(range(-50, 50), Z_L, color='black', label='Neural network estimate')
ax2.plot(range(-50, 50), Y, color='blue', label='Ground truth')

plt.xlabel(r'DFT Index $m$')
ax1.set_ylabel(r'Re$\{\hat F\{f\}[m]\}$')
ax2.set_ylabel(r'Re$\{F\{f\}[m]\}$')
ax1.legend()
ax2.legend()

plt.savefig('DFT_comparisons')

plt.close()

similarity = np.linalg.norm(Z_L - Y) / np.linalg.norm(Y)

print('||Z_L - Y||/||Y|| = %s.' % similarity)

print('Time for naive computation = %s seconds.' % (t_naive_DFT/num_examples))
print('Time for FFT computation = %s seconds.' % (t_fft/num_examples))
print('Time for NN computation = %s seconds.' % (t_nn_DFT/num_examples))




        

