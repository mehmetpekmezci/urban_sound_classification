import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

Z4 = np.load('Z4.npy')
Y = np.load('Y.npy')


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

f, (ax1, ax2) = plt.subplots(2,1,sharey=True)

ax1.plot(range(-50, 50), Z4, color='black', label='Neural network estimate')
ax2.plot(range(-50, 50), Y, color='blue', label='Ground truth')

plt.xlabel(r'DFT Index $m$')
ax1.set_ylabel(r'Re$\{\hat F\{f\}[m]\}$')
ax2.set_ylabel(r'Re$\{F\{f\}[m]\}$')
ax1.legend()
ax2.legend()

plt.savefig('DFT_comparisons')

plt.close()