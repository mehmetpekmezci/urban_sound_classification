import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from scipy.linalg import dft

rc.use('PS')

tt = np.linspace(0,5,100)

f = np.zeros(len(tt))
for i in range(5):
	f += np.cos(2*np.pi*i*tt + i)

F = dft(100).dot(f)



fig, (ax1, ax2) = plt.subplots(2,1)


'''Plot the cost.'''
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1.plot(f)
ax2.plot(np.fft.fftshift(np.real(F)))

ax1.set_xlabel(r'Discrete-time index $n$')
ax1.set_ylabel(r'$f[n]$')
ax2.set_xlabel(r'DFT index $m$')
ax2.set_ylabel(r'Re$\{F\{f\}[m]\}$')



plt.show()
plt.close()
