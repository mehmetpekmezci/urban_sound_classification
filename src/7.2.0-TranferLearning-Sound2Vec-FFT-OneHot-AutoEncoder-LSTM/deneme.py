from USCHeader import *

a = [[.1, .2, .3,.2,.5,.3], [.4, .8, .6,.5,.7,.4]]
x_data = np.array(a)
for i in range(x_data.shape[0]): #batch_size
       x_data_peak_points=scipy.signal.find_peaks(x_data[i])
       x_data[i]=np.zeros([x_data.shape[1]],np.float32)
       x_data[i][x_data_peak_points[0]]=1


print(x_data)
