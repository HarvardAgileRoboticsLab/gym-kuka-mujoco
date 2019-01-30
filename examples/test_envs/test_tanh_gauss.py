import numpy as np
import matplotlib.pyplot as plt

N = 100000
samples = np.random.normal(size=(N))
samples_1 = np.tanh(samples)
samples_2 = np.tanh(10*samples)
samples_3 = np.tanh(.1*samples)

plt.hist(samples_1, 100, label='scale = 1')
plt.hist(samples_2, 100, label='scale = 10')
plt.hist(samples_3, 100, label='scale = 0.1')
plt.legend()
plt.show()