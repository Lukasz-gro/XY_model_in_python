import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('example_prob.txt', header=None, delimiter=' ')
y = data.values
y2 = 1 - y
Temperature = np.arange(1, 201) * 0.01

plt.plot(Temperature, y)
plt.plot(Temperature, y2)
plt.savefig('Prediction.png')