import pandas as pd
import glob
import numpy as np
import os


def file_read(name):
    name = str(name)
    file = pd.read_csv(name, header=None, delimiter=' ')
    file = file.values
    file = file[:, :256]
    return file


# select a folder where are yours configurations
config = glob.glob('n=16/conf/*.txt')
config = sorted(config)

# range of temperatures
t = np.arange(1, 201) * 0.01

# destination of created files
dest = 'n=16_sin_cos'

try:
    os.mkdir(dest)
except OSError:
    print('This folder cannot be created')

for i in range(len(config)):
    data = file_read(config[i])
    cos = np.cos(data)
    sin = np.sin(data)
    new_data = np.concatenate((cos, sin), axis = 1).reshape((data.shape[0], 2*data.shape[1]))
    print(new_data.shape)
    np.savetxt('{}/T={}.txt'.format(dest, (i+1)*0.01), new_data, fmt='%f', delimiter=' ')


