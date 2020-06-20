# This file adds constant value to all angles in given configuration. It provides diversity in our training data.

import pandas as pd
import numpy as np
import glob


def file_read(name):
    name = str(name)
    file = pd.read_csv(name, header=None, delimiter=' ')
    file = file.values
    file = file[:, :256]
    return file


config = glob.glob('n=16/conf/*.txt')
config = sorted(config)


base = np.ones(256)

for i in range(len(config)):
    print(i)
    data = file_read(config[i])
    name = config[i]

    for j in range(data.shape[0]):
        diff = np.random.uniform(0, 2 * np.pi)
        const = diff * base
        data[j] += const
        for k in range(len(data[j])):
            if data[j][k] > 2 * np.pi:
                data[j][k] -= 2 * np.pi
    np.savetxt(name, data, fmt='%f', delimiter=' ')
