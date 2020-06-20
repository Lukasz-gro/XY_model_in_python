from keras.models import model_from_json
import pandas as pd
import numpy as np
import glob


def file_read(name):
    name = str(name)
    file = pd.read_csv(name, header=None, delimiter=' ')
    file = file.values
    file = file[:, :512]
    return file

# loading the model to calculate probability
json_file = open('trained_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("trained_model/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# end of loading


# enter the path of the files
config = glob.glob('n=16_sin_cos/*.txt')
config = sorted(config)
print(config)

list_of_probability = []

for i in range(len(config)):
    data = file_read(config[i])
    prediction = loaded_model.predict(data)
    print(prediction.shape, i)
    list_of_probability.append(np.mean(prediction))
    print(np.mean(prediction))

file_list = np.asarray(list_of_probability).reshape(len(config), 1)
np.savetxt('prediction.txt', file_list)
