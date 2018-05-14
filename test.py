import numpy as np
from development import train_softmax
from input.input_feature import *

dataset = AudioDataset(files_path='voiceprint_training/voiceprint_training/info.csv',
                       audio_dir='voiceprint_training/voiceprint_training/data',
                      transform=Compose([CMVN(),
                                         Feature_Cube(cube_shape=(20, 80, 40),
                                                      augmentation=True), ToOutput()]))
x,y =dataset.getData(1000)
data = {}
data['x_train'] = x[:900]
data['x_test'] = x[901:]
data['y_train'] = y[:900]
data['y_test'] = y[901:]
train_softmax.init(data, epochs = 2)