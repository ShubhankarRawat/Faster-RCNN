import numpy as np

import pandas as pd

train = pd.read_csv('train.csv')
data = pd.DataFrame()
data['format'] = train['image_name']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['x1'][i]) + ',' + str(train['y1'][i]) + ',' + str(train['x2'][i]) + ',' + str(train['y2'][i]) + ',' + train['rbc'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')


import tensorflow as tf; print(tf.__version__)