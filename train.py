import os

import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print('Loading training data...')
e0 = cv2.getTickCount()

# load training data
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('./training_data/*.npz') #在本工程目录中找到training_data目录并读取里面的所有npz文件

# if no data, exit
if not training_data:
    print("No training data in directory, exit")
    sys.exit()

for single_npz in training_data:
    with np.load(single_npz) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :]
y = label_array[1:, :]
print('Image array shape: ', X.shape)
print('Label array shape: ', y.shape)

e00 = cv2.getTickCount()
time0 = (e00 - e0)/ cv2.getTickFrequency()
print('Loading image duration:', time0)

# train test split, 8:2
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

# set start time
e1 = cv2.getTickCount()

# create MLP

model = cv2.ml.ANN_MLP_create()
# Create ANN layer
model.setLayerSizes(np.int32([38400,64,3]))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
model.setTermCriteria(( cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.001))
# BP Method
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
# speed of learning
model.setBackpropWeightScale(0.01)
# don't be part of lowest
model.setBackpropMomentumScale(0.1)
#criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
#criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
print('Training MLP ...')
#num_iter = model.train(train, train_labels, None, params = params)
num_iter = model.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

# set end time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print('Training duration:', time)
#print 'Ran for %d iterations' % num_iter

# train data
ret_0, resp_0 = model.predict(train)
prediction_0 = resp_0.argmax(-1)
true_labels_0 = train_labels.argmax(-1)

train_rate = np.mean(prediction_0 == true_labels_0)
print('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

# test data
ret_1, resp_1 = model.predict(test)
prediction_1 = resp_1.argmax(-1)
true_labels_1 = test_labels.argmax(-1)

test_rate = np.mean(prediction_1 == true_labels_1)
print('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

if not os.path.exists('mlp_xml'):
    os.makedirs('mlp_xml')

# save model
model.save('mlp_xml/mlp.xml')