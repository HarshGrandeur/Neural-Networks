import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = '/usr/Python/Neural Network/101_ObjectCategories'
TEST_DIR = 'test1'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'objects-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match



names=list()
def create_train_data():
    training_data = []
    step=0  # a variable to process a part of the entire set, currently processes 10 categories, you can change to add more categories
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=np.zeros(10)
        label[step]=1
        names.append(img)
        for i in tqdm(os.listdir(os.path.join(TRAIN_DIR,img))):

            path =os.path.join(TRAIN_DIR,img)
            path=os.path.join(path,i)
            # print(path)
            path=str(path)
            # print(path)
            img1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            # print(img)
            img2 = cv2.resize(img1, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img2),label])
            # print(training_data)
        step=step + 1
        if  step==10:
            break
    shuffle(training_data)
    np.save('train_data_101_objects.npy', training_data)
    # np.save('categories.npy',names)
    return training_data

training_data=create_train_data()

train_data=np.load('train_data_101_objects.npy')
print(train_data[1])
print(type(train_data[1][1]))
print(type(train_data[1][0]))



import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
#
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log1')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
# #
train = train_data[:-50]
test = train_data[-50:]
# #
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = np.array([i[1] for i in train])
# print(X)
# print(Y)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = np.array([i[1] for i in test])
#
model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('model2.tflearn')
model.load('model2.tflearn')
import matplotlib.pyplot as plt


test_data = test_x

for num,data in enumerate(test_x[27:36]):

    print(num)
    y = fig.add_subplot(3,3,num+1)
    orig = train_data[num][0]
    data = orig.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    print(model_out)
    print(np.argmax(model_out))
    str_label=names[np.argmax(model_out)]
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
plt.show()
