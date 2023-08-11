import os
import numpy as np
import caer 
import canaro
import cv2 as cv
import gc
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
from keras import layers, models
from keras.callbacks import LearningRateScheduler


IMG_SIZE = (80,80)
channels = 1
char_path = r'D:\OpenCV\Deep Computer Vison\Simpsons Data\simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort in descending order

char_dict = caer.sort_dict(char_dict, descending=True)
print(char_dict)

characters = []
count = 0

# Grabbing top 10 characters with most faces

for i in char_dict:
    characters.append(i[0])
    count+=1
    if count>=10:
        break
print(characters)

train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
len(train)

plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap = 'gray')
plt.show()

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet ==> (0,1)

featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(featureSet, labels, test_size=.2)
#x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

del train
del featureSet
del labels
gc.collect()

#Image Data Generator

BATCH_SIZE = 32
EPOCHS = 10
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Creating the Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to feed into dense layers
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))  # Using sigmoid for binary classification

learning_rate = 0.001
momentum = 0.9
nesterov = True
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE,epochs=EPOCHS,validation_data=(x_val,y_val),validation_steps=len(y_val)//BATCH_SIZE, callbacks=callbacks_list)

test_path = r'D:\OpenCV\Deep Computer Vison\Simpsons Data\simpsons_dataset\bart_simpson\pic_0000.jpg'
img = cv.imread(test_path)
plt.imshow(img, cmap='gray')
plt.show()

def prepare(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img,IMG_SIZE,1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])