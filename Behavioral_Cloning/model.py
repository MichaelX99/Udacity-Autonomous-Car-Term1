#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:56:29 2017

@author: mike
"""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dense, Flatten, Lambda
import math
from keras.layers import Dropout, MaxPooling2D

print("Importing Data")

first_lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        first_lines.append(line)
first_lines.remove(['center','left','right','steering','throttle','brake','speed'])

train = first_lines

train_samples, validation_samples = train_test_split(train, test_size=0.2)

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            correction = 0.25

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Get filenames
                center_name = './data/data/IMG/'+batch_sample[0].split('/')[-1]
                right_name = center_name.replace("center","right")
                left_name = center_name.replace("center","left")
                
                # Read in the images
                center_image = cv2.imread(center_name)
                right_image = cv2.imread(right_name)
                left_image = cv2.imread(left_name)
                
                # Read in the angle
                center_angle = float(batch_sample[3])
                right_angle = center_angle - correction
                left_angle = center_angle + correction
                
                # Add the center image
                images.append(cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV))
                angles.append(center_angle)

            	  # Add the flipped center image
                images.append(cv2.cvtColor(np.fliplr(center_image),cv2.COLOR_RGB2YUV))
                angles.append(-center_angle)
                
                # Add the right image
                images.append(cv2.cvtColor(right_image, cv2.COLOR_RGB2YUV))
                angles.append(right_angle)
		
                # Add the flipped right image
                images.append(cv2.cvtColor(np.fliplr(right_image),cv2.COLOR_RGB2YUV))
                angles.append(-right_angle)
                
                # Add the left image
                images.append(cv2.cvtColor(left_image, cv2.COLOR_RGB2YUV))
                angles.append(left_angle)

                # Add the flipped left image
                images.append(cv2.cvtColor(np.fliplr(left_image),cv2.COLOR_RGB2YUV))
                angles.append(-left_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

aug_num = 6

b_size = math.floor(256/aug_num)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)


im=cv2.imread('./data/data/'+line[0])
shape = np.shape(im)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=shape))
# Crop Images
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Layer 2 Convolution
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
# Layer 3 Convolution
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
# Layer 4 Convolution
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D((2, 2),dim_ordering='th'))
# Layer 5 Convolution
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(.5))
# Layer 6 Convolution
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D((2, 2),dim_ordering='th'))
# Flatten
model.add(Flatten())
# Layer 7 Fully Connected
model.add(Dense(100))
model.add(Dropout(.5))
# Layer 8 Fully Connected
model.add(Dense(50))
# Layer 9 Fully Connected
model.add(Dense(10))
# Model Output
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the model
history_object = model.fit_generator(train_generator, samples_per_epoch= aug_num*len(train_samples), validation_data=validation_generator, nb_val_samples=aug_num*len(validation_samples), nb_epoch=3)

# Save the model
model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()