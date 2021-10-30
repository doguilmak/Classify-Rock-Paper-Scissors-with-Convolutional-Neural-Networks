# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 00:59:04 2021

@author: doguilmak

dataset: https://www.kaggle.com/sanikamal/rock-paper-scissors-dataset

"""
#%%
# 1. Importing Libraries

import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

#%%
# PART 1 - CNN

start = time.time()
classifier = load_model('model.h5')
classifier.summary()
"""
# Initialization
classifier = Sequential()

# Step 1 - First Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))  
# input_shape = (64, 64, 3) size of 64x64 pictures with RGB colors (3 primary colors).

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Flattening
classifier.add(Flatten())

# Step 5 - Artificial Neural Network
classifier.add(Dense(output_dim = 128, activation = 'relu'))  # Gives 128bit output
classifier.add(Dense(output_dim = 3, activation = 'softmax'))  

#%%
# PART 2 - CNN and Pictures

from keras.preprocessing.image import ImageDataGenerator
## ImadeDataGenerator library for pictures.
## The difference from normal picture readings is that it evaluates the pictures one by one, not all at once and helps the RAM to work in a healthy way.

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
## shear_range = Side bends
## zoom_range = Zoom

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Train data
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'categorical')
## target_size= 64x64 size pictures for scan.

# Data is tested
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'categorical')

validation = train_datagen.flow_from_directory(
                                            "validation",
                                            target_size=(224, 224),
                                            color_mode="rgb",
                                            class_mode="categorical",
                                            batch_size=32,
                                            shuffle=True,
                                            seed=42,
                                            subset="validation")

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
classifier.summary()
classifier_history=classifier.fit(training_set,validation_data=validation,epochs=2, callbacks=[
            tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True)])

print(classifier_history.history.keys())
#classifier.save('model.h5')

# Plot accuracy and loss
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(classifier_history.history['accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(classifier_history.history['loss'], c='green')
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

#%%
# PART 3 - Prediction of the Classes

test_set.reset()
pred=classifier.predict_generator(test_set, verbose=1)

## Filter predictions
pred[pred > .5] = 1
pred[pred <= .5] = 0
print('Prediction successful.')
"""

from keras.preprocessing import image
import numpy as np

# Predict paper class
image = image
print('\nPrediction of paper class')
for i in range(1, 10):
    i = str(i)
    path = 'validation/paper' + i + '.png'
    img = image.load_img(path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    classes_paper = classifier.predict(images, batch_size=10)
    print(classes_paper)

# Predict rock class
print('\nPrediction of rock class')
for i in range(1, 10):
    i = str(i)
    path = 'validation/rock' + i + '.png'
    img = image.load_img(path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    classes_rock = classifier.predict(images, batch_size=10)
    print(classes_rock)
    
# Predict scissors class
print('\nPrediction of scissors class')
for i in range(1, 10):
    i = str(i)
    path = 'validation/scissors' + i + '.png'
    img = image.load_img(path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    classes_scissors = classifier.predict(images, batch_size=10)
    print(classes_scissors)


end = time.time()
cal_time = end - start
print("\nTook {} seconds to classificate objects.".format(cal_time))
