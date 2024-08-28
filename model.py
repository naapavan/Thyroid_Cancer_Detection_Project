# %% [markdown]
# # Import needed modules

# %%
# import system libs 
import os
import time
import shutil
import itertools
import pickle

# import data handling tools 
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')



import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense,Flatten, Input, Dropout

# Load Xception model
base = Xception(weights="imagenet", input_shape =(299,299,3),include_top= False)
# set base model trainable to false
for layers in base.layers:
	layers.trainable=False

base.summary()

batch_size = 32


# Define augmentation
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		validation_split=0.25,
		horizontal_flip =True
		)

# apply augmentations on dataset
train =train_datagen.flow_from_directory(
	"C:/Users/poorna/OneDrive/Desktop/Thyroid Cancer Project/dataset-thyroid",
	target_size=(299, 299),
	batch_size=batch_size,
	class_mode='categorical',
	subset='training')
val =train_datagen.flow_from_directory(
	"C:/Users/poorna/OneDrive/Desktop/Thyroid Cancer Project/dataset-thyroid",
	target_size=(299, 299),
	batch_size=batch_size,
	class_mode='categorical',
	subset='validation')
class_names=['4A','4B','4C','5','Benign','normal thyroid']

# code to plot images
def plotImages(images_arr, labels):
	fig, axes = plt.subplots(12, 4, figsize=(20,80))
	axes = axes.flatten()
	label=0
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.set_title(class_names[np.argmax(labels[label])])
		label=label+1
	plt.show()

# Define our complete models
model = Sequential()
model.add(Input(shape =(299,299,3)))
model.add(base)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(6,activation='softmax'))
model.summary()


# import adam optimizer
from tensorflow.keras.optimizers import Adam
# compile model(define metrics and loss)
model.compile(
	optimizer=Adam(learning_rate=1e-3),
	loss="categorical_crossentropy",
	metrics=["accuracy"],
)
# train model for 30 epoch
model.fit(train, epochs=30, validation_data=val)

# save model
model.save('epoch_30.h5')














# %%
