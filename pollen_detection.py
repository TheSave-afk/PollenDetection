import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import matplotlib.cm as cm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#from vis.utils import utils
from matplotlib import pyplot as plt
#from tensorflow_core.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os
import sys
from PIL import Image, ImageDraw
import tensorflow_datasets as tfds

img_width, img_height = 84, 84

train_data_dir = 'D:/Desktop/progetto/N-fold/train'
test_data_dir = 'D:/Desktop/progetto/N-fold/test'

epochs = 1
batch_size = 4

total_acc = []

# 4 classi di pollini (1,2,3,4)
class_weight = {0 : 1., 1 : 1., 2: 1., 3 : 1.,}

def get_model(learning_rate, mom):

    base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3)) ## BASE VGG

    model = Sequential()
    for l in base_model.layers:
        model.add(l)
    
    ## ARCHITETTURA SOVRASTANTE DELLA RETE ## 

    ## 1 model.add(GlobalAveragePooling2D()) -  model.add(Dense(4, activation='softmax', name='outL'))

    ## 2
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', name='outL'))
    ## ## ## 

    for layer in model.layers[:18]:
        layer.trainable = False
    
    # model.load_weights('top_model.h5')
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=mom, nesterov=True)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

    return model

######################  MAIN  ######################
#cross validation con 5 fold    
for i in range(5):
  train_path = train_data_dir + str(i+1) + '/'
  test_path = test_data_dir + str(i+1) + '/'

  # Dataset
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    #validation_split=0.2,
    #subset="training",
    class_names=['1','2','3','4'],
    #seed=0,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

  validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode='categorical',
    #validation_split=0.2,
    #subset="validation",
    class_names=['1','2','3','4'],
    #seed=0,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

  # Modello
  my_model = get_model(learning_rate = 0.0001, mom = 0.5)
  my_model.summary()

  print('------------------------------------------------------------------------')
  print(f'Training for fold {i} ...')

  filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  history = my_model.fit(
      train_ds,
      validation_data = validation_ds,
      shuffle=True,
      # steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs, 
      verbose=1,
      class_weight=class_weight,
      # callbacks=callbacks_list,
      # validation_steps=nb_validation_samples // batch_size, # comm
  )

  #print(f'Score for fold {i}: {my_model.metrics_names[0]} of {checkpoint[0]}; {my_model.metrics_names[1]} of {checkpoint[1]*100}%')
  print(max(history.history['val_accuracy']))
  max_acc = max(history.history['val_accuracy'])

  total_acc.append(max_acc)


for i in total_acc:
  final_acc += total_acc[i]

final_acc = final_acc / 5

print("total ")
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {final_acc}')
print('------------------------------------------------------------------------')

my_model.save('modello.h5', overwrite=True)