from google.colab import drive
drive.mount('/content/drive')

import os
import cv2

datadir='/content/drive/MyDrive/wp'
classes=os.listdir(datadir)    #classes contain : ['sunny', 'cloudy', 'rainy', 'foggy', 'thunderstorm']

**resizing images**

  for i in classes:
  ipath='/content/drive/MyDrive/wp/'+i
  for im in os.listdir(ipath):
    image = cv2.imread(ipath+'/'+im)
    resizedimage = cv2.resize(image,(500,500))
    cv2.imwrite(ipath+'/'+im,resizedimage)

!pip install split-folders

**data augmentation**

  # Importing necessary functions
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
train_datagen = ImageDataGenerator(
    rescale=1./255,
		rotation_range = 40,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
    width_shift_range=0.2,
    height_shift_range=0.2,
		brightness_range = (0.5, 1.5),
    fill_mode='nearest')
	

for i in classes:
  ipath='/content/drive/MyDrive/wp/'+i
  for im in os.listdir(ipath):
    img=load_img(ipath+'/'+im)
    x=img_to_array(img)
    x=x.reshape((1,)+x.shape)
    j=1
    for batch in train_datagen.flow(x,batch_size=1,save_to_dir=ipath,save_prefix='ag',save_format='jpg'):
      j+=1
      if j>5:
        break

import splitfolders
input = '/content/drive/MyDrive/wp'
splitfolders.ratio(input,output="imgdataset",seed=100,ratio=(.7,.0,.3),group_prefix=None)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
		rotation_range = 40,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
    width_shift_range=0.2,
    height_shift_range=0.2,
		brightness_range = (0.5, 1.5),
    fill_mode='nearest')


test_datagen=ImageDataGenerator(rescale=1./255)

batch_size = 32
img_height = 500
img_width = 500

 #training

train_ds = train_datagen.flow_from_directory(
  directory='/content/imgdataset/train',
  target_size=(img_height, img_width),
  batch_size=batch_size,
  classes = ['cloudy','foggy','rainy','sunny','thunderstorm'],
  class_mode='categorical')

#validation

test_ds = test_datagen.flow_from_directory(
  directory='/content/imgdataset/test',
  target_size=(img_height, img_width),
  batch_size=batch_size,
  classes = ['cloudy','foggy','rainy','sunny','thunderstorm'],
  class_mode='categorical')

train_ds.class_indices  #{'cloudy': 0, 'foggy': 1, 'rainy': 2, 'sunny': 3, 'thunderstorm': 4}

#model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D#,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32
epochs = 10
IMG_HEIGHT = 500
IMG_WIDTH = 500

#model
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),#BatchNormalization()
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),#BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    Conv2D(128, 3, padding='same', activation='relu'),#BatchNormalization(),
    MaxPooling2D(),
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, padding='same', activation='relu'),#BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    #Dense(128, activation='relu')
    #Dropout(0.5)'''
    Dense(5, activation='softmax')
])

model.summary()


#model compilation

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model fitting

history = model.fit_generator(
    train_ds,  #734 images = batch_size * steps
    epochs=epochs,
    validation_data=test_ds
  # 319 images = batch_size * steps
)
model.save('model.1')

#prediction

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import load_img
test_image = load_img('/content/drive/MyDrive/testing/sunny.jpg', target_size = (500,500))
plt.imshow(test_image)
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
print(result)

if result[0][0] == 1:
    print("CLOUDY")
elif result[0][1] == 1:
    print("FOGGY")
elif result[0][2] == 1:
    print("RAINY")
elif result[0][3] == 1:
    print("SUNNY")
elif result[0][4] == 1:
    print("THUNDERSTORM")

#output1
1/1 [==============================] - 0s 29ms/step
[[0. 0. 0. 1. 0.]]
SUNNY

test_image = load_img('/content/drive/MyDrive/testing/foggy.jpg', target_size = (500,500))
plt.imshow(test_image)
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
print(result)

if result[0][0] == 1:
    print("CLOUDY")
elif result[0][1] == 1:
    print("FOGGY")
elif result[0][2] == 1:
    print("RAINY")
elif result[0][3] == 1:
    print("SUNNY")
elif result[0][4] == 1:
    print("THUNDERSTORM")

#output2

1/1 [==============================] - 0s 18ms/step
[[0. 1. 0. 0. 0.]]
FOGGY

test_image = load_img('/content/drive/MyDrive/testing/rainy.jpg', target_size = (500,500))
plt.imshow(test_image)
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
print(result)

if result[0][0] == 1:
    print("CLOUDY")
elif result[0][1] == 1:
    print("FOGGY")
elif result[0][2] == 1:
    print("RAINY")
elif result[0][3] == 1:
    print("SUNNY")
elif result[0][4] == 1:
    print("THUNDERSTORM")

#output3
1/1 [==============================] - 0s 21ms/step
[[2.0047709e-08 2.6170829e-35 6.2406334e-22 1.0000000e+00 1.3131848e-11]]
SUNNY

#accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='test')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
