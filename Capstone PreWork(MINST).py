#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.datasets import mnist
import matplotlib.pyplot as plt

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape: ", x_train.shape)  
print("Test data shape", x_test.shape)  

# Flatten the images
image_vector_size = 28*28

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Train Samples: ", x_train.shape)  
print("Test Samples", x_test.shape)  


# In[9]:


import keras
from keras.datasets import mnist

# Setup train and test splits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training label shape: ", y_train.shape)  
print("First 5 training labels: ", y_train[:5])  

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])
print("Shape after encoding: ", y_train.shape)


# In[10]:


from keras.layers import Dense  
from keras.models import Sequential

image_size = 28*28  
num_classes = 10

model = Sequential()
model.add(Dense(units=8, activation='sigmoid', input_shape=(784,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()


# In[11]:


model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_test, y_test))


# In[13]:


score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




