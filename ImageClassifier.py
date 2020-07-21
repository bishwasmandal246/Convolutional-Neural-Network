# Import all the required modules

from keras.datasets import cifar10
import keras.utils as utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

'''x_train and y_train : 50000 Images and Labels
x_test and y_test : 10000 Images and Labels'''

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#Converting all the values between 0 to 1

x_train = x_train.astype('float32')/ 255.0
x_test = x_test.astype('float32') / 255.0

#One hot encoding

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

#Label types

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Building a Model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,3), activation='relu', padding='same',\
                 kernel_constraint=maxnorm(3))) #maxnorm does the scaling
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) #Converting into 1-D array
model.add(Dense(units=512,activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.5)) #dropping 50 percent of the neurons, helps to deal with overfitting
model.add(Dense(units=10, activation='softmax'))

#Compiling the built model

model.compile(optimizer = SGD (lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

#Training the model

model.fit(x = x_train, y = y_train, epochs = 35, batch_size=32)

#Saving the model in h5 format

model.save(filepath='cifar10_Classifier.h5')





