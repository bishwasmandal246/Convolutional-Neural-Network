# Import all the required modules

from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np

'''x_train and y_train : 50000 Images and Labels
x_test and y_test : 10000 Images and Labels. '''

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#Converting all the values between 0 to 1

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#One hot encoding

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

#Label types

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

model = load_model(filepath='cifar10_Classifier.h5')

train_results = model.evaluate( x = x_train, y = y_train)
test_results = model.evaluate( x = x_test, y = y_test)

print("\nTrain Loss:", train_results[0])
print("Train Accuracy:",train_results[1])

print("\nTest Loss:", test_results[0])
print("Test Accuracy:",test_results[1])

#predicting the output from the model using first ten testsets

for i in range(10):
    test_image = np.asarray([x_test[i]])
    results=model.predict(test_image)
    image_index = np.argmax(results)
    print("The given input image is categorized as:",labels[image_index])





