import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from keras import optimizers
from keras.utils import np_utils

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


sgd = optimizers.SGD(lr=0.1, clipnorm=0.5)
x_train = np.array([[1,0,-1,1],[1,1,0,1],[1,1,-1,1],[1,0,0,1],[0,0,0,0],[1,0,1,1],[1,1,0,1],[1,1,-1,1],[1,1,1,1],[0,0,0,0],[1,1,-1,-1],[1,1,0,0],[1,1,-1,0],[1,0,0,1],[0,0,0,0],[1,0,-1,1],[1,1,0,1],[1,1,-1,0],[1,0,0,0],[0,0,0,0],[1,1,0,-1],[1,1,0,1],[1,1,1,-1],[1,0,1,0],[0,0,0,0],[1,0,-1,1],[1,1,0,1],[1,1,-1,0],[1,1,0,0],[0,0,0,0]])
y_train = np.array([2,1,2,1,0,1,1,2,1,0,2,1,2,1,0,2,1,2,1,0,2,1,2,1,0,2,1,2,1,0])
print (x_train.shape)
print (y_train.shape)
x_test = np.array([[1,0,-1,1],[1,1,0,0],[1,1,-1,0],[1,0,0,1],[0,0,0,0],[1,0,-1,1],[1,1,0,0],[1,1,-1,1],[1,0,0,0],[0,0,0,0],[1,0,-1,0],[1,1,0,0],[1,1,-1,0],[1,1,0,0],[0,0,0,0]])
y_test = np.array([2,1,2,1,0,2,1,2,1,0,2,1,2,1,0])
# x_test =keras.utils.to_categorical(x_test)
# x_train= keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test,3)

y_train= keras.utils.to_categorical(y_train,3)

model = Sequential()
model.add(Dense(512,input_shape=(4,),activation='relu'))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=100, epochs=100)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc:', test_acc)
y_result = model.predict(x_test)
print(np.argmax(y_result,axis=1))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["accuracy","loss"],loc="lower right")
plt.show()







