import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np 
import onnx 

#loading mnist data
(trainX, trainY), (testX, testY)= mnist.load_data()

#normalize the dataset
trainX= tf.keras.utils.normalize(trainX,axis=1)
testX= tf.keras.utils.normalize(testX,axis=1)

#sample data
plt.imshow(trainX[0] , cmap="gray")
plt.show()

print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

# plot first few images
fig = plt.figure()
for i in range(0,60):
	# define subplot
	plt.subplot(6,10, 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i].squeeze(), cmap=plt.get_cmap('gray'))
fig
# show the figure
plt.show()

#Model creation
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
#hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#layer to out put
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#model compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.01, name='Adam')
loss_fn = keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
# #    initial_learning_rate=0.01,decay_steps=10000,
# #    decay_rate=0.9)
# #opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
# #loss_fn = keras.losses.SparseCategoricalCrossentropy()
# #model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

#train model
model.fit(x=trainX, y=trainY, epochs=1)

weights= model.get_weights()
print(weights)

#model performance
test_loss, test_acc = model.evaluate(x=testX,y=testY)

print('\nTest Loss:',test_loss)
print('\nTest accuracy:',test_acc)

#Predictions
predictions= model.predict([testX])

print(np.argmax(predictions[0]))

plt.imshow(testX[0], cmap="Blues")
plt.show()