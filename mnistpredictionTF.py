



import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D

EPOCHS = 5

class ExploreData(object):
	"""docstring for ExploreData"""
	def __init__(self, data, label):
		super(ExploreData, self).__init__()
		self.shape = data.shape
		self.lenLabel = len(label)
		self.lenData = len(data)

	def run(self):
		print('''
			Data Shape : {}
			Number of Data/Labels : {}
			Train Labels : {}
			'''.format(
					self.shape,
					self.lenLabel,
					self.lenData
				))

class GetData(object):
	"""docstring for GetData"""
	def __init__(self, dataset):
		super(GetData, self).__init__()
		self.dataset = dataset
		
	def load(self):
		(train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
		return (train_images, train_labels), (test_images, test_labels)

class InspectImage(object):
	"""docstring for InspectImage"""
	def __init__(self, data, sample=0):
		super(InspectImage, self).__init__()
		self.image = data[sample]
	
	def run(self):
		plt.figure()
		plt.imshow(self.image)
		plt.colorbar()
		plt.grid(False)
		plt.show()

class PreprocessData(object):
	"""docstring for PreprocessData"""
	def __init__(self, train, test):
		super(PreprocessData, self).__init__()
		self.train = train
		self.test = test

	def transform(self):
		new_train = self.train / 255.0
		new_test = self.test / 255.0
		new_train = new_train[..., tf.newaxis]
		new_test = new_test[..., tf.newaxis]
		return new_train, new_test
		
class MyModel(keras.Model):
	"""docstring for MyModel"""
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, 3, activation='relu')
		self.flatten = Flatten()
		self.d1 = Dense(128, activation='relu')
		self.d2 = Dense(10, activation='softmax')	

	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)

@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
	predictions = model(images)
	t_loss = loss_object(labels, predictions)

	test_loss(t_loss)
	test_accuracy(labels, predictions)


if __name__ == '__main__':

	mnist = keras.datasets.mnist

	getdata = GetData(mnist)
	(X_train, y_train), (X_test, y_test) = getdata.load()

	#ed = ExploreData(X_train, y_train)
	#ed.run()

	#img = InspectImage(X_train, sample = 1)
	#img.run()

	preprocess = PreprocessData(X_train, X_test)
	X_train, X_test = preprocess.transform()

	train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
	test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

	model = MyModel()

	loss_object = keras.losses.SparseCategoricalCrossentropy()
	optimizer = keras.optimizers.Adam()

	train_loss = keras.metrics.Mean(name='train_loss')
	train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	test_loss = keras.metrics.Mean(name='test_loss')
	test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	for epoch in range(EPOCHS):
		for images, labels in train_ds:
			train_step(images, labels)

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels)

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
		print (template.format(epoch+1,
							   train_loss.result(),
							   train_accuracy.result()*100,
							   test_loss.result(),
							   test_accuracy.result()*100))


