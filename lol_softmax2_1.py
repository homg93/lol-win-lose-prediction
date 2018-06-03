import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)
'''
xy = np.loadtxt('lol.csv', delimiter = ',', dtype = np.float32)
xy = MinMaxScaler(xy)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
'''
xy = pd.read_csv('lol_test.csv', header=None)
xy = MinMaxScaler(xy)

df_xy = xy.values

x_data = df_xy[:, 0:6].astype(float)
x_data[:,4:6] = df_xy[:,10:-2].astype(float)
y_data = df_xy[:, 12:14]

print x_data.shape, "\n", x_data, len(x_data)
#print(y_data.shape, "\n", y_data)

X = tf.placeholder("float", [None,6])
Y = tf.placeholder("float", [None,2])

W = tf.Variable(tf.random_normal([6,2]), name = 'weight')
b = tf.Variable(tf.random_normal([2]), name = 'bias')


logits = tf.matmul(X, W) + b

hypothesis = tf.nn.softmax(logits)

# hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
'''
W1 = tf.Variable(tf.random_normal([12,12]),name = 'weight1')
b1 = tf.Variable(tf.random_normal([12]),name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random_normal([12,1]),name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]),name = 'bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2)
'''

# One-hot Encoding
#Y_one_hot = tf.one_hot(Y, 2)
#Y_one_hot = tf.reshape(Y_one_hot, [-1, 2])

#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
#cost = tf.reduce_mean(cost_i)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis)+(1 - Y)*(tf.log(1 - hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate = 1e-1).minimize(cost)

#prediction = tf.argmax(hypothesis, 1)
#target = tf.argmax(Y_one_hot, 1)

#check_prediction = tf.equal(prediction, target)
#accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
#predicted = tf.cast(hypothesis = , dtype = tf.float32)
#accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2101):
		sess.run([train], feed_dict = {X:x_data, Y: y_data})
		if step % 200 == 0:
			print step, sess.run(cost, feed_dict= {X:x_data, Y: y_data})
			

	#h,c,a = sess.run([ hypothesis, prediction, accuracy], feed_dict= {X:x_data, Y: y_data})

	# print("\nHypothesis:\n ", h, "\nCorrect (Y):\n ", c, "\nAccuracy: ", a)
	#print "\nHypothesis:\n ", h, "\nCorrect (Y):\n ", c, "\nAccuracy: ", a
	#input_data = sess.run(hypothesis, feed_dict = {X:[[3,20,21763,35002,27951,43121,43,47,52839,41168,366,467]]})
	#input_data = sess.run(logits2, feed_dict = {X:[[49,10,50044,30570,92922,45369,40,38,74319,113575,432,361]]})
	#input_data = sess.run(logits, feed_dict = {X:[[49,10,50044,30570,432,361]]})
	#input_data = sess.run(hypothesis, feed_dict = {X:[[0.43301179,0.49626866]]})
	#input_data = sess.run(hypothesis, feed_dict = {X:[[0.48780488,0.25806452,0.41509999,0.1936706,0.22850249,0.1901986,0.34615385,0.32989691,0.21184669,0.2398918,0.43301179,0.49626866]]})
	#input_data = sess.run(hypothesis, feed_dict = {X:[[44,19,71849,60635,805,861]]})
	input_data = sess.run(hypothesis, feed_dict = {X:[  [ 0.02439024,  0.18181818,  0.00890925,  0.09511665,  0.10720268,  0.27231121]]})
	#input_data = sess.run(hypothesis, feed_dict = {X:[[ 0.04878049,0.0001,0.04638371,0.02556374,0.180067,0.28375286]]})
	# if input_data >= 0.5:
	# 	# print("\nYour team will be win!!!!",input_data)
	print "\ninput data : ",input_data
	# else:
	# 	# print("\nYour team will be lose...",input_data)
	# 	print "\nYour team will be lose...",input_data