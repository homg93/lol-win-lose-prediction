import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy = pd.read_csv('lol_test.csv', header=None)

df_xy = xy.values

x_data = df_xy[:100, 0:12].astype(float)
#x_data[:,4:6] = df_xy[:100,10:-2].astype(float)
y_data = df_xy[:100, 12:14]

#print x_data.shape, "\n", x_data, len(x_data)
#print(y_data.shape, "\n", y_data)

x_test = df_xy[100:, 0:12].astype(float)
#x_test[:,4:6] = df_xy[100:,10:-2].astype(float)
y_test = df_xy[100:, 12:14]

print x_test.shape, "\n", x_test, len(x_test)

x_data = MinMaxScaler(x_data)
x_test = MinMaxScaler(x_test)
#print x_test.shape, "\n", x_test, len(x_test)

X = tf.placeholder("float", [None,12])
Y = tf.placeholder("float", [None,2])
#Y = tf.placeholder(tf.int32, shape= [None,1])

W = tf.Variable(tf.random_normal([12,2]), name = 'weight')
b = tf.Variable(tf.random_normal([2]), name = 'bias')


logits = tf.matmul(X, W) + b

hypothesis = tf.nn.softmax(logits)

# One-hot Encoding
#Y_one_hot = tf.one_hot(Y, 2)
#Y_one_hot = tf.reshape(Y_one_hot, [-1, 2])

#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
#cost = tf.reduce_mean(cost_i)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis)+(1 - Y)*(tf.log(1 - hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate = 1e-1).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y,1))
#target = tf.argmax(Y_one_hot, 1)

#check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#predicted = tf.cast(hypothesis = , dtype = tf.float32)
#accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(1501):
		sess.run([train], feed_dict = {X:x_data, Y: y_data})
		if step % 200 == 0:
			print step, sess.run(cost, feed_dict= {X:x_data, Y: y_data})
			

	#h,c,a = sess.run([ hypothesis, prediction, accuracy], feed_dict= {X:x_data, Y: y_data})

	# print("\nHypothesis:\n ", h, "\nCorrect (Y):\n ", c, "\nAccuracy: ", a)
	
	print "Prediction: ", sess.run(hypothesis, feed_dict={X: x_test})
	print "Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test})