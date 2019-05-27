import tensorflow as tf
import pickle as pkl
from tensorflow import set_random_seed
from utils import data_list_batch_23_4_1
import scipy.stats as stats

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 4, 1], padding='SAME')

def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 1, 1],
                          strides=[1, 5, 1, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 23, 4, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

x_image = tf.reshape(x, [-1, 23, 4, 1])

W_filter_expan = weight_variable([1, 4, 1, 10])
b_filter_expan = bias_variable([10])

W_conv1_1 = weight_variable([1, 4, 1, 10])
b_conv1_1 = bias_variable([10])

W_conv1_2 = weight_variable([2, 4, 1, 10])
b_conv1_2 = weight_variable([10])

W_conv1_3 = weight_variable([3, 4, 1, 10])
b_conv1_3 = weight_variable([10])

W_conv1_5 = weight_variable([5, 4, 1, 10])
b_conv1_5 = weight_variable([10])


##### Batch Normalization #####
conv_layer_1 = conv2d(x_image, W_conv1_1) + b_conv1_1
conv_layer_2 = conv2d(x_image, W_conv1_2) + b_conv1_2
conv_layer_3 = conv2d(x_image, W_conv1_3) + b_conv1_3
conv_layer_5 = conv2d(x_image, W_conv1_5) + b_conv1_5

conv_layer = tf.concat([conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_5], 3)

conv_mean, conv_var = tf.nn.moments(conv_layer,axes=[0, 1, 2])
scale2 = tf.Variable(tf.ones([23, 1, 40]))
beta2 = tf.Variable(tf.zeros([23, 1, 40]))
conv_BN = tf.nn.batch_normalization(conv_layer, conv_mean, conv_var, beta2, scale2, 1e-3)
h_conv1 = tf.nn.relu(conv_BN)

h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([1 * 5 * 40, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool1, [-1, 1 * 5 * 40])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_dense_1 = weight_variable([100, 23])
b_dense_1 = bias_variable([23])
y_dense_1 = tf.nn.relu(tf.matmul(h_fc1_drop, W_dense_1) + b_dense_1)

# W_ouput = weight_variable([23, 2])
# b_ouput = bias_variable([2])
# y_conv = tf.nn.softmax(tf.matmul(y_dense_1, W_ouput) + b_ouput)

W_ouput = weight_variable([23, 1])
b_ouput = bias_variable([1])
y = tf.matmul(y_dense_1, W_ouput) + b_ouput

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cost = tf.reduce_mean(tf.pow((y_-y),2))
# train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

min_batch = 128

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# saver = tf.train.Saver()

# saver.restore(sess, "../CNN_std_model/cnn_all_train.ckpt")

# load test_data
# test_data = test[0]
# test_data_label = test[1][:, 0]

# predict off-target effect
# res = sess.run(y_conv, feed_dict={x: test_data, keep_prob: 1.0})

# result = res[:, 0]

def train_CNN_Lin(trainData,trainDataAll,testDataAll,ENZ):        
	for e in range(110): # for e in range(60):
		for featureData,targetLabel in data_list_batch_23_4_1(trainData):
			sess.run(train_step, feed_dict={x: featureData, y_:targetLabel, keep_prob: 0.85})
			
		targetLabelList = []
		outputList = []
		trainMSEList = []
		trainMSEloss = 0
		
		for featureData,targetLabel in trainDataAll:
			output = y.eval(feed_dict={x: featureData, y_: targetLabel, keep_prob: 1})
			trainMSEloss_temp = cost.eval(feed_dict={x: featureData, y_: targetLabel, keep_prob: 1})
			trainMSEList.append(trainMSEloss_temp)
			targetLabel_temp = targetLabel.squeeze().tolist()
			output_temp = output.squeeze().tolist()
			if not(type(output_temp) == list):
				output_temp = [output_temp]
			if not(type(targetLabel_temp) == list):
				targetLabel_temp = [targetLabel_temp]
			targetLabelList = targetLabelList + targetLabel_temp
			outputList = outputList + output_temp
			
		trainMSEloss = sum(trainMSEList) / len(trainMSEList)
		train_spcc, _ = stats.spearmanr(targetLabelList, outputList) #spearmar
	
		featureData = testDataAll[0]
		targetLabel = testDataAll[1]
		output = y.eval(feed_dict={x: featureData, y_: targetLabel, keep_prob: 1})
		testMSEloss = cost.eval(feed_dict={x: featureData, y_: targetLabel, keep_prob: 1})
		targetLabel_st = targetLabel.squeeze().tolist()
		output_cdnst = output.squeeze().tolist()
		test_spcc, _ = stats.spearmanr(targetLabel_st, output_cdnst)
		
		print("epch {0}: train_loss {1}, train_spcc {2} || test_loss {3}, test_spcc {4}".format(e,trainMSEloss,train_spcc,testMSEloss,test_spcc))
