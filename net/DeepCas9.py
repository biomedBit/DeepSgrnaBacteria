import os, sys
from os.path import exists
from os import system
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.stats
from scipy.stats import stats
from utils import data_list_batch_1_30_4

np.set_printoptions(threshold='nan')

# Model
length = 30
filter_size = [3, 5, 7]
filter_num = [100, 70, 40]
node_1 = 80
node_2 = 60
l_rate = 0.001
inputs_sg = tf.placeholder(tf.float32, [None, 1, length, 4])
inputs = inputs_sg
y_ = tf.placeholder(tf.float32, [None, 1])
targets = y_
is_training = False
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,num_filters]
    # initialise weights and bias for the filterS
    weights  = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='VALID')
    out_layer += bias
    out_layer = tf.layers.dropout(tf.nn.relu(out_layer), 0.3, is_training)
    ksize  = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 2, 1]
    out_layer = tf.nn.avg_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')
    return out_layer

L_pool_0 = create_new_conv_layer(inputs, 4, filter_num[0], [1, filter_size[0]], [1, 2],name='conv1')
L_pool_1 = create_new_conv_layer(inputs, 4, filter_num[1], [1, filter_size[1]], [1, 2],name='conv2')
L_pool_2 = create_new_conv_layer(inputs, 4, filter_num[2], [1, filter_size[2]], [1, 2],name='conv3')
with tf.variable_scope('Fully_Connected_Layer1'):
    layer_node_0 = int((length-filter_size[0])/2)+1
    node_num_0   = layer_node_0*filter_num[0]
    layer_node_1 = int((length-filter_size[1])/2)+1
    node_num_1   = layer_node_1*filter_num[1]
    layer_node_2 = int((length-filter_size[2])/2)+1
    node_num_2   = layer_node_2*filter_num[2]
    L_flatten_0  = tf.reshape(L_pool_0, [-1, node_num_0])
    L_flatten_1  = tf.reshape(L_pool_1, [-1, node_num_1])
    L_flatten_2  = tf.reshape(L_pool_2, [-1, node_num_2])
    L_flatten    = tf.concat([L_flatten_0, L_flatten_1, L_flatten_2], 1, name='concat')
    node_num     = node_num_0 + node_num_1 + node_num_2
    W_fcl1       = tf.get_variable("W_fcl1", shape=[node_num, node_1])
    B_fcl1       = tf.get_variable("B_fcl1", shape=[node_1])
    L_fcl1_pre   = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
    L_fcl1       = tf.nn.relu(L_fcl1_pre)
    L_fcl1_drop  = tf.layers.dropout(L_fcl1, 0.3, is_training)

with tf.variable_scope('Fully_Connected_Layer2'):
    W_fcl2  = tf.get_variable("W_fcl2", shape=[node_1, node_2])
    B_fcl2  = tf.get_variable("B_fcl2", shape=[node_2])
    L_fcl2_pre   = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
    L_fcl2  = tf.nn.relu(L_fcl2_pre)
    L_fcl2_drop  = tf.layers.dropout(L_fcl2, 0.3, is_training)
            
with tf.variable_scope('Output_Layer'):
    W_out  = tf.get_variable("W_out", shape=[node_2, 1])#, initializer=tf.contrib.layers.xavier_initializer())
    B_out  = tf.get_variable("B_out", shape=[1])#, initializer=tf.contrib.layers.xavier_initializer())
    outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

# Define loss function and optimizer
obj_loss    = tf.reduce_mean(tf.square(targets - outputs))
optimizer   = tf.train.AdamOptimizer(l_rate).minimize(obj_loss)
#def end: def __init__
#class end: DeepCas9


	
def train_DeepCas9(trainData,trainDataAll,testDataAll,ENZ): 
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	for e in range(110): # for e in range(60): 
		for featureData,targetLabel in data_list_batch_1_30_4(trainData):
			sess.run(optimizer, feed_dict={inputs_sg: featureData, y_:targetLabel})
			
		targetLabelList = []
		outputList = []
		trainMSEList = []
		trainMSEloss = 0
		
		for featureData,targetLabel in trainDataAll:
			output = outputs.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
			trainMSEloss_temp = obj_loss.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
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
		output = outputs.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
		testMSEloss = obj_loss.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
		targetLabel_st = targetLabel.squeeze().tolist()
		output_cdnst = output.squeeze().tolist()
		test_spcc, _ = stats.spearmanr(targetLabel_st, output_cdnst)
		
		print("epch {0}: train_loss {1}, train_spcc {2} || test_loss {3}, test_spcc {4}".format(e,trainMSEloss,train_spcc,testMSEloss,test_spcc))
