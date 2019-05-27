import math
import numpy as np
import random
from tqdm import tqdm

BZ = 128

def get_numpy_4_43_1(linelist):
	x_on_target = np.zeros((1,4,43,1))
	for i, base in	enumerate(linelist[3]):
		if base == 'A':
			x_on_target[0,0,i,0] = 1
		elif base == 'C':
			x_on_target[0,1,i,0] = 1
		elif base == 'G':
			x_on_target[0,2,i,0] = 1
		elif base == 'T':
			x_on_target[0,3,i,0] = 1
		else:
			raise(Exception())
				
	return (x_on_target,np.array([[float(linelist[1])]]))


def get_numpy_1_23_4(linelist):
	x_on_target = np.zeros((1,1,23,4))
	for i, base in	enumerate(linelist[3][10:-10]):
		if base == 'A':
			x_on_target[0,0,i,0] = 1
		elif base == 'C':
			x_on_target[0,0,i,1] = 1
		elif base == 'G':
			x_on_target[0,0,i,2] = 1
		elif base == 'T':
			x_on_target[0,0,i,3] = 1
		else:
			raise(Exception())				
	return (x_on_target,np.array([[float(linelist[1])]]))

def get_numpy_1_30_4(linelist):
	x_on_target = np.zeros((1,1,30,4))
	for i, base in	enumerate(linelist[3][6:-7]):
		if base == 'A':
			x_on_target[0,0,i,0] = 1
		elif base == 'C':
			x_on_target[0,0,i,1] = 1
		elif base == 'G':
			x_on_target[0,0,i,2] = 1
		elif base == 'T':
			x_on_target[0,0,i,3] = 1
		else:
			raise(Exception())				
	return (x_on_target,np.array([[float(linelist[1])]]))


def get_numpy_23_4_1(linelist):
	x_on_target = np.zeros((1,23,4,1))
	for i, base in	enumerate(linelist[3][10:-10]):
		if base == 'A':
			x_on_target[0,i,0,0] = 1
		elif base == 'C':
			x_on_target[0,i,1,0] = 1
		elif base == 'G':
			x_on_target[0,i,2,0] = 1
		elif base == 'T':
			x_on_target[0,i,3,0] = 1
		else:
			raise(Exception())				
	return (x_on_target,np.array([[float(linelist[1])]]))

def createBatch_4_43_1(batchList,All_bool=False):
	featureDataBatch = np.zeros((1,4,43,1))
	targetLabelBatch = np.zeros((1,1))
	if All_bool == False:
		for featureData,targetLabel in batchList:
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	if All_bool == True:
		for featureData,targetLabel in tqdm(batchList):
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	return tuple((featureDataBatch[1:,:,:,:],targetLabelBatch[1:,:]))	

def createBatch_1_23_4(batchList,All_bool=False):
	featureDataBatch = np.zeros((1,1,23,4))
	targetLabelBatch = np.zeros((1,1))
	if All_bool == False:
		for featureData,targetLabel in batchList:
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	if All_bool == True:
		for featureData,targetLabel in tqdm(batchList):
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	return tuple((featureDataBatch[1:,:,:,:],targetLabelBatch[1:,:]))

def createBatch_1_30_4(batchList,All_bool=False):
	featureDataBatch = np.zeros((1,1,30,4))
	targetLabelBatch = np.zeros((1,1))
	if All_bool == False:
		for featureData,targetLabel in batchList:
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	if All_bool == True:
		for featureData,targetLabel in tqdm(batchList):
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	return tuple((featureDataBatch[1:,:,:,:],targetLabelBatch[1:,:]))
	
def createBatch_23_4_1(batchList,All_bool=False):
	featureDataBatch = np.zeros((1,23,4,1))
	targetLabelBatch = np.zeros((1,1))
	if All_bool == False:
		for featureData,targetLabel in batchList:
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	if All_bool == True:
		for featureData,targetLabel in tqdm(batchList):
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	return tuple((featureDataBatch[1:,:,:,:],targetLabelBatch[1:,:]))

def data_list_batch_4_43_1(trainData,batch_size=BZ):
	random.shuffle(trainData)
	batchNum = math.ceil(float(len(trainData)) / float(batch_size)) 
	dataListBatch = []
	for i in range(int(batchNum)):
		dataListBatch.append(createBatch_4_43_1(trainData[i*batch_size:(i+1)*batch_size]))		
	return dataListBatch	
	
def data_list_batch_1_23_4(trainData,batch_size=BZ):
	random.shuffle(trainData)
	batchNum = math.ceil(float(len(trainData)) / float(batch_size)) 
	dataListBatch = []
	for i in range(int(batchNum)):
		dataListBatch.append(createBatch_1_23_4(trainData[i*batch_size:(i+1)*batch_size]))		
	return dataListBatch

def data_list_batch_1_30_4(trainData,batch_size=BZ):
	random.shuffle(trainData)
	batchNum = math.ceil(float(len(trainData)) / float(batch_size)) 
	dataListBatch = []
	for i in range(int(batchNum)):
		dataListBatch.append(createBatch_1_30_4(trainData[i*batch_size:(i+1)*batch_size]))		
	return dataListBatch

def data_list_batch_23_4_1(trainData,batch_size=BZ):
	random.shuffle(trainData)
	batchNum = math.ceil(float(len(trainData)) / float(batch_size)) 
	dataListBatch = []
	for i in range(int(batchNum)):
		dataListBatch.append(createBatch_23_4_1(trainData[i*batch_size:(i+1)*batch_size]))		
	return dataListBatch

def data_list_all_1_30_4(trainDataTemp,num):
    random.shuffle(trainDataTemp)
    if num == 1:
    	return createBatch_1_30_4(trainDataTemp,All_bool=True)
    batch_size = int(float(len(trainDataTemp) / float(num)))
    outputList = []
    for i in range(num+1):
    	outputList.append(createBatch_1_30_4(trainDataTemp[i*batch_size:(i+1)*batch_size],All_bool=True))
    	if (i+1)*batch_size > len(trainDataTemp) - 1 :
    		return outputList
    return outputList

def data_list_all_4_43_1(trainDataTemp,num):
    random.shuffle(trainDataTemp)
    if num == 1:
    	return createBatch_4_43_1(trainDataTemp,All_bool=True)
    batch_size = int(float(len(trainDataTemp) / float(num)))
    outputList = []
    for i in range(num+1):
    	outputList.append(createBatch_4_43_1(trainDataTemp[i*batch_size:(i+1)*batch_size],All_bool=True))
    	if (i+1)*batch_size > len(trainDataTemp) - 1 :
    		return outputList
    return outputList

def data_list_all_1_23_4(trainDataTemp,num):
    random.shuffle(trainDataTemp)
    if num == 1:
    	return createBatch_1_23_4(trainDataTemp,All_bool=True)
    batch_size = int(float(len(trainDataTemp) / float(num)))
    outputList = []
    for i in range(num+1):
    	outputList.append(createBatch_1_23_4(trainDataTemp[i*batch_size:(i+1)*batch_size],All_bool=True))
    	if (i+1)*batch_size > len(trainDataTemp) - 1 :
    		return outputList
    return outputList
    
def data_list_all_23_4_1(trainDataTemp,num):
    random.shuffle(trainDataTemp)
    if num == 1:
    	return createBatch_23_4_1(trainDataTemp,All_bool=True)
    batch_size = int(float(len(trainDataTemp) / float(num)))
    outputList = []
    batch_size
    for i in range(num+1):
    	outputList.append(createBatch_23_4_1(trainDataTemp[i*batch_size:(i+1)*batch_size],All_bool=True))
    	if (i+1)*batch_size > len(trainDataTemp) - 1 :
    		return outputList
    return outputList

####################################################################


def get_numpy_e(linelist):
	x_on_target = np.zeros((1,4,43,1))
	for i, base in	enumerate(linelist[0]):
		if base == 'A':
			x_on_target[0,0,i,0] = 1
		elif base == 'C':
			x_on_target[0,1,i,0] = 1
		elif base == 'G':
			x_on_target[0,2,i,0] = 1
		elif base == 'T':
			x_on_target[0,3,i,0] = 1
		else:
			raise(Exception())				
	return (x_on_target,np.array([[float(linelist[1])]]))

def createBatch_e(batchList,All_bool=False):
	featureDataBatch = np.zeros((1,4,43,1))
	targetLabelBatch = np.zeros((1,1))
	if All_bool == False:
		for featureData,targetLabel in batchList:
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	if All_bool == True:
		for featureData,targetLabel in tqdm(batchList):
			featureDataBatch = np.concatenate((featureDataBatch, featureData), axis=0)
			targetLabelBatch = np.concatenate((targetLabelBatch, targetLabel), axis=0)
	return tuple((featureDataBatch[1:,:,:,:],targetLabelBatch[1:]))	



def data_list_batch_e(trainData,batch_size=BZ):
	random.shuffle(trainData)
	batchNum = math.ceil(float(len(trainData)) / float(batch_size)) 
	dataListBatch = []
	for i in range(int(batchNum)):
		dataListBatch.append(createBatch_e(trainData[i*batch_size:(i+1)*batch_size]))		
	return dataListBatch	
	


def data_list_all_e(trainDataTemp,num):
    random.shuffle(trainDataTemp)
    if num == 1:
    	return createBatch_e(trainDataTemp,All_bool=True)
    batch_size = int(float(len(trainDataTemp) / float(num)))
    outputList = []
    for i in tqdm(range(num+1)):
    	outputList.append(createBatch_e(trainDataTemp[i*batch_size:(i+1)*batch_size],All_bool=False))
    	if (i+1)*batch_size > len(trainDataTemp) - 1 :
    		return outputList
    return outputList

