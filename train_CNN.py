from __future__ import division
from net.CNN_Lin import train_CNN_Lin
from net.DeepCRISPR import train_DeepCRISPR
from net.CNN import train_CNN,train_CNN_e,CNN_2,CNN_3,CNN_4,CNN_5,CNN_6,CNN_7
from net.utils import get_numpy_4_43_1,get_numpy_23_4_1,get_numpy_1_30_4,get_numpy_1_23_4,data_list_all_4_43_1,data_list_all_23_4_1,data_list_all_1_23_4,data_list_all_1_30_4,get_numpy_e,createBatch_e,data_list_batch_e,data_list_all_e
from net.DeepCas9 import train_DeepCas9
import sys
import math
import torch
import numpy as np
import random
from tqdm import tqdm

if sys.argv[1] == "Eukaryon":
	trainData = []
	with open('./data/eukaryon/mytrainset/trainset','r') as f:
		lineAll = f.readlines()
	random.shuffle(lineAll)
	for line in tqdm(lineAll):
		linelist = line.strip('\n').split("\t")
		asample = get_numpy_e(linelist)		
		trainData.append(asample)
					
	trainDataAll =  data_list_all_e(trainData,num=2)	
	net = CNN_5()
	net = net.cuda()
	train_CNN_e(net,trainData,trainDataAll)
	exit()


SET = sys.argv[1] # Set1;Set2
ENZ = sys.argv[2] # Cas9;eSpCas9;knoRecA_Cas9
NET = sys.argv[3] # CNN_Lin;DeepCRISPR;DeepCas9;CNN_2;CNN_3;CNN_4;CNN_5;CNN_6;CNN_7
cv = sys.argv[4] # Number of cross_validation
BZ = 128 # size of a batch 

path = './data/Cross_validation/' + SET + "/" + ENZ + "/" + cv + "/"
	
if NET == "CNN_Lin":

	trainData = []
	testData = []
	with open(path + "traindata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_23_4_1(linelist)		
			trainData.append(asample)
			line = f.readline()
		
	with open(path + "testdata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_23_4_1(linelist)		
			testData.append(asample)
			line = f.readline()
			
	trainDataAll =  data_list_all_23_4_1(trainData,num=4)	
	testDataAll = data_list_all_23_4_1(testData,num=1)
	print(len(trainData))
	train_CNN_Lin(trainData,trainDataAll,testDataAll,ENZ)
	
elif NET == "DeepCRISPR":

	trainData = []
	testData = []

	with open(path + "traindata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_1_23_4(linelist)		
			trainData.append(asample)
			line = f.readline()
		
	with open(path + "testdata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_1_23_4(linelist)		
			testData.append(asample)
			line = f.readline()
			
	trainDataAll =  data_list_all_1_23_4(trainData,num=4)	
	testDataAll = data_list_all_1_23_4(testData,num=1)

	train_DeepCRISPR(trainData,trainDataAll,testDataAll,ENZ)
	
elif NET == "DeepCas9":

	trainData = []
	testData = []

	with open(path + "traindata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_1_30_4(linelist)		
			trainData.append(asample)
			line = f.readline()
		
	with open(path + "testdata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_1_30_4(linelist)		
			testData.append(asample)
			line = f.readline()
			
	trainDataAll =  data_list_all_1_30_4(trainData,num=4)	
	testDataAll = data_list_all_1_30_4(testData,num=1)

	train_DeepCas9(trainData,trainDataAll,testDataAll,ENZ)


elif NET[:3] == "CNN":
	trainData = []
	testData = []
	with open(path + "traindata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_4_43_1(linelist)		
			trainData.append(asample)
			line = f.readline()
		
	with open(path + "testdata.txt",'r') as f:
		line = f.readline()
		while(line):
			linelist = line.strip('\n').split("\t")
			asample = get_numpy_4_43_1(linelist)		
			testData.append(asample)
			line = f.readline()
		
	trainDataAll =  data_list_all_4_43_1(trainData,num=4)	
	testDataAll = data_list_all_4_43_1(testData,num=1)

	if NET == "CNN_2":
		net = CNN_2()
	if NET == "CNN_3":
		net = CNN_3()
	if NET == "CNN_4":
		net = CNN_4()
	if NET == "CNN_5":
		net = CNN_5()
	if NET == "CNN_6":
		net = CNN_6()
	if NET == "CNN_7":
		net = CNN_7()	
	net = net.cuda()
	train_CNN(net,trainData,trainDataAll,testDataAll,SET,ENZ,path)

else:
	print("<NET>: parameter exception!")
