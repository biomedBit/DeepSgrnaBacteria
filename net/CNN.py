from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch
import numpy as np
import random
from tqdm import tqdm
import scipy.stats as stats
from utils import data_list_batch_4_43_1,data_list_batch_e

BZ=128

def initNetParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            
class Conv(nn.Module):
    def __init__(self, inc, outc, ksize, psize):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size = (ksize, 1), stride = (1, 1), padding = (psize, 0)),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out
        
        
class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()

        self.x1 = Conv(4, 120, 3, 1)

    def forward(self, x):
        out1 = self.x1(x)
        return out1


class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()

        self.x1 = Conv(120,120,3, 1)

    def forward(self, x):
        out1 = self.x1(x)
        return out1


#######################
#######################

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
   
        
        self.r6 = nn.Linear(1200, 1200)
        self.r7 = nn.BatchNorm1d(1200)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(1200, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
 
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)

        return out

#######################
#######################


class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.r6 = nn.Linear(600, 600)
        self.r7 = nn.BatchNorm1d(600)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(600, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)
        
        #out = self.a4(out)
        #out = out.view(out.size()[0],-1)

        return out        



#######################
#######################
        
class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.a3 = ConvBlock2()
        
        self.p3 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d3 = nn.Dropout(0.3)
        
        #self.a4 = FCN_last()
        
        self.r6 = nn.Linear(240, 240)
        self.r7 = nn.BatchNorm1d(240)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(240, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = self.a3(out)
        out = self.p3(out)
        out = self.d3(out)
        
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)
        
        #out = self.a4(out)
        #out = out.view(out.size()[0],-1)

        return out        

#######################
#######################

class CNN_5(nn.Module):
    def __init__(self):
        super(CNN_5, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.a3 = ConvBlock2()
        
        self.p3 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d3 = nn.Dropout(0.3)
        
        self.a4 = ConvBlock2()
        
        self.p4 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d4 = nn.Dropout(0.3)
        
        self.r6 = nn.Linear(120, 120)
        self.r7 = nn.BatchNorm1d(120)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(120, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = self.a3(out)
        out = self.p3(out)
        out = self.d3(out)
        
        
        #print(out.size())
        
        out = self.a4(out)
        out = self.p4(out)
        out = self.d4(out)
        
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)
        
        #out = self.a4(out)
        #out = out.view(out.size()[0],-1)

        return out

#######################
#######################

class CNN_6(nn.Module):
    def __init__(self):
        super(CNN_6, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.a3 = ConvBlock2()
        
        self.p3 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d3 = nn.Dropout(0.3)
        
        self.a4 = ConvBlock2()
        self.a5 = ConvBlock2()
        
        self.p4 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d4 = nn.Dropout(0.3)
        
        
        
        self.r6 = nn.Linear(120, 120)
        self.r7 = nn.BatchNorm1d(120)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(120, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = self.a3(out)
        out = self.p3(out)
        out = self.d3(out)
        
        
        #print(out.size())
        
        out = self.a4(out)
        out = self.a5(out)
        out = self.p4(out)
        out = self.d3(out)
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)

        return out


#######################
#######################

class CNN_7(nn.Module):
    def __init__(self):
        super(CNN_7, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.a3 = ConvBlock2()
        self.a4 = ConvBlock2()
        self.p3 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d3 = nn.Dropout(0.3)
        
        self.a5 = ConvBlock2()
        self.a6 = ConvBlock2()
        
        self.p4 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d4 = nn.Dropout(0.3)
        
        
        
        self.r6 = nn.Linear(120, 120)
        self.r7 = nn.BatchNorm1d(120)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(120, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = self.a3(out)
        out = self.a4(out)
        out = self.p3(out)
        out = self.d3(out)
        
        
        #print(out.size())
        
        out = self.a5(out)
        out = self.a6(out)
        out = self.p4(out)
        out = self.d3(out)
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)

        return out

###########################
###########################
class CNN_8(nn.Module):
    def __init__(self):
        super(CNN_8, self).__init__()
        self.a0 = ConvBlock1()
        
        self.p0 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d0 = nn.Dropout(0.3)        
        
        self.a1 = ConvBlock2()
        
        self.p1 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d1 = nn.Dropout(0.3)
        
        
        self.a2 = ConvBlock2()   
        
        self.p2 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d2 = nn.Dropout(0.3)
        
        self.a3 = ConvBlock2()
        self.a4 = ConvBlock2()
        self.p3 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d3 = nn.Dropout(0.3)
        
        self.a5 = ConvBlock2()
        self.a6 = ConvBlock2()
        
        self.p4 = nn.MaxPool2d((2, 1),(2,1),(0,0))
        self.d4 = nn.Dropout(0.3)
        
        
        
        self.r6 = nn.Linear(120, 120)
        self.r7 = nn.BatchNorm1d(120)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)
        
        self.r10 = nn.Linear(120, 1)


    def forward(self,x):
    	
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        #print(out.size())
        
        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)
        
        #print(out.size())

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)
        
        #print(out.size())
        
        out = self.a3(out)
        out = self.a4(out)
        out = self.p3(out)
        out = self.d3(out)
        
        
        #print(out.size())
        
        out = self.a5(out)
        out = self.a6(out)
        out = self.p4(out)
        out = self.d3(out)
        
        #print(out.size())
        
        out = out.view(out.size()[0],-1)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)

        return out


def train_CNN(net,trainData,trainDataAll,testDataAll,Set,Enz,path):
	net = net.cuda() 
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.999), eps=1e-08)       
	for e in range(110): #for e in range(60): 
		net = net.train()
		for featureData,targetLabel in data_list_batch_4_43_1(trainData):
			output = net(Variable(torch.from_numpy(featureData).type(torch.FloatTensor).cuda()))
			MSEloss = criterion(output,Variable(torch.from_numpy(targetLabel).type(torch.FloatTensor).cuda()))
			optimizer.zero_grad()
			MSEloss.backward()
			optimizer.step()
		
		net = net.eval()
	
		targetLabelList = []
		outputList = []
		trainMSEList = []
		trainMSEloss = 0
		
		for featureData,targetLabel in trainDataAll:
			output = net(Variable(torch.from_numpy(featureData).type(torch.FloatTensor).cuda()))
			trainMSEList.append(criterion(output,Variable(torch.from_numpy(targetLabel).type(torch.FloatTensor).cuda())).cpu().data.numpy()[0])
			targetLabel_temp = targetLabel.squeeze().tolist()
			output_temp = output.cpu().data.numpy().squeeze().tolist()
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
		output = net(Variable(torch.from_numpy(featureData).type(torch.FloatTensor).cuda()))
		testMSEloss = criterion(output,Variable(torch.from_numpy(targetLabel).type(torch.FloatTensor).cuda())).cpu().data.numpy()[0]
		targetLabel_st = targetLabel.squeeze().tolist()
		output_cdnst = output.cpu().data.numpy().squeeze().tolist()

		test_spcc, _ = stats.spearmanr(targetLabel_st, output_cdnst)
	
		
		print("epch {0}: train_loss {1}, train_spcc {2} || test_loss {3}, test_spcc {4}".format(e,trainMSEloss,train_spcc,testMSEloss,test_spcc))
		
		
		#stopEpch = 20
		#stopEpch = 30
		stopEpch = 40
		
		if e == stopEpch:
			torch.save(net.state_dict(), path + "/" + str(e)  + "_" + str(train_spcc)[:7] + "_" + str(test_spcc)[:7] + ".pkl")
			
def train_CNN_e(net,trainData,trainDataAll):
	initNetParams(net)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.999), eps=1e-08)       
	for e in range(75):
		net = net.train()
		for featureData,targetLabel in data_list_batch_e(trainData):
			output = net(Variable(torch.from_numpy(featureData).type(torch.FloatTensor).cuda()))
			MSEloss = criterion(output,Variable(torch.from_numpy(targetLabel).type(torch.FloatTensor).cuda()))
			optimizer.zero_grad()
			MSEloss.backward()
			optimizer.step()
		
		net = net.eval()
	
		targetLabelList = []
		outputList = []
		trainMSEList = []
		trainMSEloss = 0
		for featureData,targetLabel in trainDataAll:
			output = net(Variable(torch.from_numpy(featureData).type(torch.FloatTensor).cuda()))
			trainMSEList.append(criterion(output,Variable(torch.from_numpy(targetLabel).type(torch.FloatTensor).cuda())).cpu().data.numpy())
			targetLabel_temp = targetLabel.squeeze().tolist()
			output_temp = output.cpu().data.numpy().squeeze().tolist()
			if not(type(output_temp) == list):
				output_temp = [output_temp]
			if not(type(targetLabel_temp) == list):
				targetLabel_temp = [targetLabel_temp]
			targetLabelList = targetLabelList + targetLabel_temp
			outputList = outputList + output_temp
			
		trainMSEloss = sum(trainMSEList) / len(trainMSEList)
		train_spcc, _ = stats.spearmanr(targetLabelList, outputList) #spearmar
		print("{0},{1},{2},".format(e,trainMSEloss[0],train_spcc))
		
	torch.save(net.state_dict(),"./saved_models/" + "Eukaryon.pkl")
