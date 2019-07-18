from __future__ import division
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import math
import torch
import numpy as np
import random
import sys
from tqdm import tqdm


def get_numpy_4_43_1(seq43nt):
	x_on_target = np.zeros((1,4,43,1))
	for i, base in	enumerate(seq43nt):
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
				
	return x_on_target


def get_numpy_4_43_1_r(seq43nt,rposition,rbase):
	x_on_target = np.zeros((1,4,43,1))
	for i, base in	enumerate(seq43nt):
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
	
	x_on_target[0,0,rposition,0] = 0
	x_on_target[0,1,rposition,0] = 0
	x_on_target[0,2,rposition,0] = 0
	x_on_target[0,3,rposition,0] = 0
	
	if rbase == 'A':
		x_on_target[0,0,rposition,0] = 1
	elif rbase == 'C':
		x_on_target[0,1,rposition,0] = 1
	elif rbase == 'G':
		x_on_target[0,2,rposition,0] = 1
	elif rbase == 'T':
		x_on_target[0,3,rposition,0] = 1
	else:
		raise(Exception())
					
	return x_on_target


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

        self.x1 = Conv(120, 120, 3, 1)

    def forward(self, x):
        out1 = self.x1(x)
        return out1

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


flieName = sys.argv[1]
#../data/mediate_processing/Cas9_Set1.txt
#../data/mediate_processing/eSpCas9_Set1.txt
#../data/eukaryon/mytrainset/trainset
modelName = sys.argv[2]
#../saved_models/Cas9.pkl
#../saved_models/eSpCas9.pkl
#../saved_models/Eukaryon.pkl

net = CNN_5()
net.load_state_dict(torch.load(modelName))
net.cuda()	 
net = net.eval()
raw_sumv = 0
	
def predictor(position,base):
	sumv = 0
	f = open(flieName,"r")
	lines = f.readlines()
	for line in tqdm(lines):
		seq43 = line.split("\t")[3]
		asample = get_numpy_4_43_1_r(seq43,position,base)
		out = net(Variable(torch.from_numpy(asample).type(torch.FloatTensor).cuda())).cpu().data.numpy()[0][0]
		sumv = sumv + out
	dis = sumv - raw_sumv
	print(str(position) + "\t" + base + "\t" + str(dis))
	f.close()


 
f = open(flieName,"r")
lines = f.readlines()
for line in tqdm(lines):
	seq43 = line.split("\t")[3]
	asample = get_numpy_4_43_1(seq43)
	out = net(Variable(torch.from_numpy(asample).type(torch.FloatTensor).cuda())).cpu().data.numpy()[0][0]
	raw_sumv = raw_sumv + out
print("raw:\t"+str(raw_sumv))
f.close()
	
for position in range(0,43):
	for base in ["A","T","G","C"]:
		predictor(position,base)



	

