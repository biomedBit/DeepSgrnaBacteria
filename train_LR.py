from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.stats as stats

# t,p,f,c	
ENZList = ["Cas9"] # ["Cas9", "eSpCas9", "knoRecA_Cas9"]
Features = ["t","p", "f", "c","t_p","t_f", "t_c", "p_f","p_c","f_c","t_p_f","t_p_c","t_f_c","p_f_c","t_p_f_c","p4_f1","p4_c","f1_c","p4_f1_c","t_p4_f1_c",]
SETList = ["Set1"] # ["Set1","Set2"]
    
def create_npdata(path,features,flage="False"):
	l = []
	t1 = []
	t2 = []
	t3 = []
	t4 = []
	t5 = []
	t6 = []
	p1 = []
	p2 = []
	p3 = []
	p4 = []
	f1 = []
	f2 = []
	f3 = []
	f4 = []
	c = []
	
	train_num = 0
	test_num = 0
	with open(path + "traindata.txt","r") as f:
		line = f.readline()
		while(line):
			train_num = train_num + 1
			line = line.replace("(","")
			line = line.replace("[","")
			linelist = line.strip("\n").split("\t")
			l.append(float(linelist[1]))
			t1.append(float(linelist[4]))
			t2.append(float(linelist[5]))
			t3.append(float(linelist[6]))
			t4.append(float(linelist[7]))
			t5.append(float(linelist[8]))
			t6.append(float(linelist[9]))
			p1.append(float(linelist[10]))
			p2.append(float(linelist[11]))
			p3.append(float(linelist[12]))
			p4.append(float(linelist[13]))
			f1.append(float(linelist[14]))
			f2.append(float(linelist[15]))
			f3.append(float(linelist[16]))
			f4.append(float(linelist[17]))
			line = f.readline()
	with open(path + "testdata.txt","r") as f:
		line = f.readline()
		while(line):
			test_num = test_num + 1
			line = line.replace("(","")
			line = line.replace("[","")
			linelist = line.strip("\n").split("\t")
			l.append(float(linelist[1]))
			t1.append(float(linelist[4]))
			t2.append(float(linelist[5]))
			t3.append(float(linelist[6]))
			t4.append(float(linelist[7]))
			t5.append(float(linelist[8]))
			t6.append(float(linelist[9]))
			p1.append(float(linelist[10]))
			p2.append(float(linelist[11]))
			p3.append(float(linelist[12]))
			p4.append(float(linelist[13]))
			f1.append(float(linelist[14]))
			f2.append(float(linelist[15]))
			f3.append(float(linelist[16]))
			f4.append(float(linelist[17]))
			line = f.readline()	

	with open(path + "traindata_result.txt","r") as f:
		line = f.readline()
		while(line):
			v = line.strip("\n")
			c.append(float(v))
			line = f.readline()
			
	with open(path + "testdata_result.txt","r") as f:
		line = f.readline()
		while(line):
			v = line.strip("\n")
			c.append(float(v))
			line = f.readline()
					
	total_num = train_num + test_num
	assert total_num == len(l) == len(t1) == len(t2) == len(t3) == len(t4) == len(t5) == len(t6) == len(p1) == len(p2) == len(p3) == len(p4) == len(f1) == len(f2) == len(f3) == len(f4) == len(c)

	if flage != "False":
		print("##############################")
		print("t1: ",stats.spearmanr(l,t1))
		print("t2: ",stats.spearmanr(l,t2))
		print("t3: ",stats.spearmanr(l,t3))
		print("t4: ",stats.spearmanr(l,t4))
		print("t5: ",stats.spearmanr(l,t5))
		print("t6: ",stats.spearmanr(l,t6))
		print("p1: ",stats.spearmanr(l,p1))
		print("p2: ",stats.spearmanr(l,p2))
		print("p3: ",stats.spearmanr(l,p3))
		print("p4: ",stats.spearmanr(l,p4))
		print("f1: ",stats.spearmanr(l,f1))
		print("f2: ",stats.spearmanr(l,f2))
		print("f3: ",stats.spearmanr(l,f3))
		print("f4: ",stats.spearmanr(l,f4))
		print("c: ",stats.spearmanr(l,c))
		print("##############################")
	
	Xlist = []
	ylist = l
	
	if features == "t":
		for i in range(total_num):
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i]])
	elif features == "p":
		for i in range(total_num):
			Xlist.append([p1[i],p2[i],p3[i],p4[i]])
	elif features == "f":
		for i in range(total_num):
			Xlist.append([f1[i],f2[i],f3[i],f4[i]])
	elif features == "c":
		for i in range(total_num):
			Xlist.append([c[i]])
	elif features == "t_p":
		for i in range(total_num):
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],p1[i],p2[i],p3[i],p4[i]])
	elif features == "t_f":
		for i in range(total_num):	
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],f1[i],f2[i],f3[i],f4[i]])
	elif features == "t_c":
		for i in range(total_num):
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],c[i]])
	elif features == "p_f":
		for i in range(total_num):
			Xlist.append([p1[i],p2[i],p3[i],p4[i],f1[i],f2[i],f3[i],f4[i]])
	elif features == "p_c":
		for i in range(total_num):
			Xlist.append([p1[i],p2[i],p3[i],p4[i],c[i]] )
	elif features == "f_c":
		for i in range(total_num):
			Xlist.append([f1[i],f2[i],f3[i],f4[i],c[i]] )
	elif features == "t_p_f":
		for i in range(total_num):	
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],p1[i],p2[i],p3[i],p4[i],f1[i],f2[i],f3[i],f4[i]] )
	elif features == "t_p_c":
		for i in range(total_num):
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],p1[i],p2[i],p3[i],p4[i],c[i]])
	elif features == "t_f_c":
		for i in range(total_num):
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],f1[i],f2[i],f3[i],f4[i],c[i]])
	elif features == "p_f_c":
		for i in range(total_num):
			Xlist.append([p1[i],p2[i],p3[i],p4[i],f1[i],f2[i],f3[i],f4[i],c[i]])
	elif features == "t_p_f_c":
		for i in range(total_num):	
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],p1[i],p2[i],p3[i],p4[i],f1[i],f2[i],f3[i],f4[i],c[i]])
	elif features == "p4_f1":
		for i in range(total_num):
			Xlist.append([p4[i],f1[i]])
	elif features == "p4_c":
		for i in range(total_num):	
			Xlist.append([p4[i],c[i]])
	elif features == "f1_c":
		for i in range(total_num):
			Xlist.append([f1[i],c[i]])
	elif features == "p4_f1_c":
		for i in range(total_num):	
			Xlist.append([p4[i],f1[i],c[i]])
	elif features == "t_p4_f1_c":
		for i in range(total_num):	
			Xlist.append([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],p4[i],f1[i],c[i]])
	else:
		print("Features exception!")
		assert 1==2
	return (np.array(Xlist), np.array(ylist), train_num)

def train_model(SET,ENZ,cv,modelName,flage):
	'''
	ENZ: Cas9, eSpCas9, eSpCas9
	modelName: LinearRegression, Ridge, Lasso, ElasticNet, SVR, GradientBoostingRegressor
	'''
	path = "./data/Cross_validation/" + SET + "/" +ENZ + "/" + cv + "/"    
	for features in Features:
		X, y, train_num = create_npdata(path,features,flage=flage)
		scaler = MinMaxScaler() 
		scaler.fit(X) 
		X=scaler.transform(X)
		X_train, y_train = X[:train_num], y[:train_num]
		X_test, y_test = X[train_num:], y[train_num:]
		if modelName == "LinearRegression":
			model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
		if modelName == "Ridge":
			model = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   						normalize=False, random_state=None, solver='auto', tol=0.001) # Parameter adjustment
		if modelName == "Lasso":
			model = Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   						normalize=False, positive=False, precompute=False, random_state=None,
   						selection='cyclic', tol=0.0001, warm_start=False) # Parameter adjustment
		if modelName == "ElasticNet":
			model = ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      						max_iter=1000, normalize=False, positive=False, precompute=False,
      						random_state=None, selection='cyclic', tol=0.0001, warm_start=False) # Parameter adjustment
		if modelName == "SVR":
			model = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  						kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False) # Parameter adjustment
		if modelName == "GradientBoostingRegressor":
			model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             						learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             						max_leaf_nodes=None, min_impurity_decrease=0.0,
             						min_impurity_split=None, min_samples_leaf=1,
             						min_samples_split=2, min_weight_fraction_leaf=0.0,
             						n_estimators=100, presort='auto', random_state=None,
             						subsample=1.0, verbose=0, warm_start=False) # Parameter adjustment
		model.fit(X_train,y_train)
		y = model.predict(X_train)
		train_spcc, _ = stats.spearmanr(y,y_train)
		yt = model.predict(X_test)
		test_spcc, m = stats.spearmanr(yt,y_test)
		print(ENZ,modelName,features,train_spcc,test_spcc)
	return train_spcc, test_spcc



#### LinearRegression

train_model("Set1","Cas9","1","LinearRegression",flage="ture")
train_model("Set1","Cas9","2","LinearRegression",flage="False")
train_model("Set1","Cas9","3","LinearRegression",flage="False")
train_model("Set1","Cas9","4","LinearRegression",flage="False")
train_model("Set1","Cas9","5","LinearRegression",flage="False")

train_model("Set1","eSpCas9","1","LinearRegression",flage="False")
train_model("Set1","eSpCas9","2","LinearRegression",flage="False")
train_model("Set1","eSpCas9","3","LinearRegression",flage="False")
train_model("Set1","eSpCas9","4","LinearRegression",flage="False")
train_model("Set1","eSpCas9","5","LinearRegression",flage="False")

train_model("Set1","knoRecA_Cas9","1","LinearRegression",flage="False")
train_model("Set1","knoRecA_Cas9","2","LinearRegression",flage="False")
train_model("Set1","knoRecA_Cas9","3","LinearRegression",flage="False")
train_model("Set1","knoRecA_Cas9","4","LinearRegression",flage="False")
train_model("Set1","knoRecA_Cas9","5","LinearRegression",flage="False")

'''
#### Ridge
for ENZ in ENZList:
	for feature in Features:
		train_model(ENZ,"Ridge",feature)
'''
'''
#### Lasso
for ENZ in ENZList:
	for feature in Features:
		train_model(ENZ,"Lasso",feature)
'''
'''
#### ElasticNet
for ENZ in ENZList:
	for feature in Features:
		train_model(ENZ,"ElasticNet",feature)
'''
'''
#### SVR
for ENZ in ENZList:
	for feature in Features:
		train_model(ENZ,"SVR",feature)
'''
'''		
#### GradientBoostingRegressor
for ENZ in ENZList:
	for feature in Features:
		train_model(ENZ,"GradientBoostingRegressor",feature)		
'''	
