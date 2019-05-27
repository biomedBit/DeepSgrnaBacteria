import random


def cv_split(flieName,ENZ,SET):
	f = open(flieName, "r")
	line = f.readline()
	dataset = []
	while(line):
		dataset.append(line)
		line = f.readline()
	f.close()
	random.shuffle(dataset)
	cv_size = int(len(dataset)/5) + 1
	
	part1 = dataset[0*cv_size:1*cv_size]
	part2 = dataset[1*cv_size:2*cv_size]
	part3 = dataset[2*cv_size:3*cv_size]
	part4 = dataset[3*cv_size:4*cv_size]
	part5 = dataset[4*cv_size:5*cv_size]
	
	ftrain = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "1" + "/traindata.txt", "w")
	ftest = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "1" + "/testdata.txt", "w")
	for i in part1:
		ftest.write(i)
	for i in part2+part3+part4+part5:
		ftrain.write(i)
	ftrain.close()
	ftest.close()
	
	ftrain = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "2" + "/traindata.txt", "w")
	ftest = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "2" + "/testdata.txt", "w")
	for i in part2:
		ftest.write(i)
	for i in part1+part3+part4+part5:
		ftrain.write(i)
	ftrain.close()
	ftest.close()
	
	ftrain = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "3" + "/traindata.txt", "w")
	ftest = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "3" + "/testdata.txt", "w")
	for i in part3:
		ftest.write(i)
	for i in part1+part2+part4+part5:
		ftrain.write(i)
	ftrain.close()
	ftest.close()
	
	ftrain = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "4" + "/traindata.txt", "w")
	ftest = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "4" + "/testdata.txt", "w")
	for i in part4:
		ftest.write(i)
	for i in part1+part2+part3+part5:
		ftrain.write(i)
	ftrain.close()
	ftest.close()
	
	ftrain = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "5" + "/traindata.txt", "w")
	ftest = open("../Cross_validation/" + SET + "/" + ENZ + "/" + "5" + "/testdata.txt", "w")
	for i in part5:
		ftest.write(i)
	for i in part1+part2+part3+part4:
		ftrain.write(i)
	ftrain.close()
	ftest.close()

cv_split("Cas9_Set1.txt","Cas9","Set1")
cv_split("Cas9_Set2.txt","Cas9","Set2")
cv_split("eSpCas9_Set1.txt","eSpCas9","Set1")
cv_split("eSpCas9_Set2.txt","eSpCas9","Set2")
cv_split("knoRecA_Cas9_Set1.txt","knoRecA_Cas9","Set1")
cv_split("knoRecA_Cas9_Set2.txt","knoRecA_Cas9","Set2")
