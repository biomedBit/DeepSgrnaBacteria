trainList = []
ftrain = open("../trainset/trainset","r")
line = ftrain.readline()
while(line):
	seq23 = line.split("\t")[0][10:33]
	trainList.append(seq23)
	line = ftrain.readline()

processFile = "Labuhn2018Hs_210"
ftestr = open(processFile,"r")
ftestw = open(processFile + "_p","w")
line = ftestr.readline()
while(line):
	seq23 = line.split("\t")[0][10:33]
	if seq23 in trainList:
		line = ftestr.readline()
		continue
	ftestw.write(line)
	line = ftestr.readline()
