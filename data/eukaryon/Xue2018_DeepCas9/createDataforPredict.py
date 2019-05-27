mylist1 = [
"chari2015Train293T_1234.csv",
"chari2015Valid293T_10.csv",
"chari2015ValidA549_10.csv",
"chari2015ValidHepG2_10.csv",
"chari2015ValidPGP1iPS_10.csv",
"chari2015ValidSKNAS_10.csv",
"chari2015ValidU2OS_10.csv",
"concordet2-Hs_26.csv",
"concordet2-Mm_18.csv",
"doench2016_2333.csv",
"farboud2015_50.csv",
"gandhi2016-ci2_72.csv",
"hart2016-GbmAvg_4272.csv",
"hart2016-Hct1162lib1Avg_4239.csv",
"hart2016-Hct1162lib2Avg_3617.csv",
"hart2016-HelaLib1Avg_4256.csv",
"hart2016-HelaLib2Avg_3845.csv",
"hart2016-Rpe1Avg_4214.csv",
"housden2015_75.csv",
"liu2016-mm9_205.csv",
"ren2015_39.csv",
"schoenigHs_3.csv",
"schoenigMm_6.csv",
"schoenigRn_15.csv",
"wang2015hg19_2921.csv",
"xu2015_35.csv",
"xu2015TrainHl60_2076.csv",
"xu2015TrainKbm7_2076.csv",
"xu2015TrainMEsc_981.csv"
]

for i in mylist1:
	fr = open("../../data/U6/" + i,"r")
	fw = open("./data/" + i + ".input","w")
	line = fr.readline()
	line = fr.readline()
	while(line):
		if line == "\n":
			continue
		lineList = line.split(",")
		print(lineList)
		fw.write(">" + lineList[3] + "\n")
		fw.write(lineList[6][26:56] + "\n")
		line = fr.readline()
	fr.close()
	fw.close()
		
mylist2 = [
"doench2014Hs_881.csv",
"doench2014Mm_951.csv"
]

for i in mylist2:
	fr = open("../U6/" + i,"r")
	fw = open("./data/" + i + ".input","w")
	line = fr.readline()
	line = fr.readline()
	while(line):
		if line == "\n":
			continue
		lineList = line.split(",")
		fw.write(">" + lineList[3] + "\t" + lineList[1] + "\n")
		fw.write(lineList[6][26:56] + "\n")
		line = fr.readline()
	fr.close()
	fw.close()

mylist3 = [
"doench2014HsA375_1276.csv"
]

for i in mylist3:
	fr = open("../U6/" + i,"r")
	fw = open("./data/" + i +".input","w")
	line = fr.readline()
	line = fr.readline()
	while(line):
		if line == "\n":
			continue
		lineList = line.split(",")
		fw.write(">" + lineList[2] + "\t" + lineList[0] + "\n")
		fw.write(lineList[3] + "\n")
		line = fr.readline()
	fr.close()
	fw.close()

 	

