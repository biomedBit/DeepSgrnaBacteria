import tensorflow as tf
from deepcrispr import DCModelOntar
import numpy as np
import scipy.stats as stats

def get_numpy(seq):
	x_on_target = np.zeros((1,4,1,23))
	for i, base in	enumerate(seq):
		if base == 'A':
			x_on_target[0,0,0,i] = 1
		elif base == 'C':
			x_on_target[0,1,0,i] = 1
		elif base == 'G':
			x_on_target[0,2,0,i] = 1
		elif base == 'T':
			x_on_target[0,3,0,i] = 1
	return x_on_target

seq_feature_only = True
channels = 4 if seq_feature_only else 8

sess = tf.InteractiveSession()

on_target_model_dir = './trained_models/ontar_cnn_reg_seq'
# using regression model, otherwise classification model
is_reg = True
# using sequences feature only, otherwise sequences feature + selected epigenetic features
seq_feature_only = True
dcmodel = DCModelOntar(sess, on_target_model_dir, is_reg, seq_feature_only)

fw = open("./predict_results.txt","w")

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
"doench2014Hs_881.csv",
"doench2014HsA375_1276.csv",
"doench2014Mm_951.csv",
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
"xu2015TrainMEsc_981.csv",
]

for i in mylist1:
	reg_predi_list = []
	fr = open("./dataset/" + i ,"r")
	line = fr.readline()
	fw.write(str(i) + "\n")
	while(line):
		lineList = line.split("\t")
		seq23 = lineList[1][4:27] 
		x_on_target = get_numpy(seq23)	
		predicted_on_target = dcmodel.ontar_predict(x_on_target)
		predicted_on_target = np.squeeze(predicted_on_target).tolist()
		fw.write(str(predicted_on_target) + "\n")
		line = fr.readline()
	fw.write("\n")
	fr.close()

fw.close()


