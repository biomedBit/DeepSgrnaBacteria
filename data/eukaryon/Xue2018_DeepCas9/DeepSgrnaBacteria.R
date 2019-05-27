library(mxnet)
library(seqinr)
library(stringr)
DeepCas9_293T_model<-mx.model.load("DeepCas9_293T",iteration = 250)
DeepCas9_HL60_model<-mx.model.load("DeepCas9_HL60",iteration = 250)
DeepCas9_mEL4_model<-mx.model.load("DeepCas9_mEL4",iteration = 250)
source("encodeOntargetSeq.R")
source("DeepCas9_scores.R")

dataList = list(
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
"Labuhn2018Hs_210.csv",
"Labuhn2018Ms_214.csv",
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
)

for(dataName in dataList){
	encodeOntargetSeq(paste(dataName,".input",sep=""),dataName)
	DeepCas9_scores(paste(dataName,"one-hot.csv",sep="_"),dataName)
}

