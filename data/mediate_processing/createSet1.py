import random
import re
import os
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from tqdm import tqdm
import Bio.SeqUtils.MeltingTemp as Tm 
from tqdm import tqdm 


#os.system("cd ../cas-offinder-master/ ; bash DeepSgrnaBacteria.sh")		  
#os.system("cd ../ViennaRNA-2.4.11/ ; bash DeepSgrnaBacteria.sh")

seqRef = ""
with open("../Ref_genome/NC_000913.3/GCF_000005845.2_ASM584v2_genomic.fna",'r') as f:
	line = f.readline()
	line = f.readline()
	while(line):
		seqRef = seqRef + line.strip("\n")
		line = f.readline()

seqRefp = Seq(seqRef,IUPAC.unambiguous_dna)
seqRefn = seqRefp.reverse_complement()

seqRefp = str(seqRefp)
seqRefn = str(seqRefn)



def createDataset(enz):		

	fw = open("./" + enz + "_Set1_tmp.txt",'w')
	with open("../raw_data/" + enz + "_raw.txt",'r') as f:
		linelist = f.readlines()
		print("process\t"+enz)
		for line in tqdm(linelist):
			seq = line.strip('\n').split("\t")[2][:-1]
			posListP = [i.span() for i in re.finditer(seq+"[A,T,G,C]GG",seqRefp)]
			countp = len(posListP)
			posListN = [i.span() for i in re.finditer(seq+"[A,T,G,C]GG",seqRefn)]
			countn = len(posListN)

			if countp == 1 and countn == 0:
				locS,locE = posListP[0]
				
				fw.write(line.strip('\n')[:-1] +  "\t") 
				fw.write(str(seqRefp[locS-10:locE+10]) +"\t")
				fw.write(str((Tm.Tm_staluc(seq[:7])+31)/58.0) +  "\t")
				fw.write(str((Tm.Tm_staluc(seq[7:15])+19)/55.0) +  "\t")
				fw.write(str((Tm.Tm_staluc(seq[15:20])+64)/65.0)+ "\t")
				fw.write(str((Tm.Tm_staluc(seq)-36)/30) + "\t")
				fw.write(str((Tm.Tm_staluc(seqRefp[locS-5:locS])+64)/63.0) + "\t")
				fw.write(str((Tm.Tm_staluc(seqRefp[locE-3:locE+2])+67)/61.8) + "\n")

				
			elif countp == 0 and countn == 1:
				locS,locE = posListN[0]
				fw.write(line.strip('\n')[:-1] +  "\t") 
				fw.write(str(seqRefn[locS-10:locE+10]) +"\t")
				fw.write(str((Tm.Tm_staluc(seq[:7])+31)/58.0) +  "\t")
				fw.write(str((Tm.Tm_staluc(seq[7:15])+19)/55.0) +  "\t")
				fw.write(str((Tm.Tm_staluc(seq[15:20])+64)/65.0)+ "\t")
				fw.write(str((Tm.Tm_staluc(seq)-36)/30) + "\t")
				fw.write(str((Tm.Tm_staluc(seqRefn[locS-5:locS])+64)/63.0) + "\t")
				fw.write(str((Tm.Tm_staluc(seqRefn[locE-3:locE+2])+67)/61.8) + "\n")

			else:
				print(seq + " has more than 1 match!")
				extendSeqList = []
				for locS,locE in posListP:
					extendSeqList.append(seqRefp[locS-10:locE+10])
				for locS,locE in posListN:
					extendSeqList.append(seqRefn[locS-10:locE+10])
				extendSeq = extendSeqList[0]
				for i in extendSeqList:
					if not (i == extendSeq):
						print(seq+" has more than 1 match and the extending sequences of  these matches are not same!")
				if len(posListP) > len(posListN):
					locS,locE = posListP[0]
				
					fw.write(line.strip('\n')[:-1] +  "\t") 
					fw.write(str(seqRefp[locS-10:locE+10]) +"\t")
					fw.write(str((Tm.Tm_staluc(seq[:7])+31)/58.0) +  "\t")
					fw.write(str((Tm.Tm_staluc(seq[7:15])+19)/55.0) +  "\t")
					fw.write(str((Tm.Tm_staluc(seq[15:20])+64)/65.0)+ "\t")
					fw.write(str((Tm.Tm_staluc(seq)-36)/30) + "\t")
					fw.write(str((Tm.Tm_staluc(seqRefp[locS-5:locS])+64)/63.0) + "\t")
					fw.write(str((Tm.Tm_staluc(seqRefp[locE-3:locE+2])+67)/61.8) + "\n")

			
				if len(posListP) <= len(posListN):
					locS,locE = posListN[0]
				
					fw.write(line.strip('\n')[:-1] +  "\t") 
					fw.write(str(seqRefn[locS-10:locE+10]) +"\t")
					fw.write(str((Tm.Tm_staluc(seq[:7])+31)/58.0) +  "\t")
					fw.write(str((Tm.Tm_staluc(seq[7:15])+19)/55.0) +  "\t")
					fw.write(str((Tm.Tm_staluc(seq[15:20])+64)/65.0)+ "\t")
					fw.write(str((Tm.Tm_staluc(seq)-36)/30) + "\t")
					fw.write(str((Tm.Tm_staluc(seqRefn[locS-5:locS])+64)/63.0) + "\t")
					fw.write(str((Tm.Tm_staluc(seqRefn[locE-3:locE+2])+67)/61.8) + "\n")

	fw.close()



createDataset("Cas9")
createDataset("eSpCas9")
createDataset("knoRecA_Cas9")


fr = open("./POS.txt","r")
mydictPOS = {}
lineAll = fr.readlines()
for line in lineAll:
	linelist = line.strip("\n").split("\t")
	mydictPOS[linelist[0]] = linelist[1] + "\t" + linelist[2] + "\t" + linelist[3] + "\t" + linelist[4] 	
fr.close()


fr = open("./flodscores.txt","r")
mydictFlod= {}
lineAll = fr.readlines()
for line in lineAll:
	linelist = line.strip("\n").split("\t")
	mydictFlod[linelist[0]] = linelist[1] + "\t" + linelist[2] + "\t" + linelist[3] + "\t" + linelist[4] 	
fr.close()


def createDataset2(enz):
	fr = open("./" + enz + "_Set1_tmp.txt",'r')
	fw = open("./" + enz + "_Set1.txt",'w')
	line = fr.readline()
	while(line):
		lineList = line.strip("\n").split("\t")
		seq = lineList[2]
		fw.write(line[:-1])
		fw.write(mydictPOS[seq] + "\t")
		fw.write(mydictFlod[seq] + "\n")
		line = fr.readline()
	fr.close()
	fw.close()
		
createDataset2("Cas9")
createDataset2("eSpCas9")
createDataset2("knoRecA_Cas9")



