import random
def createSet2(enz):
	fr = open("./" + enz + "_cdhit_output.fasta","r")
	fw = open("../mediate_processing/" + enz + "_Set2.txt" ,"w")
	line = fr.readline()
	while(line):
		if line[0] == ">":
			fw.write(line[1:])
		line = fr.readline()	
	fw.close()
	fr.close()


createSet2("Cas9")
createSet2("eSpCas9")
createSet2("knoRecA_Cas9")

