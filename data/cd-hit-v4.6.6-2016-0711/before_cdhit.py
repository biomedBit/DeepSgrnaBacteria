def ceate_cdhit_input(enz):
	fr = open("../mediate_processing/" + enz +"_Set1.txt", "r")
	fw = open("./" + enz + "_cdhit_input.fasta","w")
	line = fr.readline()
	while(line):
		fw.write(">" + line)
		seq = line.strip("\n").split("\t")[3]
		fw.write(seq + "\n")
		line = fr.readline()

ceate_cdhit_input("Cas9")
ceate_cdhit_input("eSpCas9")
ceate_cdhit_input("knoRecA_Cas9")
