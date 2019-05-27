f = open("RNAflodresult.txt","r")
fw = open("../mediate_processing/flodscores.txt","w")
line = f.readline()
while(line):
	RNA=line[1:-1]	
	line = f.readline()
	line = f.readline()
	mfe = line.split(" ")[-1][:-2]
	line = f.readline()
	fete = line.split(" ")[-1][:-2]
	line = f.readline()
	line = f.readline()
	line = f.readline()
	linelist = line.split(" ")
	fmfe = linelist[7][:-1]
	ed = linelist[-3]
	fw.write(RNA + "\t" + mfe + "\t" + fete + "\t" + fmfe + "\t" + ed +"\n")
	line = f.readline()
fw.close()
f.close()
