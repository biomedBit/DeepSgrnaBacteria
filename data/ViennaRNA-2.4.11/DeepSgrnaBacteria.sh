cd ./src/bin/myRNAfold/
RNAfold --MEA sgRNAlib.fasta >> ../../../RNAflodresult.txt
cd ../../../
python createflodScores.py
