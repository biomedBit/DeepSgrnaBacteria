python before_cdhit.py
cdhit-est -i Cas9_cdhit_input.fasta -o Cas9_cdhit_output.fasta -c 0.8
cdhit-est -i eSpCas9_cdhit_input.fasta -o eSpCas9_cdhit_output.fasta -c 0.8
cdhit-est -i knoRecA_Cas9_cdhit_input.fasta -o knoRecA_Cas9_cdhit_output.fasta -c 0.8
python after_cdhit.py
