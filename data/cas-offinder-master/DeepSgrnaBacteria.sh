# Because the output file of single execution is too large and GitHub does not allow uploading files over 100MB, so we execute it three times.
./cas-offinder cas-offinder_input1.txt G cas-offinder_output1.txt
./cas-offinder cas-offinder_input2.txt G cas-offinder_output2.txt
./cas-offinder cas-offinder_input3.txt G cas-offinder_output3.txt
python createPOS.py
