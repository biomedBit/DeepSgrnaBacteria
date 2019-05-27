To use the Matlab version TSAM, the following packages must be installed:
1.xgboost
2.biopython
3.libsvm

The installation of these packages are as follows:

1. The installation of xgboost can be found at the website: https://xgboost.readthedocs.io/en/latest/build.html
It should be mentioned that the latest version package should be installed or errors may be occured such as cannot
use the parameter " 'objective': 'reg:tweedie' ". (we installed the xgboost-0.6a2 version)

2. The installation of the biopython can follow the decription at http://biopython.org/DIST/docs/install/Installation.html
The biopython-1.70 was installed when implementing the TSAM

3. libsvm can be downloaded from the website: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
Here, the Matlab version should be installed. We installed the libsvm 3.22.

Inaddition, the following independent packages should be installed:
4. numpy
5. scipy
6. sklearn

The python2.7 is recommended for running these codes.

######################################################################################################
Prediction cutting efficiencies or classification of sgRNAs for a given gene' s patential sgRNAs
can be done via the following action:
(under either windows or linux os)
        open your Matlab (2015 or higher);
        Downloaded and compressed the TSAM package;
        add the matlab_codes/ to your matlab working path;
        run the following command:
             Predict_score=TSAM(filepath, pretype, featype, sgtype);

##############  illustration of the parameters   ########################################################################
### filepath (string)--xxx.fa:
###                     the .fa sequence file that contain the gene sequence                    
### pretype (int)---1/2/3:
###                     pretype=1: prediction sgRNA efficiencies for cutting human and mouse genomes;
###                     pretype=2: prediction sgRNA efficiencies for cutting zebrafish genome;
###                     pretype=3: classification of sgRNAs to cut human or mouse genomes

### featype (int)---1/2/3:
###                     featype =1: using the TSAM, where all the 677 dimensions features are applied;
###                     featype =2: using the TSAM-MT1, where the cutting features is not used (674d);
###                     featype =3: using the TSAM-MT2, where the cut_per_geno has been used (675d);
### sgtype (int)---1/0:
###                     1 for cut at exon only which means the sgRNAs cutting at the exons will be predicted;
###                     0 for all, where all the potential sgRNAs will be predicted

######################################################################################################################

#### example codes(one can paste this to the command, please change the path '/' to '\' when under the windows os):
    
   Predict_score=TSAM(../example_input_files/test1.fa, 1, 1, 1);

######################################################################################################################
***The implementation of the cross-validation and independent test is not support for the Matlab version codes

If you encount some problems to run the codes, please contact Hui Peng: Hui.Peng-2@student.uts.edu.au