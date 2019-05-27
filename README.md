# Prediction of sgRNA on-target activity in bacteria by deep learning

### Dependencies
1. python 2.7.12
2. tqdm 4.19.6
3. biopython 1.73
4. numpy 1.15.2
5. scikit-learn 0.19.1
6. tensorflow 1.1.0
7. tensorflow-gpu 1.2.0
8. dm-sonnet 1.9 
9. torch 0.2.0
10. torchvision 0.1.9

### Implementation of the Main Procedures

##### Application of CNN_5layers Predictor

```
python predictor.py <model> <input_file> <output_file>
```

``<model_name>``  preictor based on model name, including Cas9, eSpCas9, knoRecA\_Cas9, Eukaryon

``<input_file>``  input file and contains 43 nt-long on-target DNA sequences.

``<output_file>`` output file contain prediction results .

Examples:

```
python predictor.py Cas9 sample.txt sample_out.txt
python predictor.py Eukaryon sample.txt sample_out.txt
```

##### Train Several CNN Modes

```
python train_CNN.py <SET> <ENZ> <NET> <CV>
```

``<SET>`` Choice of dataset.

- Set1
- Set2
- Eukaryon (no need ``<ENZ>``, ``<NET>`` and ``<CV>``, ``<NET>`` default CNN_5)

``<ENZ>`` Choice of three scenarios

- Cas9
- eSpCas9
- knoRecA_Cas9

``<NET>`` Choice of network architectures

- CNN_Lin
- DeepCRISPR
- DeepCas9
- CNN_2
- CNN_3
- CNN_4
- CNN_5
- CNN_6
- CNN_7

``<CV>``  Number of 5-fold Cross-validation

- 1
- 2
- 3
- 4
- 5

Examples:

```
python train_CNN.py Set1 Cas9 CNN_Lin 1
python train_CNN.py Set1 eSpCas9 DeepCRISPR 2
python train_CNN.py Set2 eSpCas9 DeepCas9 3
python train_CNN.py Set2 knoRecA_Cas9 CNN_5 4
python train_CNN.py Eukaryon
```

##### Train Later LinearRegression Models

```
python train_LR.py
```

Adjusting annotation can choice to LinearRegression, Ridge, Lasso, ElasticNet, SVR and GradientBoostingRegressor algorithms.

### Table of Contents

**data** - All data used in this project

- [cas-offinder-master](./data/cas-offinder-master/) Cas-OFFinder is OpenCL based, ultrafast and versatile program that searches for potential off-target sites of CRISPR/Cas-derived RNA-guided endonucleases (get from [https://github.com/snugel/cas-offinder](https://github.com/snugel/cas-offinder)) [1].
- [cd-hit-v4.6.6-2016-0711](./data/cd-hit-v4.6.6-2016-0711/) CD-HIT is a very widely used program for clustering and comparing protein or nucleotide sequences (get from [https://github.com/weizhongli/cdhit/releases/tag/V4.6.6](https://github.com/weizhongli/cdhit/releases/tag/V4.6.6)) [2].
- [ViennaRNA-2.4.11](./data/ViennaRNA-2.4.11/) ViennaRNA is widely used for the prediction and comparison of RNA secondary structures for nearly two decades (get from [https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_4_x/ViennaRNA-2.4.11.tar.gz](https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_4_x/ViennaRNA-2.4.11.tar.gz)) [3].
- [raw_data](./data/raw_data/) Major sources of raw data [4].
- [mediate_processing](./data/mediate_processing/) This is intermediate processing of data, including creating Set1, Set2, input data of LinearRegression, independent validation set, cross-validation data and POS. POS relate to CROP-IT[6] and CCTOP [7]  MITScore [8, 9], CFD-
Score [10]. 
- [Ref_genome](./data/Ref_genome/)　Reference Genome files of Human, Mouse and Escherichia coli.
- [Set1](./data/Set1/) Set1 dataset.
- [Set2](./data/Set2/) set2 dataset.
- [Cross_validation](./data/Cross_validation/) Data involved in cross validation.
- [LR_data](./data/LR_data/) Data for training later linear regression models.
- [eukaryon](./data/eukaryon) Some eukaryotic modes and eukaryotic datasets[9] including Chuai2018\_DeepCRISPR [11], Peng\_TSAM [12], Xue2018_DeepCas9 [13].

**net** - Several CNN architectures for comparison and selection.


- [CNN_lin](./net/CNN_Lin.py) One of CNN architectures for comparison　(get from [https://github.com/MichaelLinn/off_target_prediction](https://github.com/MichaelLinn/off_target_prediction)) [7]. we changed output size of last full connected layer to one value, instead two value, and changed binary cross entropy loss function to mean squared error.
- [DeepCas9](./net/DeepCas9.py) One of CNN architectures for comparison (get from [https://github.com/MyungjaeSong/Paired-Library/tree/DeepCRISPR.info/DeepCas9](https://github.com/MyungjaeSong/Paired-Library/tree/DeepCRISPR.info/DeepCas9)) [14].
- [DeepCRISPR](./net/DeepCRISPR.py) One of CNN architectures for comparison (get from [https://github.com/bm2-lab/DeepCRISPR](https://github.com/bm2-lab/DeepCRISPR)) [11].
- [CNN](./net/CNN_Lin.py)  Our built CNN　architectures, including CNN_2layers, CNN_2layers, CNN_3layers, CNN_4layers, CNN_5layers, CNN_6layers and CNN_7layers.
- [utils](./net/CNN_Lin.py) Some functions　as helper tools　of CNN IO.

**saved_models** - Some trained and saved models of CNN_5layers.
- [Cas9.pkl, eSpCas9.pkl, knoRecA\_Cas9.pkl, Eukaryon.pkl](./saved_models/)


### References

[1] Bae, S., Park, J., Kim, J.S.: Cas-offinder: a fast and versatile algorithm that searches for potential off-target sites of cas9 rna-guided endonucleases. Bioinformatics 30(10), 1473–1475 (2014)

[2] Fu, L., Niu, B., Zhu, Z., Wu, S., Li, W.: Cd-hit: accelerated for clustering the next-generation sequencing data.
Bioinformatics 28(23), 3150–3152 (2012)

[3] Lorenz, R., Bernhart, S.H., Siederdissen, C.H.Z., Tafer, H., Flamm, C., Stadler, P.F., Hofacker, I.L.: Viennarna package 2.0. Algorithms for Molecular Biology 6, 26 (2011)

[4]	Guo, J., Wang, T., Guan, C., Liu, B., Luo, C., Xie, Z., Zhang, C., Xing, X.H.: Improved sgrna design in bacteria via genome-wide activity profiling. Nucleic Acids Res 46(14), 7052–7069 (2018)

[5] Singh, R., Kuscu, C., Quinlan, A., Qi, Y., Adli, M.: Cas9-chromatin binding information enables more accurate crispr off-target prediction. Nucleic Acids Res 43(18), 118 (2015)

[6] Stemmer, M., Thumberger, T., Del Sol Keyer, M., Wittbrodt, J., Mateo, J.L.: Cctop: An intuitive, flexible and reliable crispr/cas9 target prediction tool. PLoS One 10(4), 0124633 (2015)

[7] Lin, J., Wong, K.-C.: Off-target predictions in crispr-cas9 gene editing using deep learning. Bioinformatics 37(17), 656–663 (2018)

[8] Hsu, P.D., Scott, D.A., Weinstein, J.A., Ran, F.A., Konermann, S., Agarwala, V., Li, Y., Fine, E.J., Wu, X., Shalem, O., Cradick, T.J., Marraffini, L.A., Bao, G., Zhang, F.: Dna targeting specificity of rna-guided cas9 nucleases. Nature biotechnology 31(9), 827–832 (2013)

[9] Haeussler, M., Kai, S., Eckert, H., Eschstruth, A., Miann´ e, J., Renaud, J.B., Schneider-Maunoury, S., Shkumatava, A., Teboul, L., Kent, J.: Evaluation of off-target and on-target scoring algorithms and integration into the guide rna selection tool crispor. Genome Biology 17(1), 148 (2016)

[10] Doench, J.G., Fusi, N., Sullender, M., Hegde, M., Vaimberg, E.W., Donovan, K.F., Smith, I., Tothova, Z., Wilen, C., Orchard, R.: Optimized sgrna design to maximize activity and minimize off-target effects of crispr-cas9. Nature Biotechnology 34(2), 184–191 (2016)

[11] Chuai, G., Ma, H., Yan, J., Ming, C., Hong, N., Xue, D., Chi, Z., Zhu, C., Ke, C., Duan, B.: Deepcrispr: optimized crispr guide rna design by deep learning. Genome Biology 19(1), 80 (2018)

[12] Peng, H., Zheng, Y., Blumenstein, M., Tao, D., Li, J.: Crispr/cas9 cleavage efficiency regression through boosting algorithms and markov sequence profiling. Bioinformatics 34(18), 3069–3077 (2018)

[13] Xue, L., Tang, B., Chen, W., Luo, J.S.: Prediction of crispr sgrna activity using a deep convolutional neural network. Journal of Chemical Information and Modeling 59(1), 615–624 (2019)

[14] Kim, H.K., Min, S., Song, M., Jung, S., Choi, J.W., Kim, Y., Lee, S., Yoon, S., Kim, H.H.: Deep learning improves prediction of crispr-cpf1 guide rna activity. Nat Biotechnol 36(3), 239–241 (2018)


