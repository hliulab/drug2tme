<H1> Cross-domain feature disentanglement for interpretable modeling of tumor microenvironment impact on drug response </H1>

## Introduction
**DRUG2TME** is a cross domain feature disentanglement method for  interpretable modelding of tumor microenvironment impact on drug response.
## Model architecture
![architecture](main/image.png?raw=true)

##  Usage

### Benchmark dataset
We collected cell lines data from CCLE/GDSC, PDTX/PDTC data from BCaPE, patient tumor data from TCGA, and then build a benchmark dataset, which contains hundred types of drugs.


### Conda environments
rdkit 2022.03.5

pytorch 1.12.1

python 3.9.7

scikit-learn 1.1.2

umap-learn 0.5.3

torch-geometric 2.1.0.post1

torchvision 0.13.1

matplotlib-base 3.6.0

mkl 2021.4.0

numpy 1.23.1

networkx 2.8.6

pandas 1.5.0


The deep learning models were trained on 2*NVIDIA GeForce RTX 4090 on linux.

### Basic Usage
#### Convert drugs to gragh
Run create_data_DC.py to convert drugs' SMILE to molecular graph, which is taken as the input of drug encoder.  

#### Drug encoder
The encoder_gat.py file contains the GNN encoder, which take drug's graph as input and output drug embeddings of 128 dimensions.

#### Domain adaptation
Run gradient_reveral.py (DANN) to build gradient_reversal layer and then conduct DANN domain adaptation method. DANN is a method for adversarial domain adaptation.

Run loss_and_metrics.py (MMD) to compute the MMD loss, which can measure the distribution of sourch and target domain. 

#### Drug2tme models
The encoder_decoder.py defines two encoders $E_c$ and $E_t$, discriminator for domain adaptation, cell line predictor $F_c$ and TME predictor $F_t$. The models.py defines the forward function and loss functions.

#### Training and evaluation
The pretraining.py covers the first stage of drug2tme training, you can run this file to train two encoders, discriminator, and cell line predictor to achieve domain adaptation and get trained encoders and cell line predictor that are ready for second stage training. 

The model_train.py is run to train the TME predictor, when it run the trained encoders and cell line predictor are frozen. You can choose percentage of patient-drug pairs using for train the TME predictor.

In practical use, you can chose the different task by changing the task name to generate different dataset using utils_cell_drug.py.
<b>Example code:</b>
 
 ```tcga_data = CellDrugData(root = 'data/pretrain/', task = 'tcga_label_gsva')```
 
 The tcga_data will be the tcga-drug pairs for training TME predictor.

If you want to train Drug2TME from scratch and use your own cell line or patient data. First, you need to set the path of SMILE file for drugs and gene expression file for cell lines and patients in utils_cell_drug.py. Next, you should run pretraining.py to start the first stage training. In this stage , make sure the data file name is consistent with what you set in first stage so as to generate your own dataset (see example code). In the second stage, the patient-drug pairs data will be need to train the TME predictor. 

In conclusion, you can only need to use

 ```ptyhon pretraining.py```

and 

 ```ptyhon model_train.py``` 
 
 to train a new model with setting right path for your own data in utils_cell_drug.py.


 #### Expreiments for CDX and melanoma patients
After two stage training, Drug2TME can be used for predict other datasets. You can use test.py to predict the drug response for CDX or pre/post treatment melanoma patient by changing the task name like "model_train.py" 
