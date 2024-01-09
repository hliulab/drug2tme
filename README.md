<H1> Cross-domain feature disentanglement for interpretable modeling of tumor microenvironment impact on drug response </H1>

## 1. Introduction
**DRUG2TME** is a cross domain feature disentanglement method for  interpretable modelding of tumor microenvironment impact on drug response.
## 2. Architecture
![architecture](image.png?raw=true)

## 3. Data and Usage

### 3.1 Data
We collected cell lines data from CCLE/GDSC, PDTX/PDTC data from BCaPE, patient tumor data from TCGA, and then build a benchmark dataset, which contains hundred types of drugs.

### 3.2 Basic Usage
#### 3.2.1 convert drugs to gragh
Run create_data_DC.py to convert drugs' SMILE to molecular graph, which is taken as the input of drug encoder. 

#### 3.2.2 Drug encoder
The encoder_gat.py file contains the GNN encoder, which take drug's graph as input and output drug embeddings of 128 dimensions.

#### 3.2.3 Domain adaptation
Run gradient_reveral.py (DANN) to build gradient_reversal layer and then conduct DANN domain adaptation method. DANN is a method for adversarial domain adaptation.

Run loss_and_metrics.py (MMD) to compute the MMD loss, which can measure the distribution of sourch and target domain. 

#### 3.3.4 Definition of drug2tme model
The encoder_decoder.py defines the denoising autoencoder, discriminator and classifier, and models.py defines the forward function and loss functions.

#### 3.2.5 Training and evaluation
pretraining.py & model_train.py: as the first stage of drug2tme training.we use this file to train encoders to achieve domain adaptation,asking cell lines and patients in sanme cancer type. we also train the cell predictor. 

tcga_main.py(train tme predictor) : in this stage, we freeze the encoders and cell predictor and then use some patient drug response data to train the tme predictor.

in practical use, you can run tcga_main.py or model_train.py to train the whole model, you can chose the different task by changing the task name in utils_cell_drug.py to gengeate different dataset.
<b>Example code:</b>
 
 ```tcga_data = CellDrugData(root = 'data/pretrain/', task = 'tcga')```
 
 The tcga_data will be the tcga-drug pairs for training.
