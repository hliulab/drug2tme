<H1> Cross-domain feature disentanglement for interpretable modeling of tumor microenvironment impact on drug response </H1>

## 1. Introduction
**DRUG2TME** is a cross domain feature disentanglement method for  interpretable modelding of tumor microenvironment impact on drug response.
## Architecture
![architecture](image.png?raw=true)

## 3. Usage

### 3.1 Data
We collected cell lines data from CCLE/GDSC, PDTX/PDTC data from BCaPE, patient tumor data from TCGA, and then build a benchmark dataset, which contains hundred types of drugs.




### 3.2 Basic Usage
#### 3.2.1 convert drugs to gragh
use create_data_DC.py to convert drugs' SMILE to graph;then take the graph as the input of drug encoder. 

#### 3.2.2 Drug encoder
encoder_gat.py contains the GNN encoder, which take drugs' grapg as input and output drug embeddings of 128 dims.

#### 3.2.3 DA methods
gradient_reveral.py(DANN) : we use this file to build gradient_reversal layer and then achieve DANN da method. DANN is the Adversarial method of domain adaptation.

loss_and_metrics.py(MMD) : we use kernel function to compute the mmd loss, which can measure the distribution of sourch and target domain. 

#### 3.3.4 drug2tme models
encoder_decoder.py contains dae and discriminator and classifier

models.py defines the forward function and loss functions.

#### 3.2.5 train and eval
pretraining.py & model_train.py (da and train encoders) : as the first stage of drug2tme training.we use this file to train encoders to achieve domain adaptation,asking cell lines and patients in sanme cancer type. we also train the cell predictor. 

tcga_main.py(train tme predictor) : in this stage, we freeze the encoders and cell predictor and then use some patient drug response data to train the tme predictor.

in practical use, you can run tcga_main.py or model_train.py to train the whole model, you can chose the different task by changing the task name in utils_cell_drug.py to gengeate different dataset.
<b>Example code:</b>
 
 ```tcga_data = CellDrugData(root = 'data/pretrain/', task = 'tcga')```
 
 The tcga_data will be the tcga-drug pairs for training.
