import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
import csv
from create_data_DC import smile_to_graph
import re
#import deepchem as dc

class CellDrugDataset(InMemoryDataset):
    def __init__(self, root='newtmp', task=None,
                transform=None, pre_transform=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(CellDrugDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset
        self.task = task
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            # creat_data(self.dataset)
            self.process(root, task)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.task + '_dataset.pt']
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, root, task):
        path = 'data/pre_data/pretrain/'
        if task == 'ccle_pretrain':
            cell_path = 'cell_shared_gene.csv'
            cell_features = pd.read_csv(path + cell_path)
            data = pd.read_csv(path + 'ACH_drug_label.csv')
        elif task == 'ccle_pretrain_nolabel':
            cell_path = 'cell_gsva.csv'
            cell_features = pd.read_csv(path + cell_path)
            sample = pd.read_csv(path + 'ccle_sample.csv')
        elif task == 'tcga_pretrain':
            cell_path = 'new_tcga_shared_gene.csv'
            cell_features = pd.read_csv(path + cell_path)
            sample = pd.read_csv(path + 'tcga_sample.csv')
        elif task == 'tcga_label_all':
            pdx_path = 'new_tcga_shared_gene.csv'
            pdx_features = pd.read_csv(path + pdx_path)
            data = pd.read_csv(path + 'new_tcga_label.csv')

        elif task == 'ccle_pretrain_gsva':
            cell_path = 'cell_gsva.csv'
            cell_features = pd.read_csv(path + cell_path)
            data = pd.read_csv(path + 'ACH_drug_label.csv')
        elif task == 'tcga_pretrain_gsva':
            cell_path = 'tcga_gsva.csv'
            cell_features = pd.read_csv(path + cell_path)
            sample = pd.read_csv(path + 'tcga_sample.csv')
        elif task == 'tcga_label_gsva':
            pdx_path = 'tcga_gsva.csv'
            pdx_features = pd.read_csv(path + pdx_path)
            data = pd.read_csv(path + 'new_tcga_label.csv')
        elif task == '100_tcga':
            pdx_path = 'tcga_gsva.csv'
            pdx_features = pd.read_csv(path + pdx_path)
            data = pd.read_csv(path + '100_tcga.csv')
        elif task == 'Nav':
            pdx_path = 'tcga_gsva.csv'
            pdx_features = pd.read_csv(path + pdx_path)
            data = pd.read_csv(path + 'Nav.csv')
        elif task == 'cdx':
            tcga_path = 'gsva_cdx.csv'
            tcga_features = pd.read_csv(path + tcga_path)
            data = pd.read_csv(path + 'CDX_data.csv')
        elif task == '6_patient':
            tcga_path = 'gsva_patient.csv'
            tcga_features = pd.read_csv(path + tcga_path)
            data = pd.read_csv(path + '6_data.csv')
        elif task == 'Nav_Ali':
            pdx_path = 'tcga_gsva.csv'
            pdx_features = pd.read_csv(path + pdx_path)
            data = pd.read_csv(path + 'Nav_Ali.csv')

        smiles = pd.read_csv(path + 'smiles.csv')
        smile_list = {}
        for i, row in smiles.iterrows():
            drug_name = row['drug_name']
            smile = row['smile']
            print('smiles', smile)
            c_size, features, edge_index, atoms = smile_to_graph(smile)
            if len(edge_index) > 0:
                edge_index = torch.LongTensor(edge_index).transpose(1, 0)
            else:
                edge_index = torch.LongTensor(edge_index)
            x =torch.Tensor(features)
            smile_list[drug_name] = (x, edge_index)

        data_list = []
        if task =='ccle_pretrain':
            for i in range(len(data)):
                cell_id = data.loc[i, 'sample']
                drug = data.loc[i, 'DRUG_NAME']
                cancer = data.loc[i,'Cancer']
                label1 = data.loc[i, 'Label']
                #label2 = data.loc[i, 'Z_SCORE']
                cell_feature = cell_features[str(cell_id)].values
                x, edge_index =smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label1]),
                                    #y2=torch.Tensor([label2]),
                                    cell_id=cell_id, drug=drug,cancer = cancer)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([cell_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(cell_id) + ' ' + drug + ' ' + str(label1))
        if task =='ccle_pretrain_nolabel':
            for i in range(len(sample)):
                cell = sample.loc[i,'sample']

                cell_feature = cell_features[str(cell)].values
                GCNData = DATA.Data(cell = cell)
                GCNData.input = torch.FloatTensor(np.array([cell_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(cell))
        if task =='tcga_pretrain':
            for i in range(len(sample)):
                cell = sample.loc[i,'sample']

                cell_feature = cell_features[str(cell)].values
                GCNData = DATA.Data(cell = cell)
                GCNData.input = torch.FloatTensor(np.array([cell_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(cell))

        if task =='tcga_label_all':
            for i in range(len(data)):
                model = data.loc[i, 'PATIENT']
                drug = data.loc[i, 'DRUG_NAME']
                label = data.loc[i, 'LABEL']
                #label2 = data.loc[i, 'Z_SCORE']
                pdx_feature = pdx_features[str(model)].values
                x, edge_index =smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label]),
                                    #y2=torch.Tensor([label2]),
                                    model = model, drug=drug)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(model) + ' ' + drug + ' ' + str(label))

        if task == '100_tcga':
            for i in range(len(data)):
                model = data.loc[i, 'PATIENT']
                drug = data.loc[i, 'DRUG_NAME']
                label = data.loc[i, 'LABEL']
                # label2 = data.loc[i, 'Z_SCORE']
                pdx_feature = pdx_features[str(model)].values
                x, edge_index = smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label]),
                                    # y2=torch.Tensor([label2]),
                                    model=model, drug=drug)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(model) + ' ' + drug + ' ' + str(label))
        if task =='ccle_pretrain_gsva':
            for i in range(len(data)):
                cell_id = data.loc[i, 'sample']
                drug = data.loc[i, 'DRUG_NAME']
                cancer = data.loc[i,'Cancer']
                label1 = data.loc[i, 'Label']
                #label2 = data.loc[i, 'Z_SCORE']
                cell_feature = cell_features[str(cell_id)].values
                x, edge_index =smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label1]),
                                    #y2=torch.Tensor([label2]),
                                    cell_id=cell_id, drug=drug,cancer = cancer)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([cell_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(cell_id) + ' ' + drug + ' ' + str(label1))

        if task =='tcga_pretrain_gsva':
            for i in range(len(sample)):
                cell = sample.loc[i,'sample']

                cell_feature = cell_features[str(cell)].values
                GCNData = DATA.Data(cell = cell)
                GCNData.input = torch.FloatTensor(np.array([cell_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(cell))

        if task =='tcga_label_gsva':
            for i in range(len(data)):
                model = data.loc[i, 'PATIENT']
                drug = data.loc[i, 'DRUG_NAME']
                label = data.loc[i, 'LABEL']
                #label2 = data.loc[i, 'Z_SCORE']
                pdx_feature = pdx_features[str(model)].values
                x, edge_index =smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label]),
                                    #y2=torch.Tensor([label2]),
                                    model = model, drug=drug)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(model) + ' ' + drug + ' ' + str(label))
        if task =='Nav':
            for i in range(len(data)):
                model = data.loc[i, 'PATIENT']
                drug = data.loc[i, 'DRUG_NAME']
                label = data.loc[i, 'LABEL']
                #label2 = data.loc[i, 'Z_SCORE']
                pdx_feature = pdx_features[str(model)].values
                x, edge_index =smile_list[drug]
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    y1=torch.Tensor([label]),
                                    #y2=torch.Tensor([label2]),
                                    model = model, drug=drug)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(model) + ' ' + drug + ' ' + str(label))
        if task =='cdx':
            for i in range(len(data)):
                patient = data.loc[i, 'CDX']
                drug = data.loc[i,'DRUG']
                label = data.loc[i,'Label']
                tcga_feature = tcga_features[str(patient)].values
                x, edge_index = smile_list[drug]
                GCNData = DATA.Data(x=x,edge_index=edge_index,y1=torch.Tensor([label]),patient=patient,drug=drug)
                GCNData.input = torch.FloatTensor(np.array([tcga_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(patient))

        if task == '6_patient':
            for i in range(len(data)):
                patient = data.loc[i, 'Patient']
                drug = data.loc[i, 'Drug']
                label = data.loc[i, 'Label']
                tcga_feature = tcga_features[str(patient)].values
                x, edge_index = smile_list[drug]
                GCNData = DATA.Data(x=x, edge_index=edge_index, y1=torch.Tensor([label]), patient=patient,
                                    drug=drug)
                GCNData.input = torch.FloatTensor(np.array([tcga_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(patient))

        if task =='Nav_Ali':
            for i in range(len(data)):
                model = data.loc[i, 'PATIENT']
                drug1 = data.loc[i, 'DRUG_NAME1']
                drug2 = data.loc[i, 'DRUG_NAME2']
                #label2 = data.loc[i, 'Z_SCORE']
                pdx_feature = pdx_features[str(model)].values
                x1, edge_index1 =smile_list[drug1]
                x2, edge_index2 = smile_list[drug2]
                GCNData = DATA.Data(x1=x1,
                                    edge_index1=edge_index1,
                                    x2=x2,
                                    edge_index2=edge_index2,
                                    model = model)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(model) + ' '  )

        if task =='pdx':
            for i in range(len(sample)):
                pdx_id = sample.loc[i, 'model']


                pdx_feature = pdx_features[str(pdx_id)].values

                GCNData = DATA.Data(
                                    patient_id=pdx_id)
                # append graph, label and target sequence to data list
                GCNData.input = torch.FloatTensor(np.array([pdx_feature]))
                data_list.append(GCNData)
                print(str(i) + '   ' + str(pdx_id) )


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])




def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci