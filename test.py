import os
import argparse
import random
import torch.nn.functional as F
import torch
import umap
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve, r2_score, mean_squared_error, balanced_accuracy_score, confusion_matrix, \
    precision_score, accuracy_score, cohen_kappa_score, recall_score
from torch import nn
from tqdm import tqdm
from encoder_decoder import shared_encoder,shared_decoder,micro_encoder,classifier,cls_mirco,discriminator,response_layer,stack
from itertools import chain
from itertools import cycle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.cuda.set_device(1)
import globalvar as gl
#from torch.utils.tensorboard import SummaryWriter
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
    print('The code uses CPU!!!')
device = gl.get_value('cuda')
from utils_cell_drug import *
from encoder_gat import GATNet
import math
from torch_geometric.loader import DataLoader
from models import Model
from loss_and_metrics import mmd_loss
from collections import OrderedDict
from sklearn.metrics import log_loss,precision_score,recall_score,f1_score,\
                fbeta_score,confusion_matrix
from sklearn.model_selection import KFold, train_test_split



def predicting_target(p_model,device,tcga_loader):
    print('Predict on {} tcga_drug samples...'.format(len(tcga_loader.dataset)))
    p_model.eval()
    p_total_preds_1 = torch.Tensor()
    p_total_preds_2 = torch.Tensor()
    p_total_labels = torch.Tensor()
    total = 0
    total_p_recon = 0
    total_p_domain = 0
    cls_1 = torch.Tensor()
    cls_2 = torch.Tensor()
    tcga_total = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(tcga_loader)):
            data1 = data[0].to(device)

            p, p_recons, p_embedding_1, p_embedding_2, p_embedding, cls_out_1,cls_out_2,cls_out, cls_y = p_model(data1)
            # sig = nn.Sigmoid()
            # cls_out_1 = sig(cls_out_1)
            # cls_out_2 = sig(cls_out_2)
            # cls_out = sig(cls_out)
            cls_out = cls_out.to('cpu')
            cls_y = cls_y.to('cpu')
            p_total_preds_1 = torch.cat((p_total_preds_1, cls_out), 0)
            p_total_labels = torch.cat((p_total_labels, cls_y), 0)
            cls_out_1 = cls_out_1.to('cpu')
            cls_out_2 = cls_out_2.to('cpu')
            cls_out_1 = cls_out_1.detach()
            cls_out_2 = cls_out_2.detach()
            cls_out = cls_out.detach()
            cls_1 = torch.cat((cls_1, cls_out_1), 0)
            cls_2 = torch.cat((cls_2, cls_out_2), 0)

    return  p_total_preds_1, p_total_labels,cls_1,cls_2,cls_out

def valid(c_model,device, cell_loader):
    print('Valid on {} cell_drug samples...'.format(len(cell_loader.dataset)))
    c_model.eval()
    c_total_preds = torch.Tensor()
    c_total_labels = torch.Tensor()

    total = 0


    total_c_cls = 0
    cell_total = torch.Tensor()

    with torch.no_grad():
        for batch_idx, data in enumerate(zip(cell_loader)):
            data1 = data[0].to(device)

            cell, c_embedding, c_recons, pre_out, pre_y = c_model(data1)
            pre_out = pre_out.to('cpu')
            pre_y = pre_y.to('cpu')
            c_total_preds = torch.cat((c_total_preds, pre_out), 0)
            c_total_labels = torch.cat((c_total_labels, pre_y), 0)
            pre_out = pre_out.detach()

    return c_total_preds, c_total_labels,pre_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train and eval')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--train_batch_size', default = 2048, type = int, help = 'batch for training')

    args = parser.parse_args()
    epochs,train_batch_size = args.epochs,args.train_batch_size
    label = 'mmd_test_2'

    LOG_INTERVAL = 10


    # drug_encoder = GATNet()
    # drug_params = torch.load('results/model_adv_pretraining_6/500_drug.pkl', map_location='cuda:0')
    # drug_encoder.load_state_dict(drug_params)
    # shared_en_params = torch.load('results/model_adv_pretraining_6/500_share_encoder.pkl',map_location='cuda:0')
    # shared_de_params = torch.load('results/model_adv_pretraining_6/500_share_decoder.pkl',map_location='cuda:0')
    # micro_en_params = torch.load('results/model_adv_pretraining_6/500_micro_encoder.pkl',map_location='cuda:0')
    # cls_params = torch.load('results/model_adv_pretraining_6/500_share_cls.pkl',map_location='cuda:0')
    # cls_m_params = torch.load('results/model_mmd_test_2/500_cls_m.pkl',map_location='cuda:0')
    # share_en = shared_encoder(noise_flag=True, norm_flag=True)
    # share_en.load_state_dict(shared_en_params)
    # share_de = shared_decoder()
    # share_de.load_state_dict(shared_de_params)
    # micro_en = micro_encoder(noise_flag=True, norm_flag=True)
    # micro_en.load_state_dict(micro_en_params)
    # cls = classifier()
    # cls.load_state_dict(cls_params)
    # cls_m = cls_mirco()
    # cls_m.load_state_dict(cls_m_params)

    drug_encoder = GATNet()
    share_en = shared_encoder(noise_flag=True, norm_flag=True)
    share_de = shared_decoder()
    micro_en = micro_encoder(noise_flag=True, norm_flag=True)
    cls = classifier()
    cls_m = cls_mirco()
    stack_layer = stack()
    p_params = torch.load('results/model_stack2/500_p_model.pkl', map_location='cuda:0')

    c_model = Model(drug_encoder=drug_encoder, label=True, cell=True,
                    shared_encoder=share_en, shared_decoder=share_de, classifier=cls).to(device)
    p_model = Model(drug_encoder=drug_encoder,label=True,tcga=True,
                    shared_encoder=share_en,shared_decoder=share_de,micro_encoder=micro_en,classifier=cls,cls_micro=cls_m,stack = stack_layer).to(device)
    p_no_train = Model(drug_encoder=drug_encoder, label=True, tcga=True,
                       shared_encoder=share_en, shared_decoder=share_de, micro_encoder=micro_en, classifier=cls).to(
        device)
    p_model.load_state_dict(p_params)

    data = CellDrugDataset(root='data/pre_data', task='6_patient')
    # for i in data:
    #     print(i.batch)
    # all patient and drug samples
    # tcga_train_2, tcga_val_2 = train_test_split(tcga_val_data, test_size=0.2, shuffle=True, random_state=2)
    # tcga_train_2, tcga_test = train_test_split(tcga_train_2, test_size=0.7, shuffle=True, random_state=2)
    # tcga_train_loader = DataLoader(tcga_train_2, batch_size=2048)
    # tcga_val_loader = DataLoader(tcga_val_2, batch_size=2048)
    loader = DataLoader(data,batch_size = 2048,shuffle = False)

    preds, labels,  cls_out_1,cls_out_2 ,cls_out= predicting_target(p_model, device, loader)
    # preds = preds.numpy()
    # preds = np.around(preds, 0).astype(int)
    #
    # auc_p = roc_auc_score(labels,preds)
    # print(auc_p)
    #
    # preds = preds.numpy()
    # preds = np.around(preds, 0).astype(int)
    # f1 = f1_score(labels, preds)
    # print('f1_score tcga is {}'.format(f1))
    # precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=1)
    # aupr = auc(recall, precision)
    # print('AUPR tcga is {}'.format(aupr))  #
    cls_out_1_np = cls_out_1.numpy()
    cls_out_2_np = cls_out_2.numpy()
    cls_out_np = cls_out.numpy()
    label_np = labels.numpy()

    np.savetxt('cls1_6.csv', cls_out_1_np)
    np.savetxt('cls2_6.csv', cls_out_2_np)
    np.savetxt('cls_6.csv', cls_out_np)
    # cls_out_np = cls_out.numpy()
    # label_np = labels.numpy()
    #
    # np.savetxt('cls_gdsc_pca.csv', cls_out_np)










    # c_model_test = Model(noise_flag=True, norm_flag=True, drug_encoder=drug_encoder, label=True,cell=True,
    #                 shared_encoder=share_en, shared_decoder=share_de, micro_encoder=micro_en, classifier=cls).to(device)
    # p_model_test  = Model(noise_flag=True, norm_flag=True, drug_encoder=drug_encoder,label=True,tcga=True,
    #                 shared_encoder=share_en,shared_decoder=share_de,micro_encoder=micro_en,classifier=cls).to(device)
    # checkpoint_p = torch.load(model_file_name, map_location='cuda:0')
    # checkpoint_c = torch.load(encoder_file_name, map_location='cuda:0')
    #
    # p_model_test.load_state_dict(checkpoint_p)
    # c_model_test.load_state_dict(checkpoint_c)
    #
    # loss, recon_loss, ortho_loss, cls_loss, preds, labels = predicting_target(p_model_test, device, tcga_test_loader)
    # auc = roc_auc_score(labels, preds)
    #
    # print('AUC cdx is {}'.format(auc))
    # loss, recon, cls, preds, labels = predicting_source(c_model, device, cell_test_loader)
    # auc_c = roc_auc_score(labels, preds)
    # fpr, tpr, thresholds = roc_curve(labels, preds)  # 计算真正率和假正率
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig('roc_cell.jpg')
    #
    # print('AUC cell is {}'.format(auc_c))
    # print('AUC2 cell is {}'.format(roc_auc))
    #
    # cls_loss, preds, labels = predicting_target(p_model, device, tcga_test_loader)
    # auc_t = roc_auc_score(labels, preds)
    #
    # print('AUC cdx is {}'.format(auc_t))
    # fpr, tpr, thresholds = roc_curve(labels, preds)  # 计算真正率和假正率
    # roc_auc = auc(fpr, tpr)
    # print('AUC2 tcga is {}'.format(roc_auc))
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig('roc_tcga.jpg')
    #
    # torch.save(p_model.state_dict(), model_file_name)
    # torch.save(c_model.state_dict(),encoder_file_name)
    # torch.save(discriminator.state_dict(),dis_file_name)
    # #print(P)
    # #print(T)
    # #print(cls_loss3)
    # precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=1)
    # aupr = auc(recall, precision)
    # print('AUPR tcga is {}'.format(aupr))#
    #
    #
    #



