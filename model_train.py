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
from encoder_decoder import shared_encoder,shared_decoder,micro_encoder,classifier,cls_mirco,discriminator,stack
from itertools import chain
from itertools import cycle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# torch.cuda.set_device(1)
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

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# np.random.seed(42)

def predicting_target(p_model,device,tcga_loader,epoch):
    print('Predict on {} tcga_drug samples...'.format(len(tcga_loader.dataset)))
    p_model.eval()
    p_total_preds_1 = torch.Tensor()
    p_total_preds_2 = torch.Tensor()
    p_total_labels = torch.Tensor()
    total = 0
    total_p_recon = 0
    total_p_domain = 0
    total_p_micro = 0
    total_ortho = 0
    cls_1 = []
    cls_2 = []
    tcga_total = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(tcga_loader)):
            data1 = data[0].to(device)
            p, p_recons, p_embedding_1, p_embedding_2, p_embedding, cls_out_1,cls_out_2,cls_out, cls_y = p_model(data1)
            p_embedding_1_copy = p_embedding_1.to('cpu')
            tcga_total = torch.cat((tcga_total,p_embedding_1_copy),0)
            cls_out = cls_out.to('cpu')

            cls_y = cls_y.to('cpu')
            p_total_preds_1 = torch.cat((p_total_preds_1, cls_out), 0)
            p_total_labels = torch.cat((p_total_labels, cls_y), 0)
            p_model_loss = p_model.lossfunction(p, p_recons, p_embedding_1, p_embedding_2, cls_out, cls_y)
            loss = p_model_loss['cls_loss']
            cls_1.append(cls_out_1)
            cls_2.append(cls_out_2)

            total += loss.item()

            total_p_domain += p_model_loss['cls_loss'].item()
            # mmd = mmd_loss(c_embedding,p_embedding_1)
            # loss = 0.1 * c_model_loss['recon_loss'] + \
            #        0.4 * c_model_loss['cls_loss'] + 0.1 * p_model_loss['recon_loss'] + 0.3 * p_model_loss[
            #            'cls_loss'] + 0.1 * p_model_loss['ortho_loss']
            # total_mmd += mmd.item()
        # epo_mmd = total_mmd / 23
    epo_domain = total_p_domain / (batch_idx + 1)
    print(
        'Train epoch: {} \tcls_Loss: {:.6f}'.format(
            epoch,
            epo_domain

        ))
    return epo_domain, p_total_preds_1, p_total_labels,tcga_total,cls_1,cls_2




def train_target(p_model,device,tcga_loader,epoch,optimizer):
    print('Predict on {} tcga_drug samples...'.format(len(tcga_loader.dataset)))
    p_model.zero_grad()
    p_model.train()
    p_total_preds_1= torch.Tensor()
    p_total_preds_2 = torch.Tensor()
    p_total_labels = torch.Tensor()
    total = 0
    total_p_recon = 0
    total_p_domain = 0
    total_p_micro = 0
    total_ortho = 0

    for batch_idx, data in enumerate(zip(tcga_loader)):
        data1 = data[0].to(device)
        p, p_recons, p_embedding_1, p_embedding_2, p_embedding,cls_out_1,cls_out_2, cls_out, cls_y = p_model(data1)
        cls_out = cls_out.to('cpu')
        cls_y = cls_y.to('cpu')
        p_total_preds_1 = torch.cat((p_total_preds_1, cls_out), 0)
        p_total_labels = torch.cat((p_total_labels, cls_y), 0)
        p_model_loss = p_model.lossfunction(p, p_recons, p_embedding_1, p_embedding_2, cls_out, cls_y)
        loss = p_model_loss['cls_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

        total_p_domain += p_model_loss['cls_loss'].item()
        # mmd = mmd_loss(c_embedding,p_embedding_1)
        # loss = 0.1 * c_model_loss['recon_loss'] + \
        #        0.4 * c_model_loss['cls_loss'] + 0.1 * p_model_loss['recon_loss'] + 0.3 * p_model_loss[
        #            'cls_loss'] + 0.1 * p_model_loss['ortho_loss']
        # total_mmd += mmd.item()
    # epo_mmd = total_mmd / 23
    epo_domain = total_p_domain / (batch_idx+1)
    print(
        'Train epoch: {} \tcls_Loss: {:.6f}'.format(
            epoch,
            epo_domain

        ))
    return epo_domain, p_total_preds_1,p_total_labels


def valid_target(p_model, device, tcga_loader, epoch):
    print('Predict on {} tcga_drug samples...'.format(len(tcga_loader.dataset)))
    p_model.zero_grad()
    p_model.eval()
    p_total_preds = torch.Tensor()
    p_total_labels = torch.Tensor()
    total = 0
    total_p_recon = 0
    total_p_cls = 0
    total_ortho = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(tcga_loader)):
            data1 = data[0].to(device)
            p, p_recons, p_embedding_1, p_embedding_2, p_embedding, cls_out_1,cls_out_2,cls_out, cls_y = p_model(data1)
            cls_out = cls_out.to('cpu')
            cls_y = cls_y.to('cpu')
            p_total_preds = torch.cat((p_total_preds, cls_out), 0)
            p_total_labels = torch.cat((p_total_labels, cls_y), 0)
            p_model_loss = p_model.lossfunction(p, p_recons, p_embedding_1, p_embedding_2, cls_out, cls_y)
            loss = p_model_loss['cls_loss']


            total += loss.item()

            total_p_cls += p_model_loss['cls_loss'].item()
            # mmd = mmd_loss(c_embedding,p_embedding_1)
            # loss = 0.1 * c_model_loss['recon_loss'] + \
            #        0.4 * c_model_loss['cls_loss'] + 0.1 * p_model_loss['recon_loss'] + 0.3 * p_model_loss[
            #            'cls_loss'] + 0.1 * p_model_loss['ortho_loss']
            # total_mmd += mmd.item()
        # epo_mmd = total_mmd / 23
    epo_cls = total_p_cls / (batch_idx + 1)
    print(
        'Train epoch: {} \tcls_Loss: {:.6f}'.format(
            epoch,
            epo_cls

        ))
    return epo_cls, p_total_preds, p_total_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train and eval')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--train_batch_size', default = 2048, type = int, help = 'batch for training')

    args = parser.parse_args()
    epochs,train_batch_size = args.epochs,args.train_batch_size
    label = 'stack3'

    LOG_INTERVAL = 10

    # encoder_file = '100_model_encoder_gat1500_100_in-vitro_re_gat_dex25_p1'
    # drug_encoder = GATNet()
    # params = drug_encoder.state_dict()  # 获得模型的原始状态以及参数。
    # checkpoint = torch.load('data/pretrained_drug_model/' + encoder_file + '.pkl', map_location='cuda:0')
    # new_state_dict = OrderedDict()
    # for k1, k2 in zip(params.keys(), checkpoint.keys()):
    #     name = k1
    #     new_state_dict[name] = checkpoint[k2]
    # drug_encoder.load_state_dict(new_state_dict)


    shared_en_params = torch.load('results/model_adv_pretraining_6/500_share_encoder.pkl',map_location='cuda:0')
    shared_de_params = torch.load('results/model_adv_pretraining_6/500_share_decoder.pkl',map_location='cuda:0')
    micro_en_params = torch.load('results/model_adv_pretraining_6/500_micro_encoder.pkl',map_location='cuda:0')
    cls_params = torch.load('results/model_adv_pretraining_6/500_share_cls.pkl', map_location='cuda:0')
    drug_params = torch.load('results/model_adv_pretraining_6/500_drug.pkl', map_location='cuda:0')
    drug_encoder = GATNet()
    drug_encoder.load_state_dict(drug_params)
    share_en = shared_encoder(noise_flag=True, norm_flag=True)
    share_en.load_state_dict(shared_en_params)
    share_de = shared_decoder()
    share_de.load_state_dict(shared_de_params)
    micro_en = micro_encoder(noise_flag=True, norm_flag=True)
    micro_en.load_state_dict(micro_en_params)
    cls = classifier()
    cls.load_state_dict(cls_params)
    cls_m = cls_mirco()
    stack_layer = stack()
    # c_model = Model(drug_encoder=drug_encoder,label=True,cell=True,
    #                 shared_encoder=share_en,shared_decoder=share_de,micro_encoder=micro_en,classifier=cls).to(device)
    p_model = Model(drug_encoder=drug_encoder,label=True,tcga=True,
                    shared_encoder=share_en,shared_decoder=share_de,micro_encoder=micro_en,classifier=cls,cls_micro=cls_m,stack = stack_layer).to(device)
    p_no_train = Model(drug_encoder=drug_encoder,label=True,tcga=True,
                    shared_encoder=share_en,shared_decoder=share_de,micro_encoder=micro_en,classifier=cls).to(device)

    # tcga_data = CellDrugDataset(root='data/pre_data', task='tcga_5_flu')
    #cell_data = CellDrugDataset(root='data/pre_data', task='cell')
    tcga_val_data = CellDrugDataset(root='data/pre_data', task='tcga_label_gsva')  # all patient and drug samples
    #pdx_data = CellDrugDataset(root='data/pre_data', task='pdx_label')


    # length_tcga = len(tcga_data)

    tcga_train_2,tcga_val_2 = train_test_split(tcga_val_data,test_size=0.2,shuffle=True)
    tcga_train_2, tcga_test = train_test_split(tcga_train_2, test_size=0.995, shuffle=True)

    tcga_train_loader = DataLoader(tcga_train_2,batch_size=2048)
    tcga_val_loader = DataLoader(tcga_val_2,batch_size=2048)
    #tcga_test_loader = DataLoader(tcga_test,batch_size=64)


    model_params = [#drug_encoder.parameters(),

                    cls_m.parameters(),
                    stack_layer.parameters()
                    ]

    optimizer = torch.optim.AdamW(chain(*model_params),lr=0.001,weight_decay=1e-6)

    if not os.path.exists('results/model_' + str(label)):
        os.makedirs('results/model_' + str(label))
    save_name = 'results/model_' + str(label)
    save_file = '{}_'.format(epochs)
    model_file_name = save_name + '/' + save_file + 'p_model.pkl'
    cls_m_name = save_name + '/' + save_file + 'cls_m.pkl'
    encoder_file_name = save_name + '/' + save_file + 'c_model.pkl'
    pic = save_name+'/'+'.png'
    # train_file_AUCs = save_name + '/' + save_file + str(i) + '--train_AUCs--' + task + '.txt'
    # test_file_AUCs = save_name + '/' + save_file + str(i) + '--test_AUCs--' + task + '.txt
    train_losses = []
    valid_losses = []
    c_recon_train = []
    c_recon_val = []
    p_recon_train = []
    p_recon_val = []
    ortho_train = []
    ortho_val = []
    c_cls_train = []
    c_cls_val = []
    p_cls_train = []
    p_cls_val =[]
    c_auc_train = []
    c_auc_val = []
    p_auc_train = []
    p_auc_val = []
    mmd_train = []
    mmd_val = []
    pdx_auc_val = []
    pdx_only_val = []
    stopping_monitor = 0
    best_loss = float('inf')
    best_auc=0
    #writer = SummaryWriter(log_dir="runs/result_1", flush_secs=120)


    tcga_train_domain = []
    tcga_train_micro = []
    tcga_val_domain = []
    tcga_val_micro = []
    tcga_valid_auc = []
    best_auc_2 = 0
    best_loss2 = float('inf')
    # #第二阶段训练
    for epoch in range(1000):
        loss_domain, preds_1, labels = train_target(p_model,device,tcga_train_loader,epoch,optimizer)
        tcga_train_domain.append(loss_domain)
        loss_domain_2,preds_1_val, labels_2,tcga_total,cls_out_1,cls_out_2  = predicting_target(p_model,device,tcga_val_loader,epoch)
        tcga_val_domain.append(loss_domain_2)
        auc = roc_auc_score(labels_2, preds_1_val)
        p_auc_val.append(auc)
        print('AUC tcga is {}'.format(auc))
        if auc>best_auc:
            best_auc = auc
            stopping_monitor = 0
            #torch.save(p_model.state_dict(), model_file_name)
        else:
            stopping_monitor += 1
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        for name, param in stack_layer.named_parameters():
            print(name, param.data)
    torch.save(p_model.state_dict(), model_file_name)
    print(best_auc)
    loss_domain_2, preds_1_val, labels_2, tcga_total, cls_out_1, cls_out_2 = predicting_target(p_model, device,
                                                                                               tcga_val_loader, epoch)

    f = open("cls.txt", "w")
    f2 = open("cls2.txt", "w")
    f.writelines(str(cls_out_1))
    f.close()
    f2.writelines(str(cls_out_2))
    f2.close()
    plt.figure(figsize=(12, 8))
    #
    #
    #
    #
    #
    #
    plt.subplot(2, 2, 4)
    plt.plot(p_auc_val, color='red', label='p_auc')
    plt.legend()
    plt.savefig( 'cls_phase_2.jpg')
    # auc = roc_auc_score(T, P)
    # print('AUC calculated by sklearn tool is {}'.format(auc))


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



