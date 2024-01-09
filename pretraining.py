import os
import argparse
import random

import pandas as pd
import torch.nn.functional as F
import torch

from itertools import chain, cycle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve, r2_score, mean_squared_error, balanced_accuracy_score, confusion_matrix, \
    precision_score, accuracy_score, cohen_kappa_score, recall_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.cuda.set_device(0)
import globalvar as gl
from encoder_decoder import shared_encoder, shared_decoder, micro_encoder, discriminator,classifier
import umap

gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
    print('The code uses GPU...')
    print(torch.cuda.current_device())
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
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split
import torch.nn as nn
import time
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
def pretrain(c_model, p_model, dis,cell_loader, tcga_loader, dev, optimizer, epoch):
    since = time.time()
    c_model.zero_grad()
    p_model.zero_grad()
    dis.zero_grad()
    c_model.train()
    p_model.train()
    dis.train()
    total = 0
    total_c_cls = 0
    total_p_cls = 0
    total_c_recon = 0
    total_p_recon = 0
    total_ortho = 0
    total_dann = 0

    for batch_idx, data in enumerate(zip(cell_loader,cycle(tcga_loader))):
        data1 = data[0].to(dev)
        data2 = data[1].to(dev)

        cell, c_embedding, c_recons,c_cls,c_label = c_model(data1)

        tcga,t_embedding,t_embedding2,t_latent,t_recon = p_model(data2)

        embedding = torch.cat((c_embedding, t_embedding), dim=0)
        #embedding = F.normalize(embedding, dim=1)
        out = dis(embedding)
        # print(out)
        truth = torch.cat((torch.zeros(c_embedding.shape[0], 1), torch.ones(t_embedding.shape[0], 1)),
                          dim=0).view(-1, 1).to(dev)
        # print(truth)
        loss_fn = nn.BCEWithLogitsLoss()
        dann = loss_fn(out, truth)
        c_model_loss = c_model.lossfunction(cell, c_recons,c_cls,c_label)
        p_model_loss = p_model.lossfunction(tcga, t_recon,t_embedding,t_embedding2)
        #mmd2 = mmd_loss(c_embedding, t_embedding)

        loss = c_model_loss['loss'] + p_model_loss['loss'] + dann

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        total_c_recon += c_model_loss['recon_loss'].item()
        total_p_cls += p_model_loss['recon_loss'].item()
        total_c_cls += c_model_loss['cls_loss'].item()
        total_ortho += p_model_loss['ortho_loss'].item()
        total_dann += dann.item()

    epo_loss = total / (batch_idx+1)
    epo_c_recon = total_c_recon / (batch_idx + 1)
    epo_p_recon = total_p_recon / (batch_idx + 1)
    epo_c_cls = total_c_cls / (batch_idx+1)

    epo_ortho = total_ortho / (batch_idx + 1)

    epo_dann = total_dann / (batch_idx+1)
    print(
        'Train epoch: {} \tLoss: {:.6f}\tC_cls_Loss: {:.6f}\tdann_Loss: {:.6f}'.format(
            epoch, epo_loss,
            epo_c_cls,
            epo_dann
        ))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return epo_loss, epo_c_recon, epo_p_recon, epo_c_cls,epo_ortho,epo_dann


def valid(c_model,device, cell_loader):
    print('Valid on {} cell_drug samples...'.format(len(cell_loader.dataset)))
    c_model.eval()
    c_total_preds_val = torch.Tensor()
    c_total_labels_val = torch.Tensor()
    total = 0
    total_c_cls = 0
    cell_total = torch.Tensor()

    with torch.no_grad():
        for batch_idx, data in enumerate(zip(cell_loader)):
            data1 = data[0].to(device)
            cell, c_embedding, c_recons, pre_out, pre_y = c_model(data1)
            pre_out = pre_out.to('cpu')
            pre_y = pre_y.to('cpu')
            c_total_preds_val = torch.cat((c_total_preds_val, pre_out), 0)
            c_total_labels_val = torch.cat((c_total_labels_val, pre_y), 0)
            c_embedding = c_embedding.to('cpu')
            cell_total = torch.cat((cell_total,c_embedding),0)
            c_model_loss = c_model.lossfunction(cell, c_recons, pre_out, pre_y)
            loss = c_model_loss['cls_loss']
            total += loss.item()
            total_c_cls += c_model_loss['cls_loss'].item()

    epo_loss = total / (batch_idx + 1)
    epo_cls = total_c_cls / (batch_idx + 1)

    print(
        'Loss: {:.6f}'.format(
            epo_cls

        ))
    return epo_cls , c_total_preds_val, c_total_labels_val,cell_total

def predicting_target(p_model,device,tcga_loader,epoch):
    print('Predict on {} tcga_drug samples...'.format(len(tcga_loader.dataset)))
    p_model.eval()
    p_total_preds = torch.Tensor()
    p_total_labels = torch.Tensor()

    total = 0
    total_p_recon = 0

    total_p_cls = 0
    tcga_total = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(tcga_loader)):
            data1 = data[0].to(device)
            p, p_recons, p_embedding_1, p_embedding_2, p_embedding, cls_out_1,cls_out_2,cls_out, cls_y = p_model(data1)
            p_embedding_1_copy = p_embedding_1.to('cpu')
            tcga_total = torch.cat((tcga_total,p_embedding_1_copy),0)
            cls_out = cls_out.to('cpu')

            cls_y = cls_y.to('cpu')
            p_total_preds = torch.cat((p_total_preds, cls_out), 0)
            p_total_labels = torch.cat((p_total_labels, cls_y), 0)

            p_model_loss = p_model.lossfunction(p, p_recons, p_embedding_1, p_embedding_2, cls_out, cls_y)
            loss = p_model_loss['cls_loss']

            total += loss.item()

            total_p_cls += p_model_loss['cls_loss'].item()
            total_p_recon += p_model_loss['recon_loss'].item()

    epo_p_cls = total_p_cls / (batch_idx + 1)
    epo_p_recon = total_p_recon / (batch_idx + 1)
    print(
        'Train epoch: {} \tcls_Loss: {:.6f}'.format(
            epoch,
            epo_p_cls

        ))
    return epo_p_cls,epo_p_recon, p_total_preds, p_total_labels,tcga_total


def main():
    label = 'adv_pretraining_7'

    cell_feature = CellDrugDataset(root='data/pre_data', task='ccle_pretrain_gsva')
    tcga_feature = CellDrugDataset(root='data/pre_data', task='tcga_pretrain_gsva')
    tcga_valid_feature = CellDrugDataset(root='data/pre_data', task='tcga_label_gsva')

    cell_train, cell_val = train_test_split(cell_feature, test_size=0.2, shuffle=True)

    cell_train_loader = DataLoader(cell_train, batch_size=2048)
    cell_val_loader = DataLoader(cell_val,batch_size = 2048)
    tcga_pretrain_loader = DataLoader(tcga_feature, batch_size=2048,shuffle = True)
    tcga_val_loader = DataLoader(tcga_valid_feature,batch_size = 2048,shuffle = True)
    # total_cell = torch.Tensor()
    # total_tcga = torch.Tensor()
    # for data1 in cell_feature:
    #     data1 = data1.input
    #     total_cell = torch.cat((total_cell, data1), 0)
    # for data2 in tcga_feature:
    #     data2 = data2.input
    #     total_tcga = torch.cat((total_tcga, data2), 0)
    #
    # features = torch.cat((total_cell, total_tcga), dim=0)
    # print(features)
    # umap_before = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(features)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.scatter(umap_before[:len(total_cell), 0], umap_before[:len(total_cell), 1], c="#219EBC",s = 20,label = 'GDSC')
    # plt.scatter(umap_before[len(total_cell):, 0], umap_before[len(total_cell):, 1], c="#FA8600",s = 20,label = 'TCGA')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.legend()
    # plt.savefig('umap_raw.jpg')



    encoder_file = '100_model_encoder_gat1500_100_in-vitro_re_gat_dex25_p1'
    drug_encoder = GATNet()
    params = drug_encoder.state_dict()  # 获得模型的原始状态以及参数。
    checkpoint = torch.load('data/pretrained_drug_model/' + encoder_file + '.pkl', map_location='cuda:0')
    new_state_dict = OrderedDict()
    for k1, k2 in zip(params.keys(), checkpoint.keys()):
        name = k1
        new_state_dict[name] = checkpoint[k2]
    drug_encoder.load_state_dict(new_state_dict)

    drug_encoder = drug_encoder.cuda()

    shared_en = shared_encoder(noise_flag=True, norm_flag=True)

    share_de = shared_decoder()
    micro_en = micro_encoder(noise_flag=True, norm_flag=True)
    dis = discriminator().cuda()
    cls = classifier().cuda()


    p_unlabel_model = Model(shared_encoder=shared_en, shared_decoder=share_de, micro_encoder=micro_en, tcga=True).to(
        device)
    c_model = Model(drug_encoder = drug_encoder,shared_encoder=shared_en, shared_decoder=share_de, micro_encoder=micro_en, cell=True,label=True,classifier=cls)
    p_valid_model = Model(drug_encoder = drug_encoder,shared_encoder=shared_en,shared_decoder=share_de,micro_encoder=micro_en,tcga=True,label=True,classifier=cls).to(device)

    model_params = [  drug_encoder.parameters(),
        shared_en.parameters(),
        micro_en.parameters(),
        share_de.parameters(),

        dis.parameters(),
        cls.parameters()
    ]

    optimizer = torch.optim.AdamW(chain(*model_params), lr=0.0005, weight_decay=1e-6)

    if not os.path.exists('results/model_' + str(label)):
        os.makedirs('results/model_' + str(label))
    save_name = 'results/model_' + str(label)
    save_file = '{}_'.format(500)
    share_en_name = save_name + '/' + save_file + 'share_encoder.pkl'
    share_de_name = save_name + '/' + save_file + 'share_decoder.pkl'
    micro_en_name = save_name + '/' + save_file + 'micro_encoder.pkl'
    cls_name = save_name + '/' + save_file + 'share_cls.pkl'
    drug_name = save_name + '/' + save_file + 'drug.pkl'
    pic = save_name + '/' + '.png'
    # train_file_AUCs = save_name + '/' + save_file + str(i) + '--train_AUCs--' + task + '.txt'
    # test_file_AUCs = save_name + '/' + save_file + str(i) + '--test_AUCs--' + task + '.txt
    train_losses = []
    valid_losses = []
    c_recon_train = []
    c_recon_val = []
    c_cls_train = []
    c_cls_val = []
    p_recon_train = []
    p_recon_val = []
    ortho_train = []
    ortho_val = []
    mmd_train = []
    mmd_val = []
    dann_train = []
    c_auc_val = []
    stopping_monitor = 0
    best_loss = float('inf')
    t_auc_val = []
    p_cls_val = []
    t_auc_val = []
    best_auc = 0
    for epoch in range(500):

        loss, c_recon, p_recon, c_cls,ortho,dann = pretrain(c_model, p_unlabel_model,
                                                                          dis,
                                                                          cell_train_loader,
                                                                          tcga_pretrain_loader, device, optimizer, epoch=epoch)

        train_losses.append(loss)
        c_recon_train.append(c_recon)
        p_recon_train.append(p_recon)
        c_cls_train.append(c_cls)
        dann_train.append(dann)
        c_val_cls , c_total_preds_val, c_total_labels_val,cell_total = valid(c_model,device,cell_val_loader)

        c_cls_val.append(c_val_cls)
        auc_c = roc_auc_score(c_total_labels_val, c_total_preds_val)
        c_auc_val.append(auc_c)
        print(auc_c)
        p_cls, p_recon, p_total_preds, p_total_labels, tcga_total = predicting_target(p_valid_model,device
                                                                                              ,tcga_val_loader,epoch)
        p_recon_val.append(p_recon)
        p_cls_val.append(p_cls)
        auc_t = roc_auc_score(p_total_labels, p_total_preds)
        t_auc_val.append(auc_t)
        print(auc_t)
        if auc_t > best_auc:
            best_auc = auc_t
            stopping_monitor = 0
            print('save model weights')
            torch.save(shared_en.state_dict(), share_en_name)
            torch.save(share_de.state_dict(), share_de_name)
            torch.save(micro_en.state_dict(), micro_en_name)
            torch.save(cls.state_dict(), cls_name)
            torch.save(drug_encoder.state_dict(), drug_name)
        else:
            stopping_monitor += 1
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        # if stopping_monitor > 20:
        #     break
    print(best_auc)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(c_cls_train, color='green', label='valid loss')
    plt.plot(c_cls_val, color='red', label='train loss')

    plt.subplot(2, 3, 2)
    plt.plot(c_auc_val, color='red', label='c_recon_train')
    plt.plot(t_auc_val, color='green', label='c_recon_train')

    plt.subplot(2, 3, 3)
    plt.plot(dann_train, color='red', label='p_recon_train')
    #plt.plot(p_recon_val, color='green', label='p_recon_train')

    # plt.subplot(2, 3, 4)
    # plt.plot(ortho_train, color='red', label='ortho_train')
    # plt.plot(ortho_val, color='green', label='ortho_val')

    # plt.subplot(2, 3, 5)
    # plt.plot(mmd_train, color='red', label='dann_train')
    # plt.plot(mmd_val, color='green', label='dann_val')
    plt.legend()
    plt.savefig('dann_loss.jpg')

    # loss2, c_recon_loss2, p_recon_loss2,  mmd2, p1_total, p2_total = valid(c_unlabel_model, p_unlabel_model,
    #                                                                                    dis,
    #                                                                                    cell_all_loader,
    #                                                                                    tcga_all_loader, device, epoch)
    #
    # features2 = torch.cat((p1_total, p2_total), dim=0)
    #
    # # 将特征降到二维空间中
    # umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(features2)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.scatter(umap_embeddings[:len(p1_total), 0], umap_embeddings[:len(p1_total), 1], c="#FA8600", s=20, label='TCGA')
    # plt.scatter(umap_embeddings[len(p1_total):, 0], umap_embeddings[len(p1_total):, 1], c="#219EBC", s=20, label='GDSC')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.legend()
    # plt.savefig('umap_after.jpg')

    # loss3, c_recon_loss3, p_recon_loss3, ortho_loss3, mmd3, p1_total2, p2_total2 = valid(c_unlabel_model, p_unlabel_model,
    #                                                                                    dis,
    #                                                                                    cell_val_loader,
    #                                                                                    pdx_loader, device, epoch)
    #
    # features2 = torch.cat((p1_total2, p2_total2), dim=0)
    #
    # # 将特征降到二维空间中
    # umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(features2)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.scatter(umap_embeddings[:len(p1_total2), 0], umap_embeddings[:len(p1_total2), 1], c="#d5695d")
    # plt.scatter(umap_embeddings[len(p1_total2):, 0], umap_embeddings[len(p1_total2):, 1], c="#5d8ca8")
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of feature1 and feature2', fontsize=15)
    # plt.savefig('umap_pdx.jpg')


if __name__ == '__main__':
    main()
