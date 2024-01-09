import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
import globalvar as gl
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
device = gl.get_value('cuda')

from copy import deepcopy

class Model(nn.Module):
    def __init__(self,latent_dim=128,noise_flag:bool=False,norm_flag:bool=False,
                 shared_encoder = None,shared_decoder = None,micro_encoder = None,classifier = None,cls_micro = None,label:bool = False,cell:bool = False,tcga:bool = False,
                 drug_encoder = None,response_layer = None,stack = None,combine:bool=False,**kwargs):
        super(Model,self).__init__()
        self.latent_dim=latent_dim
        self.noise_flag = noise_flag
        self.norm_flag = norm_flag
        self.shared_encoder = shared_encoder

        self.response_layer = response_layer

        self.shared_decoder = shared_decoder
        self.micro_encoder = micro_encoder
        self.classifier = classifier
        self.cls_micro = cls_micro
        self.label = label
        self.cell = cell
        self.tcga = tcga
        self.combine = combine
        #drug
        self.drug_gat = drug_encoder
        #self.drug_fc_g1 = nn.Linear(128, latent_dim)
        self.stack = stack

    def forward(self,data1):
        if self.combine == True :
            x1, edge_index1, batch1, x2,edge_index2,input = data1.x1, data1.edge_index1, data1.batch, data1.x2, data1.edge_index2,data1.input,
            # deal drug
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2,dim=1)
            with torch.no_grad():
                x1, arr1 = self.drug_gat(x1, edge_index1, batch1)
                x2, arr2 = self.drug_gat(x2, edge_index2, batch1)
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1)
            input = F.normalize(input, dim=1)
            embedding = self.shared_encoder(input)

            embedding2 = self.micro_encoder(input)
            lx_1 = torch.cat((embedding, x1), 1)
            # lx_1 = torch.sum(torch.stack([embedding, x1], dim=0), dim=0)
            lx_1 = F.normalize(lx_1, dim=1)

            cls_out_1 = self.classifier(lx_1)
            # cls_out_1 = cls_out_1.detach()
            lx_2 = torch.cat((embedding2, x2), 1)
            # lx_2 = torch.sum(torch.stack([embedding2, x1], dim=0), dim=0)
            lx_2 = F.normalize(lx_2, dim=1)
            cls_out_2 = self.cls_micro(lx_2)
            cls_out = torch.cat((cls_out_1, cls_out_2), dim=1)
            cls_out = self.stack(cls_out)
            return input,recon,embedding,embedding2,latent,cls_out_1,cls_out_2,cls_out,cls_y

        if self.label:
            x1, edge_index1, batch1, input, y1 = data1.x, data1.edge_index, data1.batch, data1.input, data1.y1
            # deal drug
            x1 = F.normalize(x1,dim=1)
            with torch.no_grad():
                x1, arr1 = self.drug_gat(x1, edge_index1, batch1)

            x1 = F.normalize(x1,dim=1)
            input = F.normalize(input,dim=1)

            if self.cell == True and self.tcga == False:

                embedding = self.shared_encoder(input)

                recon = self.shared_decoder(embedding)
                recon = F.normalize(recon,dim=1)
                ex = torch.cat((embedding,x1),1)
                #ex = torch.sum(torch.stack([embedding, x1], dim=0), dim=0)
                ex = F.normalize(ex, dim=1)
                #res_ex = self.response_layer(ex)
                cls_out = self.classifier(ex)

                #cls_out = torch.sigmoid(cls_out)
                cls_y = y1.view(-1,1)

                return input,embedding,recon,cls_out,cls_y

            if self.cell == False and self.tcga == True:
                embedding = self.shared_encoder(input)

                embedding2 = self.micro_encoder(input)

                latent = torch.mean(torch.stack([embedding, embedding2], dim=0), dim=0)
                # latent = torch.cat((embedding,embedding2),1)
                latent = F.normalize(latent, dim=1)
                recon = self.shared_decoder(latent)
                recon = F.normalize(recon, dim=1)
                lx_1 = torch.cat((embedding,x1),1)
                #lx_1 = torch.sum(torch.stack([embedding, x1], dim=0), dim=0)
                lx_1 = F.normalize(lx_1, dim=1)
                #res_lx_1 = self.response_layer(lx_1)
                cls_out_1 = self.classifier(lx_1)
                cls_out_1 = cls_out_1.detach()
                lx_2 = torch.cat((embedding2, x1), 1)
                #lx_2 = torch.sum(torch.stack([embedding2, x1], dim=0), dim=0)
                lx_2 = F.normalize(lx_2, dim=1)
                cls_out_2 = torch.Tensor()
                if self.cls_micro is not None:
                    cls_out_2 = self.cls_micro(lx_2)
                    cls_out = torch.cat((cls_out_1,cls_out_2),dim=1)
                    cls_out = self.stack(cls_out)
                    #cls_out = cls_out_1 + cls_out_2
                    #cls_out = cls_out_2
                else:
                    cls_out = cls_out_1
                # print('output1 is {}'.format(cls_out_1))
                # print('-------------')
                # print('output2 is {}'.format(cls_out_2))
                # print('------------')
                # lx = torch.cat((embedding,x1),1)
                # lx = F.normalize(lx,dim=1)
                # cls_out = self.classifier(lx)
                #cls_out = torch.sigmoid(cls_out)
                cls_y = y1.view(-1,1)
                return input,recon,embedding,embedding2,latent,cls_out_1,cls_out_2,cls_out,cls_y

        if self.label == False:
            input= data1.input
            input = F.normalize(input, dim=1)

            if self.cell == True and self.tcga == False:
                embedding = self.shared_encoder(input)

                recon = self.shared_decoder(embedding)
                recon = F.normalize(recon, dim=1)
                return input,embedding,recon
            if self.cell == False and self.tcga == True:
                embedding = self.shared_encoder(input)

                embedding2 = self.micro_encoder(input)

                #latent = torch.cat((embedding, embedding2), 1)
                latent = torch.mean(torch.stack([embedding, embedding2], dim=0), dim=0)
                latent = F.normalize(latent, dim=1)
                recon = self.shared_decoder(latent)
                recon = F.normalize(recon, dim=1)
                return input,embedding,embedding2,latent,recon
    def lossfunction(self,*args,**kwargs):
        if self.label:
            if self.cell:
                input = args[0]
                recon = args[1]
                cls_out = args[2]
                cls_y = args[3]

                recon_fn = nn.MSELoss()
                recon_loss = recon_fn(input,recon)
                cls = nn.BCEWithLogitsLoss()
                cls_loss = cls(cls_out, cls_y)
                loss = recon_loss + cls_loss
                return {'loss':loss,'recon_loss':recon_loss,'cls_loss':cls_loss}

            elif self.tcga:
                input = args[0]
                recon = args[1]
                p1 = args[2]
                p2 = args[3]
                cls_out=args[4]
                cls_y = args[5]

                recon_fn = nn.MSELoss()
                recon_loss = recon_fn(input, recon)
                cls = nn.BCEWithLogitsLoss()
                cls_loss = cls(cls_out,cls_y)

                p1_l2_norm = torch.norm(p1, p=2, dim=1, keepdim=True).detach()
                p1_l2 = p1.div(p1_l2_norm.expand_as(p1) + 1e-6)

                p2_l2_norm = torch.norm(p2, p=2, dim=1, keepdim=True).detach()
                p2_l2 = p2.div(p2_l2_norm.expand_as(p2) + 1e-6)

                ortho_loss = torch.mean(torch.square(torch.diagonal(torch.matmul(p1_l2, p2_l2.t()))))

                loss = recon_loss + cls_loss + ortho_loss
                return {'loss':loss,'recon_loss':recon_loss,'cls_loss':cls_loss,'ortho_loss':ortho_loss}

        if self.label == False:
            if self.cell:
                input = args[0]
                recon = args[1]

                recon_loss = F.mse_loss(input, recon)


                loss = recon_loss

                return {'loss': loss, 'recon_loss': recon_loss}

            if self.tcga:
                input = args[0]
                recon = args[1]
                p1 = args[2]
                p2 = args[3]
                recon_loss = F.mse_loss(input, recon)

                p1_l2_norm = torch.norm(p1, p=2, dim=1, keepdim=True).detach()
                p1_l2 = p1.div(p1_l2_norm.expand_as(p1) + 1e-6)

                p2_l2_norm = torch.norm(p2, p=2, dim=1, keepdim=True).detach()
                p2_l2 = p2.div(p2_l2_norm.expand_as(p2) + 1e-6)

                # diagonal = torch.diag((p1_l2.t().mm(p2_l2)).pow(2)).float()
                ortho_loss = torch.mean(torch.square(torch.diagonal(torch.matmul(p1_l2, p2_l2.t()))))
                loss = recon_loss + ortho_loss

                return {'loss': loss, 'recon_loss': recon_loss,'ortho_loss': ortho_loss}


