import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
import globalvar as gl
import os
from typing import Any, Optional, Tuple
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
device = gl.get_value('cuda')
#from gradient_reversal import RevGrad
from torch.autograd import Function

class response_layer(nn.Module):
    def __init__(self, input_dim=128, out_dim=128, dropout=0.2, **kwargs):
        super(response_layer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout

        modules = []
        # hidden_dims = deepcopy(h_dims)
        # hidden_dims.insert(0,input_dim)

        # build encoder1
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, out_dim),
            )
        )
        self.encoder = nn.Sequential(*modules)

    def encode(self, input: Tensor):
        embedding = self.encoder(input)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, input):
        embedding = self.encode(input)
        return embedding


class shared_encoder(nn.Module):
    def __init__(self, input_dim=1328, out_dim=128, dropout=0.5, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs):#17171gene
        super(shared_encoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.noise_flag = noise_flag
        self.norm_flag = norm_flag

        modules = []
        # hidden_dims = deepcopy(h_dims)
        # hidden_dims.insert(0,input_dim)

        # build encoder1
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim),
            )
        )
        self.encoder = nn.Sequential(*modules)

    def encode(self, input: Tensor):
        if self.noise_flag:
            embedding = self.encoder(input + torch.randn_like(input, requires_grad=False) * 0.01)
        else:
            embedding = self.encoder(input)

        if self.norm_flag:
            return F.normalize(embedding, p=2, dim=1)
        else:
            return embedding

    def forward(self, input):
        embedding = self.encode(input)
        return embedding


class micro_encoder(nn.Module):
    def __init__(self, input_dim=1328, out_dim=128, dropout=0.5, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs):#1328
        super(micro_encoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.noise_flag = noise_flag
        self.norm_flag = norm_flag

        modules = []
        # hidden_dims = deepcopy(h_dims)
        # hidden_dims.insert(0,input_dim)

        # build encoder1
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim),
            )
        )
        self.encoder = nn.Sequential(*modules)

    def encode(self, input: Tensor):
        if self.noise_flag:
            embedding = self.encoder(input + torch.randn_like(input, requires_grad=False) * 0.01)
        else:
            embedding = self.encoder(input)

        if self.norm_flag:
            return F.normalize(embedding, p=2, dim=1)
        else:
            return embedding

    def forward(self, input):
        embedding = self.encode(input)
        return embedding


# share_decoder
class shared_decoder(nn.Module):
    def __init__(self, latent_dim=128, out_dim=1328, dropout=0.5,
                 **kwargs):
        super(shared_decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.dropout = dropout

        modules = []
        # hidden_dims = deepcopy(h_dims)
        # hidden_dims.insert(0,input_dim)

        # build encoder1
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout),
                nn.Linear(1024,out_dim),
            )
        )
        self.decoder = nn.Sequential(*modules)

    def decode(self, z: Tensor):
        result = self.decoder(z)
        return result

    def forward(self, input):
        recon = self.decode(input)
        return recon


class classifier(nn.Module):
    def __init__(self, latent_dim=256,dropout = 0.2,
                 **kwargs):
        super(classifier, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout

        modules = []
        modules.append(
            nn.Sequential(

                nn.Linear(256, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(dropout),

                nn.Linear(32, 1),

                
                
            )
        )
        self.classifier = nn.Sequential(*modules)

    def forward(self, input):
        cls_out = self.classifier(input)
        return cls_out


class cls_mirco(nn.Module):
    def __init__(self, latent_dim=256,
                 **kwargs):
        super(cls_mirco, self).__init__()
        self.latent_dim = latent_dim
        #self.dropout = dropout

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                #nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                #nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                #nn.Dropout(dropout),
                nn.Linear(16,1),
                # nn.ReLU(),
                # nn.BatchNorm1d(2),
                # # nn.Dropout(dropout),
                # nn.Linear(2, 1),
            )
        )
        self.classifier = nn.Sequential(*modules)

    def forward(self, input):
        cls_out = self.classifier(input)
        return cls_out

class GradReverse(torch.autograd.Function):
    """
        重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)



class discriminator(nn.Module):
    def __init__(self, inc=128):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 64)
        self.fc2_1 = nn.Linear(64, 32)
        self.fc3_1 = nn.Linear(32, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x = self.fc3_1(x)
        return x


class stack(nn.Module):
    def __init__(self, inc=2):
        super(stack, self).__init__()
        self.fc1 = nn.Linear(inc, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x