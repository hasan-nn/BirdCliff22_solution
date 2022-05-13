
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np

from timm import create_model

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()
        
def split(x,num_splits = 6):
    b,c,h,w = x.size()
    cutoff = w // num_splits * num_splits
    
    x = x[:,:,:,:cutoff]
    #print(x.shape)
    b,c,h,w = x.size()

    ksize = stride = (h,w // num_splits)

    patches = x.unfold(2, ksize[0], stride[0]).unfold(3, ksize[1], stride[1]).squeeze(2).permute(0,2,1,3,4)

    patches = patches.contiguous().view(b * num_splits,*patches.shape[2:])

    return patches

def merge(patches,num_splits = 6):
    b,c,h,w = patches.size()

    stitched = patches.contiguous().view(b // num_splits, c, -1, h*w).permute(0, 1, 3, 2) 
    stitched = stitched.contiguous().view(b // num_splits, c*h*w, -1)
    stitched = F.fold(stitched, output_size=(h, w*num_splits), kernel_size=(h,w), stride=(h,w))

    return stitched

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamped_Sigmoid(nn.Module):
    def __init__(self,min=1e-2,max=1.0-1e-2):
        super().__init__()
        self._min = min
        self._max = max
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        x = torch.clamp(x,min=self._min,max=self._max)
        return x


class Activation(nn.Module):

    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'clamped_sigmoid':
            self.activation = Clamped_Sigmoid(**params)
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class Backbone(nn.Module):
    def __init__(self,name = 'resnet18',in_channels=1,pretrained=True,depth=5):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )
        self.backbone = create_model(name, **kwargs)
        self.in_channels = in_channels
        self.features_channels = self.backbone.feature_info.channels()
        self.depth = depth

    
    def forward(self,x):
        return self.backbone(x)


class Conv3x3(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel):
        conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,padding='same',padding_mode='zeros')
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        super().__init__(conv,bn,relu)

class Classification_Head(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.2, activation=None):

        dropout = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(dropout, linear, activation)

class Classification_HeadV4(nn.Sequential):
    def __init__(self, in_channels,classes, dropout=0.2, activation=None,groups=8,groups_out=1):
        out_channels = in_channels // groups * groups_out
        if out_channels % groups != 0:
            out_channels = int(math.ceil(out_channels / groups) * groups)

        print(f'Classification Head: in_channels : {in_channels}, out_channels : {out_channels}, groups : {groups}')


        conv0 = nn.Conv1d(in_channels,out_channels,kernel_size=2,stride=1,padding=0,bias=False,groups=groups)
        bn0 = nn.BatchNorm1d(out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(p=dropout, inplace=False)
        conv1 = nn.Conv1d(out_channels,classes,kernel_size=1,stride=1,padding=0,bias=True)
        activation = Activation(activation)
        super().__init__(conv0,bn0,relu,dropout,conv1,activation)



class Attention_FT(nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self,x):
        b,c,h,w = x.shape
        x = self.conv1(x)

        f_avg = x.mean(dim=-1).unsqueeze(-1)
        t_avg = x.mean(dim=-2).unsqueeze(-2)
 
        att_map = torch.matmul(f_avg,t_avg)
        att_map = self.softmax(att_map.view(b,c,-1)).view(b,c,h,w)
        x_att = x * att_map

        x = x + x_att
        x = self.relu(x)
        return x

class CoW(nn.Module):
    def __init__(self,in_channels,coeffs_channels):
        super().__init__()
        mods = []
        ins_chs = [in_channels,*coeffs_channels]
        for i,coeff_channels in enumerate(coeffs_channels):
            mods.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=ins_chs[i],out_channels=coeff_channels,kernel_size=1,bias=False,padding=0),
                    nn.BatchNorm1d(coeff_channels),
                    nn.PReLU(),
                )
            )
        
        mods.append(
            nn.Sequential(
                    nn.Conv1d(in_channels=coeffs_channels[-1],out_channels=1,kernel_size=1,bias=False,padding=0),
                    nn.Softmax(dim=-1)
                )

        )

        self.mods = nn.ModuleList(mods)
    
    def forward(self,x):
        for mod in self.mods:
            x = mod(x)
        x = x.permute(0,2,1)
        #x = torch.clamp(x,min=0.1)
        #s = x.sum(dim=1).unsqueeze(1)
        #x = x / s
        return x

class SED_V5(nn.Module):

    def __init__(
        self,
        backbone = 'resnet18',
        in_channels=1,
        pretrained_backbone =True,
        backbone_depth = 5,
        num_classes=152,
        dropout=0.2,
        activation=None,
        k=8,
        k2=1,
        use_attention = True
        ):
        super().__init__()


        self.k = k
        self.use_attention = use_attention

        self.backbone = Backbone(name=backbone,in_channels=in_channels,pretrained=pretrained_backbone,depth=backbone_depth)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()

        feature_channels = self.backbone.features_channels[-1]
        print(f'Backbone {backbone} with {feature_channels} feature channels...')

        if feature_channels > 512:
            self.bot = Conv3x3(feature_channels,512,1)
            feature_channels = 512
            print(f'Adjusted N# Features to 512....')
        else:
            self.bot = None

        
        self.fc = Classification_HeadV4(feature_channels * k,num_classes,dropout=dropout,activation=activation,groups=k,groups_out=k2)

        if use_attention:
            self.att = Attention_FT(feature_channels)
        

    def forward(self,x,num_splits=None):

        b,c,h,w = x.size()
        if num_splits is not None:
            x = split(x,num_splits)
        
        features = self.backbone(x)[-1]
        if self.bot is not None:
            features = self.bot(features)

        if self.use_attention:
            features = self.att(features)

        oc = features.shape[1]
        features = features.reshape(b,num_splits,oc,-1)
        features = features.permute(0,2,1,3).reshape(b,oc,-1)
        features,indices = torch.topk(features,k=self.k,dim=-1)
        
        indices = torch.div(indices % (h*w), w, rounding_mode='trunc').float()  / h
        features = features.reshape(b,-1,1)
        indices = indices.reshape(b,-1,1)

        samples = torch.cat([features,indices],dim=-1)

        y_out = self.fc(samples)
        y_out = y_out[:,:,0]
        return y_out 
    
    

class SED_V7(nn.Module):

    def __init__(
        self,
        backbone = 'resnet34',
        in_channels=1,
        pretrained_backbone =True,
        backbone_depth = 5,
        num_classes=152,
        dropout=0.2,
        activation=None,
        coeff_channels = [128,64],
        num_splits=6
        ):
        super().__init__()



        self.num_splits = num_splits
        self.backbone = Backbone(name=backbone,in_channels=in_channels,pretrained=pretrained_backbone,depth=backbone_depth)
        self.pool = GeM()

        feature_channels = self.backbone.features_channels[-1]
        print(f'Backbone {backbone} with {feature_channels} feature channels...')


        self.cow = CoW(feature_channels,coeff_channels)
        self.fc = Classification_Head(feature_channels,num_classes,dropout,activation)


        

    def forward(self,x):

        b,c,h,w = x.size()

        x = split(x,self.num_splits)

        features = self.backbone(x)[-1]
        features = self.pool(features)
        features = features[:,:,0,0]


        _,ch = features.size()
        features = features.view(-1,self.num_splits,ch)

        weights = self.cow(features.permute(0,2,1))
        w_features = (features * weights).sum(dim=1)
        y_out = self.fc(w_features)

        return y_out,weights.squeeze(-1)
    
    def forward_5s(self,x):

        b,c,h,w = x.size()

        #x = split(x,self.num_splits)

        features = self.backbone(x)[-1]
        features = self.pool(features)
        features = features[:,:,0,0]


        y_out = self.fc(features)
        return y_out
    
    def forward_debug(self,x):

        b,c,h,w = x.size()

        x = split(x,self.num_splits)

        features = self.backbone(x)[-1]
        features = self.pool(features)
        features = features[:,:,0,0]


        _,ch = features.size()
        features = features.view(-1,self.num_splits,ch)

        weights = self.cow(features.permute(0,2,1))
        print(weights)
        w_features = (features * weights).sum(dim=1)
        y_out = self.fc(w_features)
        return y_out


        
def create_sed_model(name='sed_v1',**kwargs):

    if name == 'sed_v5':
        return SED_V5(**kwargs)
    elif name == 'sed_v7':
        return SED_V7(**kwargs)

    else:
        raise NotImplementedError(f'SED model {name} is not implemented')

if __name__ == '__main__':

    """
    x = torch.rand([6, 1, 224, 1350])
    kwargs = dict(
        backbone = 'regnety_008',
        in_channels=1,
        pretrained_backbone =False,
        backbone_depth = 5,
        num_classes=152,
        dropout=0.2,
        activation=None,
        coeff_channels = [128,64],
        num_splits=6
    )
    net = create_sed_model(name='sed_v7',**kwargs)

    with torch.no_grad():
        y,cow_weights = net(x)
    print(y.shape)
    print(cow_weights.shape)
    """


    