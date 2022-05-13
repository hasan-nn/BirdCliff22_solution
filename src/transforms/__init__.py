#import librosa
import random
from turtle import forward
from matplotlib import transforms
from rasterio import pad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.distributions import Beta
from torch_audiomentations import Compose, OneOf, SomeOf, PitchShift ,AddColoredNoise,AddBackgroundNoise,Gain

from audiomentations.core.transforms_interface import BaseWaveformTransform

import numpy as np
import colorednoise as cn

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

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None,cow_weights=None,class_weights=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        X = torch.nan_to_num(X,nan=0.0)

        returns = [X,Y]

        if weight is not None:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            returns.append(weight)

        if cow_weights is not None:
            cow_weights = torch.cat([cow_weights.unsqueeze(1),cow_weights[perm].unsqueeze(1)],dim=1)
            cow_weights,_ = torch.max(cow_weights,dim=1)
            returns.append(cow_weights)
        
        if class_weights is not None:
            class_weights = coeffs.view(-1, 1) * class_weights + (1 - coeffs.view(-1, 1)) * class_weights[perm]
            returns.append(class_weights)
        
        return returns


def addFreqCoords(x):
    b,c,h,w = x.size()
    coords = torch.arange(start=h-1,end=-1,step=-1,device=x.device).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(b,1,1,w) / (h-1)
    x = torch.cat([x,coords],dim=1)
    return x
    
class AddPinkSNR(BaseWaveformTransform):

    supports_multichannel = True

    def __init__(
        self, min_snr_in_db=5, max_snr_in_db=40.0, p=0.5
    ):

        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db


    def apply(self, samples, sample_rate):
        snr = np.random.uniform(self.min_snr_in_db, self.max_snr_in_db)
        a_signal = np.sqrt(samples ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(samples))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (samples + pink_noise * 1 / a_pink * a_noise).astype(samples.dtype)
        return augmented



#from .pcen import StreamingPCENTransform as PCEN
class Normalize(nn.Module):
    def __init__(self,min=-80.0,max=0.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self,x):
        return (x - self.min)/(self.max - self.min)

class MelSpec(nn.Sequential):
    def __init__(
        self,
        nfft=2048,
        hop_length=1250,
        n_mels=128,
        sample_rate=32000,
        stype='magnitude',
        normalize = False
        ):

        assert stype in ['magnitude','power']

        t1 = T.Spectrogram(
            n_fft=nfft,
            hop_length=hop_length,
            normalized=False,
            power = 1 if stype == 'magnitude' else 2,
            pad_mode='constant'
        )

        t2 = T.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=20,
            f_max=20000,
            n_stft=nfft//2+1,
            norm='slaney',
            mel_scale='slaney'
        )

        t3 = T.AmplitudeToDB(
            stype=stype,
            top_db=80.0
        )

        mods = [t1,t2,t3]

        if normalize:
            mods.append(Normalize())
        super().__init__(*mods)

class SpecFrequencyMask(nn.Module):
    def __init__(
        self,
        min_y = 0.2,
        max_y = 0.8,
        min_masked_mels= 0.1,
        max_masked_mels=0.2,
        pad_value = -80.0,
        p=1.0,
        ):

        super().__init__()
        assert min_y < max_y
        assert max_y - max_masked_mels > min_y
        assert 0 < min_masked_mels <= max_masked_mels
        self.maxy = max_y - max_masked_mels 
        self.miny = min_y
        self.min_mm = min_masked_mels
        self.max_mm = max_masked_mels
        self.p = p
        self.pad_value = pad_value

        self.dist1 = torch.distributions.Uniform(low=self.miny,high=self.maxy,validate_args=True)
        self.dist2 = torch.distributions.Uniform(low=self.min_mm,high=self.max_mm,validate_args=True)

    
    def forward(self,x):
        if not self.training:
            return x

        b,c,h,w = x.size()

        for i in range(b):
            if torch.rand(1) > self.p:
                continue
            
            start = self.dist1.sample().item()
            width = self.dist2.sample().item()

            start = int(start * h)
            width = int(width * h)
            x[i,:,start:start+width,:] = self.pad_value
        
        return x

class SpecTimeMask(nn.Module):
    def __init__(
        self,
        min_masked_steps = 0.2,
        max_masked_steps = 0.5,
        pad_value = -80.0,
        p=1.0,
        ):
        super().__init__()
        assert 0 < min_masked_steps < max_masked_steps < 1.0
        self.min_ms = min_masked_steps
        self.max_ms = max_masked_steps
        self.p = p
        self.pad_value = pad_value

        self.dist1 = torch.distributions.Uniform(low=0,high=1.0,validate_args=True)
        self.dist2 = torch.distributions.Uniform(low=self.min_ms,high=self.max_ms,validate_args=True)

    
    def forward(self,x):
        if not self.training:
            return x

        b,c,h,w = x.size()

        for i in range(b):
            if torch.rand(1) > self.p:
                continue

            start = self.dist1.sample().item()
            width = self.dist2.sample().item()

            start = int(start * w)
            width = int(width * w)
            x[i,:,:,start:start+width] = self.pad_value
        
        return x


class Transform_01(nn.Module):
    def __init__(
        self,
        nfft=2048,
        hop_length=1250,
        n_mels=128,
        sample_rate=32000,
        stype='magnitude',
        normalize = False
        ):
        super().__init__()
        self.pre_tfm = Compose(
            transforms=[

                OneOf(
                    transforms=[

                        AddBackgroundNoise(
                                background_paths='/home/hasan/Desktop/kaggle/datasets/ff1010bird_wav/background',
                                min_snr_in_db=0,
                                max_snr_in_db=3,
                                mode='per_batch',
                                p = 1.0,
                                sample_rate=sample_rate,
                            ),
                            ],p=0.25),

            ],p=1.0)
                
        

        self.mel_tfm = MelSpec(nfft,hop_length,n_mels,sample_rate,stype,normalize)

    def forward(self,x):
        #print(x)
        x = self.pre_tfm(x)
        #print(x)
        x = torch.nan_to_num(x,nan=0.0)
        x = self.mel_tfm(x)
        x = torch.nan_to_num(x,nan=0.0)
        return x
        

class Transform_02(nn.Module):
    def __init__(
        self,
        nfft=2048,
        hop_length=1250,
        n_mels=128,
        sample_rate=32000,
        stype='magnitude',
        normalize = False,
        num_splits = 6.0,
        ):
        super().__init__()
        self.num_splits = num_splits

        self.gain = Gain(min_gain_in_db=-20, max_gain_in_db = 6, p=0.8,mode='per_example')

        self.noise = AddColoredNoise(
                        min_snr_in_db = -6,
                        max_snr_in_db = 3,
                        min_f_decay=-2,
                        max_f_decay=2,
                        p=0.8,
                        sample_rate=sample_rate,
                        mode='per_example'
                    )

        self.back_noise = AddBackgroundNoise(
                        background_paths='/home/hasan/Desktop/kaggle/datasets/ff1010bird_wav/background',
                        min_snr_in_db=0,
                        max_snr_in_db=3,
                        mode='per_batch',
                        p = 0.3,
                        sample_rate=sample_rate,
                    )
    
        self.mel_tfm = MelSpec(nfft,hop_length,n_mels,sample_rate,stype,normalize)

        self.mel_fmask = SpecFrequencyMask(
            min_y=0.15,
            max_y=0.85,
            min_masked_mels=0.05,
            max_masked_mels=0.25,
            pad_value= 0.0 if normalize else -80.0,
            p=0.5
        )

        self.mel_tmask = SpecTimeMask(
            min_masked_steps=0.2,
            max_masked_steps=0.4,
            pad_value= 0.0 if normalize else -80.0,
            p=0.5
        )

    def forward(self,x):
        x = self.gain(x)
        x = self.noise(x)
        x = self.back_noise(x)
        x = torch.nan_to_num(x,nan=0.0)
        x = self.mel_tfm(x)
        x = torch.nan_to_num(x,nan=0.0)
        x = split(x,self.num_splits)
        x = self.mel_fmask(x)
        x = self.mel_tmask(x)
        x = merge(x,self.num_splits)
        return x

def get_tfm(level=0,**kwargs):
    if level ==0:
        return MelSpec(**kwargs)
    elif level ==1:
        return Transform_01(**kwargs)

    elif level == 2:
        return Transform_02(**kwargs)
    else:
        raise NotImplementedError



if __name__ == '__main__':
    #x = torch.ones((2,1,5,5))
    #addFreqCoords(x)
    #"""
    import librosa
    import numpy as np
    import soundfile
    import matplotlib.pyplot as plt
    #from audiomentations import 

    fig,axs = plt.subplots(2,1,figsize=(30,10))
    id = 'XC175511'
    x,_ = librosa.load(f'/home/hasan/Desktop/kaggle/datasets/birdclef-2022/train_audio/akepa1/{id}.ogg',sr=32000)
    x = x[:30*32000]
    x = x.reshape(1,1,-1)
    x = torch.from_numpy(x).float()

    gain = Gain(min_gain_in_db=-25, max_gain_in_db=-24, p=1.0,mode='per_example')

    noise = AddColoredNoise(
                    min_snr_in_db= -6,
                    max_snr_in_db= 3,
                    min_f_decay=1,
                    max_f_decay=2,
                    p=1.0,
                    sample_rate=32000,
                    mode='per_example'
                )
    x_aug = noise(gain(x))


    x_aug = x_aug[0,0,:].cpu().numpy()
    #print(x.shape)
    #
    #print(x)

    #x_back = librosa.feature.inverse.mel_to_audio(x_aug,sr=32000,n_fft=2048,hop_length=714,power=1)
    soundfile.write(f'/home/hasan/Desktop/kaggle/datasets/try_augs/{id}.wav',x_aug,samplerate=32000)
    #"""









