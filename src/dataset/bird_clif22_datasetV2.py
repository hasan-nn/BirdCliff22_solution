import random
from cv2 import repeat
import torch
from torch.utils.data import Dataset


import cv2
import os
import json
import librosa
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from audiomentations import AddBackgroundNoise,AddGaussianSNR,AddShortNoises,Gain,Compose,OneOf
from transforms import AddPinkSNR
from librosa.effects import time_stretch



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#Bird Counts
bird_counts = {
    "afrsil1": 16, "houspa": 642, "redava": 37, "zebdov": 122, "akekee": 6, "apapan": 127, "warwhe1": 122, "iiwi": 91,
    "akepa1": 13, "akiapo": 17, "hawama": 65, "hawcre": 23, "omao": 41, "elepai": 18, "akikik": 3, "amewig": 65, "sora": 158,
    "mallar3": 613, "houfin": 446, "cangoo": 392, "eurwig": 362, "norsho": 117, "bknsti": 219, "aniani": 15, "kauama": 15,
    "saffin": 154, "spodov": 122, "oahama": 19, "jabwar": 88, "redjun": 100, "arcter": 218, "skylar": 553, "comsan": 505,
    "barpet": 15, "bcnher": 467, "categr": 124, "comgal1": 205, "osprey": 237, "belkin1": 113, "leasan": 98, "grbher3": 108,
    "norcar": 575, "moudov": 279, "pibgre": 173, "bkbplo": 285, "rudtur": 202, "dunlin": 503, "brant": 139, "semplo": 81, "laugul": 93,
    "gnwtea": 504, "lesyel": 167, "wesmea": 411, "hawgoo": 11, "lobdow": 94, "leater1": 104, "sposan": 126, "commyn": 240, "hawcoo": 19,
    "bkwpet": 4, "sooshe": 13, "blkfra": 42, "blknod": 16, "refboo": 44, "bongul": 61, "buwtea": 55, "ribgul": 94, "gwfgoo": 277, "glwgul": 39,
    "brnboo": 13, "brnnod": 21, "sooter1": 26, "brnowl": 503, "rocpig": 165, "brtcur": 26, "rinphe": 269, "whiter": 31, "rettro": 34, "bubsan": 5,
    "buffle": 15, "bulpet": 7, "burpar": 20, "cacgoo1": 46, "snogoo": 103, "calqua": 147, "normoc": 455, "gadwal": 252, "caster1": 181, "canvas": 10,
    "comwax": 199, "chbsan": 16, "chemun": 12, "chukar": 41, "cintea": 17, "rorpar": 327, "nutman": 81, "compea": 57, "coopet": 3, "crehon": 2,
    "mauala": 4, "sander": 126, "ercfra": 6, "norpin": 121, "fragul": 36, "wiltur": 85, "gamqua": 187, "hoomer": 22, "golphe": 11, "grefri": 17,
    "gresca": 16, "gryfra": 48, "hawhaw": 3, "hawpet1": 3, "hudgod": 8, "incter1": 8, "japqua": 43, "kalphe": 17, "layalb": 3, "lcspet": 13,
    "wessan": 42, "lessca": 8, "whfibi": 46, "redpha1": 41, "lotjae": 44, "madpet": 19, "magpet1": 16, "masboo": 27, "maupar": 1, "merlin": 74,
    "mitpar": 59, "norhar2": 31, "pagplo": 53, "palila": 7, "parjae": 62, "pomjae": 9, "pecsan": 79, "peflov": 38, "perfal": 186, "wetshe": 24,
    "puaioh": 3, "reccar": 44, "rempar": 39, "rinduc": 30, "ruff": 40, "sheowl": 128, "shtsan": 3, "sopsku1": 6, "towsol": 101, "wantat1": 19,
    "whttro": 23, "yebcar": 20, "yefcan": 69}

max_bird_count = np.array(list(zip(*bird_counts.items()))[-1]).max() # 642

#152 classes                      
class_names = [
    'afrsil1', 'akekee', 'akepa1', 'akiapo', 'akikik', 'amewig', 'aniani', 'apapan', 'arcter', 'barpet', 'bcnher', 'belkin1',
    'bkbplo', 'bknsti', 'bkwpet', 'blkfra', 'blknod', 'bongul', 'brant', 'brnboo', 'brnnod', 'brnowl', 'brtcur', 'bubsan',
    'buffle', 'bulpet', 'burpar', 'buwtea', 'cacgoo1', 'calqua', 'cangoo', 'canvas', 'caster1', 'categr', 'chbsan', 'chemun',
    'chukar', 'cintea', 'comgal1', 'commyn', 'compea', 'comsan', 'comwax', 'coopet', 'crehon', 'dunlin', 'elepai', 'ercfra',
    'eurwig', 'fragul', 'gadwal', 'gamqua', 'glwgul', 'gnwtea', 'golphe', 'grbher3', 'grefri', 'gresca', 'gryfra', 'gwfgoo', 
    'hawama', 'hawcoo', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'hoomer', 'houfin', 'houspa', 'hudgod', 'iiwi', 'incter1', 
    'jabwar', 'japqua', 'kalphe', 'kauama', 'laugul', 'layalb', 'lcspet', 'leasan', 'leater1', 'lessca', 'lesyel', 'lobdow', 
    'lotjae', 'madpet', 'magpet1', 'mallar3', 'masboo', 'mauala', 'maupar', 'merlin', 'mitpar', 'moudov', 'norcar', 'norhar2', 
    'normoc', 'norpin', 'norsho', 'nutman', 'oahama', 'omao', 'osprey', 'pagplo', 'palila', 'parjae', 'pecsan', 'peflov', 'perfal', 
    'pibgre', 'pomjae', 'puaioh', 'reccar', 'redava', 'redjun', 'redpha1', 'refboo', 'rempar', 'rettro', 'ribgul', 'rinduc', 'rinphe', 
    'rocpig', 'rorpar', 'rudtur', 'ruff', 'saffin', 'sander', 'semplo', 'sheowl', 'shtsan', 'skylar', 'snogoo', 'sooshe', 'sooter1', 
    'sopsku1', 'sora', 'spodov', 'sposan', 'towsol', 'wantat1', 'warwhe1', 'wesmea', 'wessan', 'wetshe', 'whfibi', 'whiter', 'whttro', 
    'wiltur', 'yebcar', 'yefcan', 'zebdov']

scored_classese = [
    'akiapo', 'aniani', 'apapan', 'barpet', 'crehon', 'elepai', 'ercfra',
    'hawama', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'houfin', 'iiwi',
    'jabwar', 'maupar', 'omao', 'puaioh', 'skylar', 'warwhe1', 'yefcan']

#repeats = [5,4,1,]
num_classes = len(class_names)

#class count weights
bins = np.arange(start = 15,stop=105,step=5) / 100

cc_weights = []
for bird_name in class_names:
    count = bird_counts[bird_name]
    ccw = find_nearest(bins,1.0 - count / max_bird_count) * 10
    cc_weights.append(ccw)
cc_weights = np.array(cc_weights) 
#print(cc_weights)
def load_json(path):
    return json.load(open(path,'r'))

def pad_audio(x,full_length):
    audio_length = x.shape[0]
    if audio_length >= full_length:
        return x
    padded_x = np.zeros(full_length,dtype=x.dtype)
    padded_x[0:audio_length] = x
    return padded_x

def crop_audio(x,crop_length):
    audio_length = x.shape[0]
    if audio_length == crop_length:
        return x, (0,audio_length)

    elif audio_length < crop_length:
        
        x = pad_audio(x,crop_length)
        se = (0,audio_length)
    else:
        
        start = np.random.randint(audio_length + 1 - crop_length)
        x = x[start:start+crop_length]
        se = (start,start+crop_length)
    return x,se

def stretch_time_steps(x,steps,crop_size,min_rate=0.8,max_rate=1.25,p=0.5):
    for i in range(steps):
        if random.random() > p:
            continue
        r = max_rate - min_rate
        
        rate = r * np.random.random_sample() + min_rate
        s,e = i*crop_size,i*crop_size+crop_size
        x[s:e],_ = crop_audio(time_stretch(x[s:e],rate=rate),crop_size)
    return x


def internal_mixup(x,x_orig,se,alpha=0.5):
    x_rest = np.zeros(x_orig.shape[0] - x.shape[0],dtype=x.dtype)

    x_rest[:se[0]] = x_orig[:se[0]]
    x_rest[se[0]:] = x_orig[se[1]:]

    if x_rest.shape[0] < x.shape[0]:
        b = np.random.randint(x.shape[0] - x_rest.shape[0])
        x[b:b+x_rest.shape[0]] = x[b:b+x_rest.shape[0]] * alpha + (1.0 - alpha) * x_rest
        return x

    x_rest,_ = crop_audio(x_rest,x.shape[0])
    x = x * alpha + (1.0 - alpha) * x_rest
    return x 

class BirdClif22Dataset(Dataset):
    def __init__(self,
                 path='../../../../datasets/birdclef-2022',
                 split='train',#train,val,test
                 fold = 0,
                 crop_length = 30,#(seconds)
                 sampling_rate = 32000,
                 n_fft = 2048,
                 hop_length = 1250,
                 totensor=False,
                 pr_weight = 1.0,
                 sec_weight = 1.0,
                 neg_weight = 1.0,
                 rating_min_weight = 0.0,
                 use_ccw = False,
                 filter = 'mel' #['mel' 'pcen']
                 ): 


        super().__init__()
        assert split in ['train','val','test']

        self.train = split == 'train'
        if self.train:
            self.get_train_tfm()

        self.dataset_path = path
        self.train_audio_path = f'{path}/train_audio'
        self.gen_data_path = f'{path}/generated_data'

        self.bird_counts = load_json(f'{self.gen_data_path}/bird_counts.json')
        self.audio_durations = load_json(f'{self.gen_data_path}/durations.json')
        self.classes = {c : i for i,c in enumerate(class_names)}

        self.split = split
        self.fold = fold
        self.crop_length_sec = crop_length 
        self.crop_length = crop_length * sampling_rate

        self.tt = totensor
        self.load_data(path)

        self.pr_weight = pr_weight
        self.sec_weight = sec_weight
        self.neg_weight = neg_weight
        self.sr = sampling_rate
        self.rmw = rating_min_weight
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.filter = filter

        self.use_ccw = use_ccw
        self.min_crop = 5.0 * self.sr
        

    

    def load_data(self,path):
        meta = []
        if self.split == 'test':
            df = pd.read_csv(f'{self.gen_data_path}/splits/local_test.csv')
        else:
            df = pd.read_csv(f'{self.gen_data_path}/splits/cv_folds.csv')
            if self.split == 'train':
                df = df.loc[(df['fold'] != self.fold)]
            else:
                df = df.loc[(df['fold'] == self.fold)]

        for i,row in df.iterrows():
            primary_label = row['primary_label']
            secondary_labels = [l.lstrip('\'').rstrip('\'') for l in row['secondary_labels'].lstrip('[').rstrip(']').split(', ') if l!= '']
            filename = row['filename']
            rating = row['rating']
            duration = row['duration(sec)']

            labels = set([primary_label,*secondary_labels])
            sc_intr = labels.intersection(set(scored_classese))
            
            #if duration > self.crop_length_sec and len(sc_intr) > 0:
            #    for l in sc_intr:
            #        if l not in a.keys():
            #            a[l] = []
            #        a[l].append(duration)

            
            if self.train and (len(sc_intr) > 0):
                ov = False
                ov2 = False
                ov3 = False
                for l in sc_intr:
                    if bird_counts[l] < 70:
                        ov = True
                    if bird_counts[l] < 30:
                        ov2 = True
                    if bird_counts[l] < 5:
                        ov3 = True

                if ov and (duration > self.crop_length_sec):
                    repeats = math.floor((duration - self.crop_length_sec)/ 5.0 + 1)
                elif ov3:
                    repeats = 1
                elif ov2:
                    repeats = 1
                else:
                    repeats = 1

                meta.extend([(filename,primary_label,secondary_labels,rating)]*repeats)
            else:
                meta.append((filename,primary_label,secondary_labels,rating))

        self.meta = meta

    def get_train_tfm(self):
        bg_path = '/home/hasan/Desktop/kaggle/datasets/ff1010bird_wav/background'
        self.train_tfm = Compose(
            transforms = [
                OneOf(
                    transforms=[
                        Gain(min_gain_in_db=-6, max_gain_in_db=6.0, p=1.0)],p=0.5),
                OneOf(
                    transforms=[
                        AddGaussianSNR(min_snr_in_db=-9, max_snr_in_db=5, p=1.0),
                        AddPinkSNR(min_snr_in_db=-3, max_snr_in_db=5, p=1.0),

                    ],p=0.6)
            ],p=1.0)

    
    def __getitem__(self, index):
        filename,primary_label,secondary_labels,rating = self.meta[index]

        x_orig, _ = librosa.load(f'{self.train_audio_path}/{filename}',sr=self.sr)
        
        start_pad = int(math.ceil(x_orig.shape[0] / self.min_crop))

        if self.train:
            x,se = crop_audio(x_orig,self.crop_length)
            num_splits = int(x.shape[0] // self.min_crop)
            #if start_pad > 0 and random.random() <= 0.5:
            #    x = stretch_time_steps(
            #        x,
            #        steps=min(start_pad,num_splits),
            #        crop_size=int(self.min_crop),
            #        min_rate=0.8,
            #        max_rate=1.25,
            #        p=0.5)

            if x_orig.shape[0] > self.crop_length:
                if random.random() > 0.5:
                    x = internal_mixup(x,x_orig,se,alpha= 0.5 * np.random.random_sample() + 0.5)

            #x = self.train_tfm(x,sample_rate=self.sr)

            num_splits = int(x.shape[0] // self.min_crop)
            cow_weights = np.ones(num_splits,dtype=np.float64) 
            if start_pad <= (num_splits - 1):
                cow_weights[start_pad:] = 0.0

            x = x[np.newaxis,...]

        else:

            bsize = max(1,int(math.ceil(x_orig.shape[0] / self.crop_length)))
            _crop_length = self.crop_length * bsize
            x,se = crop_audio(x_orig,_crop_length)

            num_splits = int(x.shape[0] // self.min_crop)
            cow_weights = np.ones(num_splits,dtype=np.float64) 
            if start_pad <= (num_splits - 1):
                cow_weights[start_pad:] = 0.0

            x = x.reshape(bsize,-1)
            cow_weights = cow_weights.reshape(bsize,-1)


        
        y = np.zeros(num_classes,dtype=np.float64)
        weights = np.ones(num_classes,dtype=np.float64) * self.neg_weight

        pr_idx = self.classes[primary_label]
        y[pr_idx] = 1.0
        weights[pr_idx] = self.pr_weight

        sec_idxs = [self.classes[sec_label] for sec_label in secondary_labels]
        y[sec_idxs] = 1.0
        weights[sec_idxs] = self.sec_weight

        if self.use_ccw:
            weights *= cc_weights

        rating_weight = np.array((rating / 5.0) + self.rmw)


        data = {
            'x' : x,
            'y_gt' : y,
            'weights' : weights,
            'rating_weight' : rating_weight,
            'cow_weights' : cow_weights,
            'seconds' : x_orig.shape[0] / self.min_crop
        }

        return data

    def to_tensor(self, arr):
        return torch.from_numpy(arr.transpose((2, 0, 1))).float()

    def ToTensor(self, img, mask):
        return self.to_tensor(img), self.to_tensor(mask)
  
    def __len__(self):
        return len(self.meta)




if __name__ == '__main__':
    #"""
    dataset = BirdClif22Dataset(

                 path='../../../../datasets/birdclef-2022',
                 split='val',#train,val,test
                 fold = 0,
                 crop_length = 30,#(seconds)
                 sampling_rate = 32000,
                 n_fft = 2048,
                 hop_length = 716,
                 totensor=False,
                 pr_weight = 1.0,
                 sec_weight = 1.0,
                 neg_weight = 1.0,
                 rating_min_weight = 0.0,
                 filter = 'mel' #['mel' 'pcen']
                 )
    dataset.__getitem__(1501)
    #"""