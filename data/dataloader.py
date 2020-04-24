import os
import glob
import gzip
import pickle
import copy
import argparse
import torch
import numpy as np
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from neuralnet.analysis import MultiscaleFFT
import librosa

class AudioFeatures(object):
    def __init__(self, l_audio, r_audio, l_fft, r_fft):
        self.l_audio = l_audio
        self.r_audio = r_audio
        self.l_fft   = l_fft
        self.r_fft   = r_fft

class SoundSource(object):
    def __init__(self, azimuth, elevation, distance, pos):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.pos = pos

class SoundReceiver(object):
    def __init__(self, pos, orientation, hrtf, spot):
        self.pos = pos
        self.orientation = orientation
        self.hrtf = hrtf
        self.spot = spot

class RoomAcoustics(object):
    def __init__(self, size, freq_rt60, global_rt60, diffusion, absorption):
        self.size = size
        self.freq_rt60 = freq_rt60
        self.global_rt60 = global_rt60
        self.diffusion = diffusion
        self.absorption = absorption

"""
The architecture design refers to torch_ddsp by ircam: https://github.com/acids-ircam/ddsp_pytorch 
"""
class VASTDataset(Dataset):

    def __init__(self, datadir, transforms=None, fft_scales=[64, 3], split='train'):
        scales = []
        for s in range(fft_scales[1]):
            scales.append(fft_scales[0] * (2 ** s))
        self.multi_fft = MultiscaleFFT(scales)
        self.feature_files = []
        from_dir =  os.path.join(datadir, 'raw', split)
        to_dir =  os.path.join(datadir, 'data', split)
        if (not os.path.exists(to_dir)):
            os.makedirs(to_dir)
        if len(glob.glob(to_dir +'/*.pkl.gz')) == 0:
            self.preprocess_dataset(from_dir, to_dir)
        feat_files = sorted(glob.glob(to_dir + '/*.pkl.gz'))
        self.feature_files.extend(feat_files)
        

    def preprocess_dataset(self, from_dir, to_dir):
        cur_id = 0
        for root, dirs, files in os.walk(from_dir, topdown=False):
            for name in files:
                x = loadmat(os.path.join(root, name))
                l_ir     = x['RIR'][0,0][0].T
                r_ir     = x['RIR'][0,0][1].T
                room     = x['Room'][0,0]
                source   = x['Source'][0,0]
                receiver = x['Receiver'][0,0]
                l_fft    = self.multi_fft(torch.from_numpy(l_ir))
                r_fft    = self.multi_fft(torch.from_numpy(r_ir))

                # features = AudioBatch(l_audio=l_ir, r_audio=r_ir, l_fft=l_fft, r_fft=r_fft)

                for i in range(room[0].shape[1]):
                    features = AudioFeatures(l_audio=l_ir[i, :], r_audio=r_ir[i, :], l_fft=l_fft[i], r_fft=r_fft[i])
                    # features['source']    = SoundSource(source[0][0, i], source[1][0, i], source[2][0, i], source[3][:, i])
                    # features['receiver']  = SoundReceiver(receiver[0][:, i], receiver[1][0, i], receiver[2][0, 0], receiver[3][0, i])
                    # features['room']      = RoomAcoustics(
                    #     size =(room[0][0,i], room[0][1,i], room[0][2,i]),
                    #     freq_rt60 = room[1][:, i],
                    #     global_rt60 = room[2][:, i],
                    #     diffusion  = room[4][:, i],
                    #     absorption = (room[3][0,0][0][:, i], room[3][0,0][1][:, i], room[3][0,0][2][:, i], \
                    #                     room[3][0,0][3][:, i], room[3][0,0][4][:, i], room[3][0,0][5][:, i])
                    # )
                    to_path = to_dir + '/seq_' + str(cur_id) + '.pkl.gz'
                    print('save {} to {}'.format(i, to_path))
                    with gzip.open(to_path, 'wb') as f_out:
                        pickle.dump(features,f_out)
                    cur_id += 1
    
    def __getitem__(self, idx):
        loaded  = torch.load(self.features_files[idx]).item()
        l_audio = torch.from_numpy(loaded['l_audio']).unsqueeze(0)
        r_audio = torch.from_numpy(loaded['r_audio']).unsqueeze(0)
        l_fft   = torch.from_numpy(loaded['l_fft']).unsqueeze(0)
        r_fft   = torch.from_numpy(loaded['r_fft']).unsqueeze(0)
        # room    = loaded['room']
        # source  = loaded['source']
        # receiver= loaded['receiver']
        return l_audio, r_audio, l_fft, r_fft

    def __len__(self):
        return len(self.feature_files)

def load_dataset(vastdir='./dataset/VAST/', batch_size=16, num_workers=4, **kwargs):
    val_set = None
    test_set = None
    train_set = VASTDataset(vastdir, split='train')
    val_set   = VASTDataset(vastdir, split='val')
    test_set  = VASTDataset(vastdir, split='test')

    print(len(train_set.feature_files))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, **kwargs)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=(train_type == 'random'), num_workers=nbworkers, pin_memory=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=(train_type == 'random'), num_workers=nbworkers, pin_memory=False, **kwargs)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = load_dataset('./dataset/VAST')
    print(train_loader)
    print(valid_loader)
    print(test_loader)