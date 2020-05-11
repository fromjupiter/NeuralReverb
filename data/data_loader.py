import os
import glob
import math
import torch
import pickle
import librosa
import argparse
import torchaudio
import numpy as np
from itertools import chain
from scipy.io import loadmat
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import IterableDataset, Dataset, DataLoader

from neuralnet.analysis import MultiscaleFFT

class AudioFeatureBlock(object):
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


def vast_worker_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    room_count = len(dataset.room_files)
    worker_id = worker_info.id
    per_worker = int(math.ceil(room_count / float(worker_info.num_workers)))
    dataset.room_files = dataset.room_files[worker_id * per_worker : worker_id * per_worker + per_worker]


class VASTDataset(IterableDataset):
    def __init__(self, datadir, anechoic_path, transforms=None, fft_scales=[128, 3], n_mfcc=33):
        # TODO: consider move pre-process here to save training time (Currently disk space is not enough)
        self.multi_fft = MultiscaleFFT(fft_scales)
        self.room_files = [] # configured by worker init function
        self.room_files.extend(sorted(glob.glob(os.path.join(datadir, '*.mat'))))
        self._analyze()
        # TODO: Ground truth is not the same as anechoic sound
        self.anechoic = self._read_room(anechoic_path)[0].unsqueeze(0)
        # self.mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc = n_mfcc)

    def _analyze(self):
        self.max_ir_len = 0
        for room in self.room_files:
            x = loadmat(room)
            self.sr = x['GlobalParams'][0,0][0][0,0]
            self.freq_bin = x['GlobalParams'][0,0][1][0]
            self.max_ir_len = max(self.max_ir_len, x['RIR'][0,0][1].shape[0])

    def __iter__(self):
        return chain.from_iterable(map(self._map_room, self.room_files))

    def _read_room(self, room_path):
        x = loadmat(room_path)
        l_ir     = torch.from_numpy(x['RIR'][0,0][0].T)
        r_ir     = torch.from_numpy(x['RIR'][0,0][1].T)
        l_ir     = F.pad(l_ir, (0,self.max_ir_len - l_ir.size()[1]), mode='constant', value=0)
        r_ir     = F.pad(r_ir, (0,self.max_ir_len - r_ir.size()[1]), mode='constant', value=0)
        room     = x['Room'][0,0]
        source   = x['Source'][0,0]
        receiver = x['Receiver'][0,0]
        # l_mfcc   = self.mfcc(l_ir)
        # r_mfcc   = self.mfcc(r_ir)
        # l_fft    = self.multi_fft(l_ir)
        # r_fft    = self.multi_fft(r_ir)
        # return iter(zip(l_ir, r_ir, l_fft, r_fft))
        y = torch.cat((l_ir.unsqueeze(1), r_ir.unsqueeze(1)), dim=1)
        return y

    def _map_room(self, room_path):
        y = self._read_room(room_path)
        return iter(zip(y, self.anechoic.repeat(y.size(0), 1, 1)))

def get_loader(name='VAST', datadir='./dataset/VAST/train', anechoic_path='./dataset/VAST/anechoic.mat', batch_size=16, num_workers=1, **kwargs):
    if name=='VAST':
        vast_set = VASTDataset(datadir, anechoic_path)
    else:
        print('Other dataset not available yet!')
        exit(-1)
    data_loader = DataLoader(vast_set, batch_size=batch_size, num_workers=num_workers, pin_memory=False, worker_init_fn=vast_worker_fn, **kwargs)
    return data_loader


if __name__ == '__main__':
    train_loader = get_loader(datadir='./dataset/VAST/train', batch_size=16)
    valid_loader = get_loader(datadir='./dataset/VAST/val', batch_size=16)
    test_loader  = get_loader(datadir='./dataset/VAST/test', batch_size=16)
    train_cnt = 0
    val_cnt = 0
    test_cnt = 0
    for tup in train_loader:
        train_cnt += len(tup[0])
    print("train count: {}".format(train_cnt))
    for tup in valid_loader:
        val_cnt += len(tup[0])
    print("val count: {}".format(val_cnt))
    for tup in test_loader:
        test_cnt += len(tup[0])
    print("test count: {}".format(test_cnt))