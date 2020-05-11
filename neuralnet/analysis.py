# -*- coding: utf-8 -*-
"""
    MultiscaleFFT code is from DDSP_torch project: https://github.com/acids-ircam/ddsp_pytorch  
"""
import torch
import torch.nn as nn
from pyworld import dio
import numpy as np

# Lambda for computing squared amplitude
amp = lambda x: x[...,0]**2 + x[...,1]**2

class Analysis(nn.Module):
    """
    Generic class for trainable analysis modules.
    """
    
    def __init__(self):
        super(Analysis, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass

class MultiscaleFFT(Analysis):
    """
    Compute the FFT of a signal at multiple scales
    
    Arguments:
            block_size (int)    : size of a block of conditioning
            sequence_size (int) : size of the conditioning sequence
    Return value:
            stfts : (scales, freq_bin, time_step)
    """
    
    def __init__(self, fft_scales, overlap=0.75, reshape=True):
        super(MultiscaleFFT, self).__init__()
        self.apply(self.init_parameters)
        scales = []
        for s in range(fft_scales[1]):
            scales.append(fft_scales[0] * (2 ** s))
        self.scales = scales
        self.overlap = overlap
        self.reshape = reshape
        self.windows = nn.ParameterList(
                nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False)\
            for scale in self.scales)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        stfts = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False)
            stfts.append(amp(cur_fft))
        
        if (self.reshape):
            stft_tab = []
            for b in range(x.shape[0]):
                cur_fft = []
                for s, _ in enumerate(self.scales):
                    cur_fft.append(stfts[s][b])
                stft_tab.append(cur_fft)
            stfts = stft_tab
        return stfts
