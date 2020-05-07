import logging
import itertools
import torch
import torch.nn as nn
from abc import ABC,abstractmethod

from neuralnet.analysis import MultiscaleFFT

class BaseModel(nn.Module):
    """
    Base model for neural reverberator.
    Input is binaural audio signal and its anechoic recording,
    Output is the synthesized reverbed audio.
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        logging.info('Initializing model {}'.format(self.__class__.__name__))
        self.multiScaleFFT = MultiscaleFFT(scales=config['fft_scales'], overlap=config['overlap'])
        self.scale = config['fft_scales'][0]

    def set_input(self, y, y_orig):
        self.y = y
        self.y_orig = y_orig

    """
    Encode audio to latent variable

    Arguments:
        y: binaural audio, batch_size x 2 x audio_samples
    """
    @abstractmethod
    def encode(self, y):
        pass

    """
    decode latent variable to reverberator's parameters

    Arguments:
        z: latent_variable, batch_size x 2 x hidden_size
    """
    @abstractmethod
    def decode(self, z):
        pass

    """
    Pass learned parameters to a reveberator and synthesize new sound
    
    Arguments:
        y_orig: anechoic binaural audio, batch_size x 2 x audio_samples
        params: reverberator's parameters
    """
    @abstractmethod
    def synth(self, y_orig, params):
        pass
    
    """
    Arguments:
        y: binaural audio, batch_size x 2 x audio_samples
        y_orig: anechoic binaural audio, batch_size x 2 x audio_samples
    """
    def forward(self):
        # y_stfts : list of (batch_size x 2 x seq_len x freq_bins), length is scales
        self.y_stfts = self.multiScaleFFT(self.y)
        # z : batch_size x 2 x seq_len x hidden_size
        self.z = self.encode(self.y_stfts[0])
        self.params = self.decode(z)
        self.y_synth = synth(self.y_orig, self.params)
    
    @abstractmethod
    def optimize_parameters(self):
        # self.optimizer.zero_grad()
        # self.loss.backward()
        # self.optimizer.step()
        pass

class SimpleReverberator(BaseModel):
    def __init__(self, config):
        super(SimpleReverberator, self).__init__(config)
        self.hidden_size = config['hidden_size']
        lp = [self.parameters()]
        self.optimizer = torch.optim.Adam(itertools.chain(*lp), lr=config['learning_rate'])
        self.encoder = nn.GRU(int(self.scale/2)+1, self.hidden_size, 2, batch_first=True, bidirectional=False)
        self.decoder = nn.Linear(self.hidden_size, 2)

    def encode(self, y):
        return self.encoder(y)

    def decode(self, z):
        return self.decoder(z)

    """
    A simple exp decay reverberator
    """
    def synth(self, y_orig, params):
        wetdry = params[:,0]
        decay = params[:,1]
        # Pad the input sequence
        y_orig = nn.functional.pad(y_orig, (0, self.size), "constant", 0)
        # Compute STFT
        Y_S = torch.rfft(y, 1)
        # Compute the current impulse response
        idx = torch.sigmoid(wetdry) * identity
        imp = torch.sigmoid(1 - wetdry) * y_orig
        dcy = torch.exp(-(torch.exp(decay) + 2) * torch.linspace(0,1, self.size).to(y_orig.device))
        final_impulse = idx + imp * dcy
        # Pad the impulse response
        impulse = nn.functional.pad(final_impulse, (0, self.size), "constant", 0)
        if y.shape[-1] > self.size:
            impulse = nn.functional.pad(impulse, (0, y.shape[-1] - impulse.shape[-1]), "constant", 0)
        IR_S = torch.rfft(impulse.detach(),1).expand_as(Y_S)
        # Apply the reverb
        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]
        # Invert the reverberated signal
        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))
        return y

    def optimize_parameters(self):
        self.forward()
        stfts = self.y_stfts
        stfts_rec = self.multiScaleFFT(self.y_synth)
        
        lin_loss = sum([torch.mean(abs(stfts[i] - stfts_rec[i])) for i in range(len(stfts_rec))])
        log_loss = sum([torch.mean(abs(torch.log(stfts[i]+1e-4) - torch.log(stfts_rec[i] + 1e-4))) for i in range(len(stfts_rec))])
        self.loss = lin_loss + log_loss
        
        self.loss.backward()
        self.optimizer.step()
