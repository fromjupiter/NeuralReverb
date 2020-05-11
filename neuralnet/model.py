import logging
import itertools
import torch
import torch.nn as nn
from abc import ABC,abstractmethod

from neuralnet.analysis import MultiscaleFFT
from neuralnet.loss import MSSTFTLoss

class BaseModel(nn.Module):
    """
    Base model for neural reverberator.
    Input is binaural audio signal and its anechoic recording,
    Output is the synthesized reverbed audio.
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        logging.info('Initializing model {}'.format(self.__class__.__name__))
        self.multiScaleFFT = MultiscaleFFT(fft_scales=config['fft_scales'], overlap=config['overlap'], reshape=False)
        self.scale = config['fft_scales'][0]

    def set_input(self, y, y_orig):
        self.y = y
        self.y_orig = y_orig

    """
    Encode audio to latent variable

    Arguments:
        fft: spectrogram of the binaural audio, batch_size x 2 x freq_bins x seq_len
    """
    @abstractmethod
    def encode(self, fft):
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
    Return:
        synth_y: reverb audio, batch_size x 2 x audio_samples
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
        # y_stfts : list of (batch_size x 2, freq_bins, seq_len), list length is fft_scales
        self.y_stfts = self.multiScaleFFT(self.y.view(-1, self.y.size(2)))
        # z : batch_size x 2 x seq_len x hidden_size
        self.z = self.encode(self.y_stfts[0].view(self.y.size(0), 2, self.y_stfts[0].size(1), self.y_stfts[0].size(2)))
        self.params = self.decode(self.z)
        self.y_synth = self.synth(self.y_orig, self.params)
        return self.y_synth
    
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
        self.encoder = nn.GRU(int(self.scale/2+1)*2, self.hidden_size, 1, batch_first=True, bidirectional=False)
        self.decoder = nn.Linear(self.hidden_size, 4)
        self.criterion = MSSTFTLoss(config)
        self.optimizer = torch.optim.Adam(itertools.chain(*[self.encoder.parameters(), self.decoder.parameters()]), lr=config['learning_rate'])

    def set_input(self, y, y_orig):
        super().set_input(y, y_orig)
        # parameters for exp decay reverberator
        self.size     = y.size(2)
        self.impulse  = nn.Parameter(torch.rand(1, self.size) * 2 - 1, requires_grad=False).to(y.device)
        self.identity = nn.Parameter(torch.zeros(1, self.size), requires_grad=False).to(y.device)
        self.identity[:,0] = 1

    def encode(self, fft):
        # TODO: use pack_padded_sequence

        # change fft to: batch_size x seq_len x  (2 x freq_bins)
        fft = fft.view(fft.size(0), -1, fft.size(3)).permute(0, 2, 1)
        return self.encoder(fft)[1]

    def decode(self, z):
        z = z.permute(1,0,2)
        z = z.view(z.size(0), -1)
        return self.decoder(z)

    def synth(self, y_orig, params):
        l_wetdry = params[:,0]
        l_decay = params[:,1]
        r_wetdry = params[:,2]
        r_decay = params[:,3]
        l_y = self._exp_decay_reverb(self.y_orig[:,0,:].squeeze(1), l_wetdry, l_decay)
        r_y = self._exp_decay_reverb(self.y_orig[:,1,:].squeeze(1), r_wetdry, r_decay)
        return torch.cat((l_y.unsqueeze(1), r_y.unsqueeze(1)), dim=1)
    
    """
    A simple exp decay reverberator
    Arguments:
        y: binaural audio, batch_size x 2 x audio_samples
        wetdry: (batch_size,)
        decay:  (batch_size,)
    """
    def _exp_decay_reverb(self, y, wetdry, decay):
        # Pad the input sequence
        # y = nn.functional.pad(y, (0, self.size), "constant", 0)
        # Compute STFT
        Y_S = torch.rfft(y, 1)
        # Compute the current impulse response
        idx = torch.sigmoid(wetdry).unsqueeze(1) * self.identity.expand(len(wetdry), -1)
        imp = torch.sigmoid(1 - wetdry).unsqueeze(1) * self.impulse.expand(len(wetdry), -1)
        dcy = torch.exp(-(torch.exp(decay) + 2).unsqueeze(1) * torch.linspace(0,1, self.size).to(y.device))
        final_impulse = idx + imp * dcy
        # Pad the impulse response
        impulse = final_impulse
        # impulse = nn.functional.pad(final_impulse, (0, self.size), "constant", 0)
        # if y.shape[-1] > self.size:
        #     impulse = nn.functional.pad(impulse, (0, y.shape[-1] - impulse.shape[-1]), "constant", 0)
        IR_S = torch.rfft(impulse,1).expand_as(Y_S)
        # IR_S = torch.rfft(impulse.detach(),1).expand_as(Y_S)
        # Apply the reverb
        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]
        # Invert the reverberated signal
        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))
        return y

    def optimize_parameters(self):
        self.forward()
        self.loss = self.criterion(self.y_synth.view(-1, self.y_synth.size(2)), self.y_stfts)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
