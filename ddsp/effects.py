
import torch
import torch.nn as nn
from ddsp.synth import SynthModule
    
class Effects(SynthModule):
    """
    Generic class for effects
    """
    
    def __init__(self):
        super(Effects, self).__init__()
        self.apply(self.init_parameters)
    
    def n_parameters(self):
        """ Return number of parameters in the module """
        return 0

    def forward(self, z):
        z, conditions = z
        return z
    