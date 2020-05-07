
import torch
import torch.nn as nn

"""
#######################

High-level synthesizer modules classes

#######################
"""

class SynthModule(nn.Module):
    """
    Generic class defining a synthesis module.
    """
    
    def __init__(self, amortized='input'):
        """ Initialize as module """
        super(SynthModule, self).__init__()
        # Handle amortization
        self.amortized = amortized

    def init_parameters(self, m=None):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.001, 0.001)

    def __hash__(self):
        """ Dirty hack to ensure nn.Module compatibility """
        return nn.Module.__hash__(self)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        pass

    def n_parameters(self):
        """ Return number of parameters in the module """
        return 0

class Add(Synth):
    
    def __init__(self, modules=[]):
        super(Add, self).__init__(modules)

    def forward(self, z):
        z, conditions = z
        z_f = None
        for modules in self._modules.values():
            if (z_f is None):
                z_f = modules((z, conditions))
                continue
            z_f += modules((z, conditions))
        return z_f

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum([mod.n_parameters() for mod in self._modules.values()])

class Mul(Synth):
    
    def __init__(self, modules=[]):
        super(Mul, self).__init__(modules)

    def forward(self, z):
        z, _ = z
        z_f = torch.ones_as(z)
        for modules in self._modules.values():
            z_f *= modules(z)
        return z_f

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum([mod.n_parameters() for mod in self._modules.values()])
