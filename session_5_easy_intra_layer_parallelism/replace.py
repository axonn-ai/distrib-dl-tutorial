import torch
import torch.nn as nn
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax
from axonn.intra_layer import Linear
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 



def replace_linear_layers_with_custom(model):
        for name, module in model.named_modules():
            if name == 'layers':  # Check if the module is 'layers' within the model
                for layer_idx, layer_module in enumerate(module):
                    for attr_name, attr_module in layer_module.named_children():
                        if isinstance(attr_module, nn.Linear):  # Check if the child module is nn.Linear
                            setattr(layer_module, attr_name, Linear(attr_module.in_features, attr_module.out_features))
        return model
