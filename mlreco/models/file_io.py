# Dummy model that only passes the input to the output, used for testing purposes.

import torch
import torch.nn as nn

class FileIOPlaceHolder(nn.Module):
    def __init__(self):
        super(FileIOPlaceHolder, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        out = {}
        return out
    
class FileIOPlaceHolderLoss(nn.Module):
    
    def __init__(self):
        super(FileIOPlaceHolderLoss, self).__init__()
        
    def forward(self, **kwargs):
        
        output = {
            'accuracy': 0.0,
            'loss': 0.0
        }
        
        return output
