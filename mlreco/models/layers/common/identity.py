import torch.nn as nn


class Identity(nn.Module):
    """No-op torch module."""
    def __init__(self):
        """Initialize the module."""
        # Initialize the parent class
        super().__init__()

    def forward(self, input_data):
        """Returns the input as is.
        
        Parameters
        ----------
        input_data : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
             Input sparse tensor
        """
        return input_data

