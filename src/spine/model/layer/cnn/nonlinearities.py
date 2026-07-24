"""Custom non-linear activation functions."""

import torch


class Mish(torch.nn.Module):
    """Mish non-linearity layer.

    Reference: https://arxiv.org/pdf/1908.08681.pdf
    """

    def __init__(self):
        """Initialize the layer."""
        super().__init__()

    def forward(self, input_data):
        """Pass tensor through the layer.

        Parameters
        ----------
        input_data : sparse.SparseTensor
            Sparse input tensor

        Return
        ------
        sparse.SparseTensor
            Sparse output tensor
        """
        out = torch.nn.functional.softplus(input_data.F)
        out = torch.tanh(out)
        out = out * input_data.F

        return input_data.replace_features(out)
