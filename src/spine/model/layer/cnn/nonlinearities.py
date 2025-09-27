"""Custom non-linear activation functions."""

import MinkowskiEngine as ME
import torch


class MinkowskiMish(torch.nn.Module):
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
        input_data : ME.SparseTensor
            Sparse input tensor

        Return
        ------
        ME.SparseTensor
            Sparse output tensor
        """
        out = torch.nn.functional.softplus(input_data.F)
        out = torch.tanh(out)
        out = out * input_data.F

        return ME.SparseTensor(
            out, coords_key=input_data.coords_key, coords_manager=input_data.coords_man
        )
