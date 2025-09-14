import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Initialize a linear layer with weight and bias parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            device (str, optional): The device to store the parameters.
            dtype (torch.dtype, optional): The data type of the parameters.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device=device, dtype=dtype))

    def forward(self, x):
        """ 
            learning: the self.weight matrix is of shape (d_out, d_in) and the input tensor x is of shape (..., d_in)
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")