import torch
import torch.nn as nn
from einops import einsum
import math

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
        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0,
                std=math.sqrt(2 / (in_features + out_features)),
                ),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ 
            learning: the self.weight matrix is of shape (d_out, d_in) and the input tensor x is of shape (..., d_in)
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:str=None, dtype:torch.dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                std=1,
                a=-3,  # min value
                b=3    # max value
            )
        )

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        """
        LEARNING:
        # This line uses advanced (fancy) indexing in PyTorch:
        # - self.weight is a tensor of shape (num_embeddings, embedding_dim)
        # - token_ids is a tensor of any shape (...), containing integer indices
        # When you do self.weight[token_ids], PyTorch returns a tensor where each index in token_ids
        # selects the corresponding row (embedding vector) from self.weight.
        # The output shape is (..., embedding_dim), matching the shape of token_ids plus the embedding dimension.
        # This is a very efficient way to look up embeddings for a batch of token ids.
        """
        return self.weight[token_ids]