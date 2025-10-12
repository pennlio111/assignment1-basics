
import token
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) # upcast to float32 for numerical stability
        norm_factor = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * norm_factor).to(x.dtype) # downcast to original dtype

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        ### LEARNING:
        # the weight shape must be declared in the constructor, by a random initialization
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(x @ w1) * (x @ w3) @ w2
        swish_gate = torch.sigmoid(x @ self.w1.T) * (x @ self.w1.T)  # Swish activation
        gated = swish_gate * (x @ self.w3.T)  # Element-wise multiplication with gate
        ### LEARNING:
        # the input x is of shape (..., d_model)!!! a row vector
        # So the output must be of the same shape (..., d_model)!!! a row vector
        return gated @ self.w2.T

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: str = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
    
    def _get_rotation_values(self, token_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute theta values for each dimension pair
        # k ranges from 1 to d_k//2
        # LEARNING:
        # 1. torch.arange is used to create a tensor of integers from 1 to d_k//2
        
        k = torch.arange(1, self.d_k // 2 + 1, device=token_positions.device, dtype=torch.float32)
        # Compute the frequency for each dimension pair
        # LEARNING:
        # 1. 2 * (k - 1) is used to compute the frequency for each dimension pair
        # Compute angles: position * frequency
        # Result shape: (..., seq_len, d_k//2)
        # LEARNING:
        # 1. token_positions.unsqueeze(-1) is used to add a new dimension to the token_positions tensor
        # 2. freqs is a tensor of shape (d_k//2,)
        freqs = 1.0 / (self.theta ** (2 * (k - 1) / self.d_k))
        # 3. angles is a tensor of shape (..., seq_len, d_k//2)
        angles = token_positions.unsqueeze(-1) * freqs
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        return cos_vals, sin_vals

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_vals, sin_vals = self._get_rotation_values(token_positions)
        # output the same shape as x
        output = torch.empty_like(x)
        # Split x into even and odd indices (pairs of dimensions)
        # LEARNING:
        # 1. split the tensor into even and odd indices: 0, 2, 4, ... and 1, 3, 5, ...
        x1 = x[..., 0::2]  # Even indices: 0, 2, 4, ...
        x2 = x[..., 1::2]  # Odd indices: 1, 3, 5, ...
        
        # Apply rotation to each pair
        # RoPE rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rotated = x1 * cos_vals - x2 * sin_vals
        x2_rotated = x1 * sin_vals + x2 * cos_vals
        
        # Interleave the rotated values back
        output[..., 0::2] = x1_rotated # even indices
        output[..., 1::2] = x2_rotated # odd indices
        
        return output