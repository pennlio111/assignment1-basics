
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