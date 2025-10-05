
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