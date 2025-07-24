import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):

    def __init__(self, 
                 units: int,
                 **kwargs,
                 ):
        nn.Module.__init__(self)
        
        self.lay_conv_1 = nn.Conv2d(units, units, kernel_size=3, stride=1, padding=1)
        self.lay_bn_1 = nn.BatchNorm2d(units)
        
        self.lay_conv_2 = nn.Conv2d(units, units, kernel_size=3, stride=1, padding=1)
        self.lay_bn_2 = nn.BatchNorm2d(units)
        
        self.lay_act = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x
        x = self.lay_conv_1(x)
        x = self.lay_bn_1(x)
        x = self.lay_act(x)
        
        x = self.lay_conv_2(x)
        x = self.lay_bn_2(x)
        
        x = x + residual
        x = self.lay_act(x)
        
        return x
    
    
class LogitContrastiveLoss(nn.Module):
    
    def __init__(self, eps: float = 1e-5):
        super(LogitContrastiveLoss, self).__init__()
        self.eps = eps
        
    def forward(self, sim: torch.Tensor,
                ) -> torch.Tensor:
        
        sim = sim.clamp(min=self.eps, max=1.0 - self.eps)
        
        logits = torch.log(sim / (1.0 - sim))
        
        N2 = sim.size(0)
        mask = torch.eye(N2, dtype=torch.bool, device=sim.device)
        logits = logits.masked_fill(mask, float('-inf'))
        
        N = N2 // 2
        positive_idx = torch.arange(N, 2*N, device=sim.device)
        positive_idx = torch.cat([positive_idx, torch.arange(0, N, device=sim.device)], dim=0)
        
        return F.cross_entropy(logits, positive_idx)