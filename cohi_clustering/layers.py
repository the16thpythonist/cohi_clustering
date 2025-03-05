import torch
import torch.nn as nn


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
        
        self.lay_act = nn.SiLU()
        
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