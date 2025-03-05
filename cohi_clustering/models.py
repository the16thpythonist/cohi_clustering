import os
from typing import List, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from cohi_clustering.data import ImageData
from cohi_clustering.layers import ResNetLayer


class ContrastiveCNN(pl.LightningModule):
    
    def __init__(self, 
                 input_shape: List[int],
                 resnet_units: int = 64,
                 embedding_units: List[int] = [128, 256],
                 projection_units: List[int] = [256, 1024],
                 contrastive_factor: float = 1.0,
                 **kwargs,
                 ):
        pl.LightningModule.__init__(self)
        
        self.input_shape = input_shape
        self.resnet_units = resnet_units
        self.embedding_units = embedding_units
        self.projection_units = projection_units
        self.contrastive_factor = contrastive_factor
        
        self.hparams.update({
            'input_shape': input_shape,
            'projection_units': projection_units,
            'contrastive_factor': contrastive_factor,
        })
        
        self.in_channels, self.in_height, self.in_width = input_shape
        
        # ~ convolutional encoder
        # The following defines the ResNet-based convolutional encoder architecture that maps the 
        # input image data into the latent space representation.
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, resnet_units, kernel_size=3, stride=1, padding=1),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            nn.MaxPool2d(kernel_size=2),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            nn.MaxPool2d(kernel_size=2),
        ])
        # This will be the size of the flattened output of the convolutional encoder.
        self.flatten_size = resnet_units * (self.in_height // 4) * (self.in_width // 4)
        
        # ~ embedding layers
        # These layers define a simple MLP network which will transform the immediate flattened
        # representation into the embedding space.
        self.emb_layers = nn.ModuleList()
        prev_units = self.flatten_size
        for c, units in enumerate(self.embedding_units, start=1):
            if c == len(self.embedding_units):
                lay = nn.Linear(prev_units, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.SiLU(),
                )
            
            self.emb_layers.append(lay)
            prev_units = units
            
        self.embedding_size = prev_units
            
        # ~ projection layers
        # These layers define a simple MLP network which will transform the embedding space into
        # the projection space on top of which the contrastive loss will be calculated.
        self.proj_layers = nn.ModuleList()
        prev_units = self.embedding_size
        for c, units in enumerate(self.projection_units, start=1):
            if c == len(self.projection_units):
                lay = nn.Linear(prev_units, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.SiLU(),
                )
            
            self.proj_layers.append(lay)
            prev_units = units
            
        self.projection_size = prev_units
            
    def forward(self, image_data: ImageData):
        x = image_data.x
        
        # ~ convolutional encoder
        for lay in self.conv_layers:
            x = lay(x)
        
        # flatten the final convolutional layer output    
        x = x.view(x.size(0), -1)
        
        # ~ embedding layers (mlp)
        for lay in self.emb_layers:
            x = lay(x)
            
        embedding = x
        
        # ~ projection layers (mlp)
        for lay in self.proj_layers:
            x = lay(x)
            
        projection = x
        
        result = {
            'embedding': embedding,
            'projection': projection,
        }
        return result