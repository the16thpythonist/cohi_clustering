import os
import random
import math
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.aggr import SumAggregation

from cohi_clustering.data import ImageData
from cohi_clustering.data import ImageDataLoader
from cohi_clustering.data import data_list_from_image_dicts
from cohi_clustering.data import GraphData
from cohi_clustering.data import GraphDataLoader
from cohi_clustering.data import ContrastiveGraphDataLoader
from cohi_clustering.data import data_list_from_graph_dicts
from cohi_clustering.data import collate_image_data
from cohi_clustering.layers import ResNetLayer
from cohi_clustering.layers import LogitContrastiveLoss


# ========
# GENERIC
# ========

class AbstractModel(pl.LightningModule):
    """
    The abstract base model class that all models should inherit from. This class provides model agnostic 
    base functionality such as the saving and loading to and from persistent checkpoints.
    """
    
    @classmethod
    def load(cls, path: str):
        """
        Loads the model from a persistent CKPT path at the given absolute ``path``. Returns the 
        reconstructed model instance.
        
        :returns: model instance
        """
        try:
            model = cls.load_from_checkpoint(path)
            model.eval()
            return model
            
        except Exception as exc:
            # Even if we can't load the model itself directly we can load the state dict and the hyperparameters.
            # One of the most common reasons for a problem with the model loading is that the model was exported 
            # with a prior version of the package and the current version has changed the model architecture in 
            # a backward-incompatible way. In this case, we give a meaningful error message to inform the user 
            # that downgrading the package might be required.
            info = torch.load(path)
            
            raise exc
    
    def save(self, path: str) -> None:
        """
        Saves the model as a persistent file to the disk at the given ``path``. The file will be a torch
        ckpt file which is in essence a zipped archive that contains the model's state dictionary and the 
        hyperparameters that were used to create the model. Based on this information, the model can later 
        be reconstructed and loaded.
        
        :param path: The absolute file path of the file to which the model should be saved.
        
        :returns: None
        """
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__,
        }, path)


# =======================================
# CONVOLUTIONAL NEURAL NETWORKS (IMAGES)
# =======================================

class AbstractImageModel(AbstractModel):

    def forward_image_dicts(self, image_dicts: List[Dict], batch_size: int = 200) -> List[Dict]:
        
        if isinstance(image_dicts, ImageDataLoader):
            data_loader = image_dicts
            
        else:
            data_list = data_list_from_image_dicts(image_dicts)
            data_loader = ImageDataLoader(data_list, batch_size=batch_size)
        
        results: List[Dict] = []
        for data in data_loader:
        
            num = data.batch_size
            info: dict = self(data)

            for index in range(num):
                result = {}
                for key, value in info.items():
                    result[key] = value[index].detach().cpu().numpy()
                    
                results.append(result)
        
        return results


class ResNetEncoder(AbstractImageModel):
    
    def __init__(self, 
                 input_shape: List[int],
                 resnet_units: int = 256,
                 embedding_units: List[int] = [128, 512],
                 projection_units: List[int] = [128, 1024],
                 **kwargs,
                 ):
        pl.LightningModule.__init__(self)
        
        self.input_shape = input_shape
        self.resnet_units = resnet_units
        self.embedding_units = embedding_units
        self.projection_units = projection_units
        
        self.hparams.update({
            'input_shape': input_shape,
            'resnet_units': resnet_units,
            'embedding_units': embedding_units,
            'projection_units': projection_units,
        })
        
        self.in_channels, self.in_height, self.in_width = input_shape
        
        # ~ convolutional encoder
        # The following defines the ResNet-based convolutional encoder architecture that maps the 
        # input image data into the latent space representation.
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, resnet_units, kernel_size=1, stride=1, padding=0),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            nn.MaxPool2d(kernel_size=2),
            #nn.AvgPool2d(kernel_size=2),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            ResNetLayer(units=resnet_units),
            nn.MaxPool2d(kernel_size=2),
            #nn.AvgPool2d(kernel_size=2),
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
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                )
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
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
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                )
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
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
            
        #x = F.normalize(x, p=2, dim=-1)
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


class ClusterLevelScheduler(pl.Callback):
    
    def __init__(self, 
                 warmup_epochs: int, 
                 epochs_per_level: int,
                 ):
        pl.Callback.__init__(self)
        self.warmup_epochs = warmup_epochs
        self.epochs_per_level = epochs_per_level
        
        self.cluster_level = 0
        
    def on_train_epoch_start(self, trainer, pl_module):
        
        self.cluster_level = max(0, (trainer.current_epoch - self.warmup_epochs) // self.epochs_per_level)
        pl_module.cluster_level = min(self.cluster_level, pl_module.cluster_depth)


class SimCLRContrastiveCNN(AbstractImageModel):
    
    def __init__(self, 
                 input_shape: List[int],
                 resnet_units: int = 256,
                 embedding_units: List[int] = [128, 512],
                 projection_units: List[int] = [512, 1024],
                 cluster_units: List[int] = [512, 128],
                 cluster_depth: int = 3,
                 contrastive_factor: float = 1.0,
                 contrastive_tau: float = 1.0,
                 cluster_factor: float = 0.0,
                 cluster_tau: float = 1.0,
                 learning_rate: float = 1e-3,
                 **kwargs,
                 ):
        pl.LightningModule.__init__(self)
        
        self.input_shape = input_shape
        self.resnet_units = resnet_units
        self.embedding_units = embedding_units
        self.projection_units = projection_units
        self.cluster_units = cluster_units
        self.cluster_depth = cluster_depth
        self.contrastive_factor = contrastive_factor
        self.contrastive_tau = contrastive_tau
        self.cluster_factor = cluster_factor
        self.cluster_tau = cluster_tau
        self.learning_rate = learning_rate
        
        self.hparams.update({
            'input_shape': input_shape,
            'resnet_units': resnet_units,
            'embedding_units': embedding_units,
            'projection_units': projection_units,
            'cluster_units': cluster_units,
            'cluster_depth': cluster_depth,
            'contrastive_factor': contrastive_factor,
            'contrastive_tau': contrastive_tau,
            'cluster_factor': cluster_factor,
            'cluster_tau': cluster_tau,
            'learning_rate': learning_rate
        })
        
        self.encoder = ResNetEncoder(
            input_shape=input_shape,
            resnet_units=resnet_units,
            embedding_units=embedding_units,
            projection_units=projection_units,
        )
        
        self.cluster_level = self.cluster_depth
        self.num_clusters = 2 ** (cluster_depth + 1)
        cluster_units = cluster_units + [self.num_clusters - 1]
        
        # ~ clustering projector
        prev_units = self.encoder.embedding_size
        self.cluster_layers = nn.ModuleList()
        for c, units in enumerate(cluster_units, start=1):
            if c == len(cluster_units):
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    #nn.Sigmoid(),
                )
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
                )
            
            self.cluster_layers.append(lay)
            prev_units = units
            
    def forward(self, image_data: ImageData):
        result = self.encoder(image_data)
        
        clustering = result['embedding']
        for lay in self.cluster_layers:
            clustering = lay(clustering)
            
        clustering = F.sigmoid(clustering)
        result['clustering'] = clustering
        result['cluster_probas'] = self.probability_vec_with_level(clustering, self.cluster_depth + 1)
        result['cluster_index'] = torch.argmax(result['cluster_probas'], dim=1)
        
        return result
    
    def training_step(self, image_data: ImageData, *args, eps: float = 1e-6, **kwargs):
        
        image_data = image_data[0]
        x, y = image_data.x, image_data.y
        batch_size = image_data.batch_size
        
        result: dict = self.forward(image_data)

        transforms = [
            # self.transform_noise, 
            self.transform_rotate,
            self.transform_gaussian_blur,
            self.transform_translate,
        ]
        random.shuffle(transforms)
        image_data_pos_1 = transforms.pop()(image_data)
        result_pos_1 = self.forward(image_data_pos_1)

        random.shuffle(transforms)
        image_data_pos_2 = transforms.pop()(image_data)
        result_pos_2 = self.forward(image_data_pos_2)

        # ~ contrastive loss
        
        z_0 = result['projection']
        z_0 = F.normalize(z_0, p=2, dim=-1)
        
        z_1 = result_pos_1['projection']
        z_1 = F.normalize(z_1, p=2, dim=-1)
        
        z_2 = result_pos_2['projection']
        z_2 = F.normalize(z_2, p=2, dim=-1)
        
        z = torch.cat([z_0, z_1, z_2], dim=0)
        sim_neg = torch.mm(z, z.t())
        mask = (torch.ones_like(sim_neg) - torch.eye(3 * batch_size, device=self.device)).bool()
        sim_neg = sim_neg.masked_select(mask).view(3 * batch_size, -1)
        # sim_neg = torch.clamp(sim_neg, -1 + eps, 1 - eps)
        # sim_neg = 2 * torch.arctanh(sim_neg)
        sim_neg_exp = torch.exp(sim_neg / self.contrastive_tau)
        
        sim_pos_1 = (z_0 * z_1).sum(dim=-1)
        sim_pos_2 = (z_1 * z_2).sum(dim=-1)
        sim_pos_3 = (z_2 * z_0).sum(dim=-1)
        sim_pos = torch.cat([sim_pos_1, sim_pos_2, sim_pos_3], dim=0)
        # sim_pos = torch.clamp(sim_pos, -1 + eps, 1 - eps)
        # sim_pos = 2 * torch.arctanh(sim_pos)
        #sim_pos = (z_1 * z_2).sum(dim=-1)
        #sim_pos = torch.cat([sim_pos, sim_pos], dim=0)
        sim_pos_exp = torch.exp(sim_pos / self.contrastive_tau)
        
        c = torch.mm(z_2, z_1.t())
        eye = torch.eye(c.size(0), device=self.device).float()
        
        l_pos = (1.0 - (c * eye)).pow(2).mean()
        l_neg = torch.max(torch.zeros_like(eye, device=self.device), c * (1.0 - eye)).pow(2).mean()
        loss = l_pos + 0.02 * l_neg
        
        #sim_pos_exp = torch.exp(sim_pos / 0.2)
        #loss = (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
        # loss = (-torch.log(sim_pos_exp / (torch.clamp_max(sim_neg_exp.sum(dim=-1), 0.05)))).mean()
        self.log('loss_contr', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('sim_neg', sim_neg.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('sim_pos', sim_pos.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('cluster_level', self.cluster_level, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        
        # ~ clustering loss
        if self.cluster_factor != 0:
            
            loss += self.cluster_factor * self.training_clustering(result, result_pos_1, result_pos_2, batch_size)
        
        return loss
    
    def training_clustering(self, result: dict, result_pos_1: dict, result_pos_2: dict, batch_size: int, eps: float = 1e-8):
        
        cl_0 = result['clustering']
        cl_1 = result_pos_1['clustering']
        cl_2 = result_pos_2['clustering']
        
        loss_reg = 0.0
        # cl_1_mean = cl_1.mean(dim=0)
        # cl_2_mean = cl_2.mean(dim=0)
        # loss_reg = (
        #     (cl_1_mean - 0.5).abs().mean() +
        #     (cl_2_mean - 0.5).abs().mean()
        #     # F.binary_cross_entropy(cl_1_mean, torch.ones_like(cl_1_mean) * 0.5) +
        #     # F.binary_cross_entropy(cl_2_mean, torch.ones_like(cl_2_mean) * 0.5)
        # )
        tau = self.cluster_tau
        #tau = 
        
        loss_clust = 0.0
        clust_sim_pos = 0.0
        clust_sim_neg = 0.0
        for level in range(1, self.cluster_depth + 2):
            probs_0 = self.probability_vec_with_level(cl_0, level)
            probs_1 = self.probability_vec_with_level(cl_1, level)
            probs_2 = self.probability_vec_with_level(cl_2, level)
            
            # tree contrastive loss
            #sim_pos = torch.sqrt((probs_1 * probs_2) + eps).sum(dim=-1)
            sim_pos_0 = (torch.sqrt(probs_0 + eps) * torch.sqrt(probs_1 + eps)).sum(dim=-1)
            sim_pos_1 = (torch.sqrt(probs_1 + eps) * torch.sqrt(probs_2 + eps)).sum(dim=-1)
            sim_pos_2 = (torch.sqrt(probs_2 + eps) * torch.sqrt(probs_0 + eps)).sum(dim=-1)
            sim_pos = torch.cat([sim_pos_0, sim_pos_1, sim_pos_2], dim=0)
            sim_pos_exp = torch.exp(sim_pos / tau)
            #loss_clust -= sim_pos.mean()
            
            probs = torch.cat([probs_0, probs_1, probs_2], dim=0)
            probs_sqrt = torch.sqrt(probs + eps)
            #sim_neg = torch.sqrt((probs.unsqueeze(dim=0) * probs.unsqueeze(dim=1)) + eps).sum(dim=-1)
            sim_neg = torch.mm(probs_sqrt, probs_sqrt.t())
            mask = (torch.ones_like(sim_neg) - torch.eye(3 * batch_size, device=self.device)).bool()
            sim_neg = sim_neg.masked_select(mask).view(3 * batch_size, -1)
            sim_neg_exp = torch.exp(sim_neg / tau)
            #loss_clust += sim_neg.mean()
            
            probs_1_sqrt = torch.sqrt(probs_1 + eps)
            probs_2_sqrt = torch.sqrt(probs_2 + eps)
            
            c = torch.mm(probs_2_sqrt, probs_1_sqrt.t())
            eye = torch.eye(c.size(0), device=self.device).float()
            l_pos = (1.0 - (c * eye)).pow(2).mean()
            #l_neg = torch.max(torch.zeros_like(eye, device=self.device), c * (1.0 - eye)).pow(2).mean()
            l_neg = (c * (1.0 - eye)).pow(2).mean()
            
            #loss_clust += sim_neg.mean() - sim_pos.mean()
            if level <= self.cluster_level + 1:
                
                loss_clust = (1 / (self.cluster_level+1)) * (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
                #loss_clust += (1 / (self.cluster_level+1)) * (l_pos + 0.05 * l_neg)
                clust_sim_pos += (1 / (self.cluster_level+1)) * sim_pos.mean()
                clust_sim_neg += (1 / (self.cluster_level+1)) * sim_neg.mean()
            
            # if level == self.cluster_depth + 1:
            #     loss_clust += (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
            #     clust_sim_pos += sim_pos.mean()
            #     clust_sim_neg += sim_neg.mean()
            
            # tree balancing regularization
            probs = torch.cat([probs_1, probs_2], dim=0).mean(dim=0)
            for leftnode in range(0, int((2**level)/2)):
                loss_reg -= (1 / 2**level) * (0.5 * torch.log(probs[2*leftnode]) + 0.5 * torch.log(probs[2*leftnode+1]))
                pass
            
        # clust_sim_pos_exp = torch.exp(clust_sim_pos / self.contrastive_tau)
        # clust_sim_neg_exp = torch.exp(clust_sim_neg / self.contrastive_tau)
        # loss_clust = (-torch.log(clust_sim_pos_exp / (clust_sim_neg_exp.sum(dim=-1)))).mean()
        
        self.log('loss_reg', loss_reg, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('loss_clust', loss_clust, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('clust_sim_pos', clust_sim_pos.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('clust_sim_neg', clust_sim_neg.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss_clust + 0.01 * loss_reg
        
    
    def transform_noise(self, image_data: ImageData) -> ImageData:
        x = image_data.x
        x = x + (torch.randn_like(x) * 0.1).to(self.device)
        return ImageData(x=x, y=image_data.y)
    
    def transform_rotate(self, image_data: ImageData, angle_range: tuple = (-15, 15)) -> ImageData:
        x = image_data.x
        x = torchvision.transforms.functional.rotate(x, angle=random.uniform(*angle_range))
        return ImageData(x=x, y=image_data.y)
    
    def transform_gaussian_blur(self, image_data: ImageData, kernel_size: int = 3, sigma: tuple = (0.1, 1.0)) -> ImageData:
        x = image_data.x
        x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=kernel_size, sigma=random.uniform(*sigma))
        return ImageData(x=x, y=image_data.y)
    
    def transform_translate(self, image_data: ImageData, max_translation: int = 4) -> ImageData:
        x = image_data.x
        
        # Generate random translation values for x and y directions
        translation_x = random.randint(0, max_translation)
        translation_y = random.randint(0, max_translation)
        
        # Use torch.roll to perform circular translation
        x_translated = torch.roll(x, shifts=(translation_y, translation_x), dims=(-2, -1))
        
        return ImageData(x=x_translated, y=image_data.y)
    
    def probability_vec_with_level(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        prob_vec = torch.tensor([], requires_grad=True).to(self.device)
        
        # for u in torch.arange(2 ** level - 2, 2**(level + 1) - 2, dtype=torch.long):
           
        #     u = u + 1
        #     probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).to(self.device)
        #     while True:
                
        #         if u/2 > torch.floor(u/2):
        #             # go left
        #             u = torch.floor((u-0.1)/2).long()
        #             probability_u *= (feature[:, u])
                    
        #         elif u/2 == torch.floor(u/2):
        #             # go right
        #             u = torch.floor((u-0.1)/2).long()
        #             probability_u *= (1.0 - feature[:, u])
                    
        #         else:
        #             print("hello")
                    
        #         if u <= 0:
        #             break
                    
        #     prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(dim=1)), dim=1)
        
        for u in range(2 ** level - 2, 2**(level + 1) - 2):
            
            u = u + 1
            probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).to(self.device)
            
            while True:
                #print('u', u)
                if u % 2 != 0:
                    #print('left')
                    # go left
                    u = math.floor((u - 0.1) / 2)
                    probability_u *= (feature[:, u])
                else:
                    #print('right')
                    # go right
                    u = math.floor((u - 0.1) / 2)
                    probability_u *= (1.0 - feature[:, u])    
                    
                if u <= 0:
                    break
                    
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(dim=1)), dim=1)
                        
        return prob_vec
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            # l2 regularization on the weights. 
            weight_decay=1e-6
        )
    

class ClusteringCallback(pl.Callback): 
    
    def __init__(self, warmup_epochs: int, factor: float):
        
        pl.Callback.__init__(self)
        self.warmup_epochs = warmup_epochs
        self.factor = factor
        
    def on_train_epoch_start(self, trainer, pl_module):
        
        if trainer.current_epoch < self.warmup_epochs:
            pl_module.cluster_factor = 0.0
        else:
            pl_module.cluster_factor = self.factor
 
 
# ===============================
# GRAPH NEURAL NETWORKS (GRAPHS)
# ===============================

class AbstractGraphModel(AbstractModel):
    
    def forward_graphs(self, 
                       graphs: List[dict], 
                       batch_size: int = 1000
                       ) -> List[dict]:
        
        if isinstance(graphs, GraphDataLoader):
            data_loader = graphs
        
        else:
            data_list = data_list_from_graph_dicts(graphs)
            data_loader = GraphDataLoader(data_list, batch_size=batch_size)
            
        results: List[dict] = []
        for data in data_loader:
            
            info: dict = self(data)
            
            num = np.max(data.batch.cpu().detach().numpy()) + 1
            for index in range(num):
                
                result: Dict[str, np.ndarray] = {}
                node_mask = data.batch == index
                edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
                
                for key, value in info.items():
                    
                    if isinstance(value, torch.Tensor):
                        
                        if key.startswith('graph'):
                            result[key] = value[index].detach().numpy()
                            
                        elif key.startswith('node'):
                            result[key] = value[node_mask].detach().numpy()
                            
                        elif key.startswith('edge'):
                            result[key] = value[edge_mask].detach().numpy()
                            
                results.append(result)
                
                
class GATv2Encoder(pl.LightningModule):
    
    def __init__(self,
                 in_features: int,
                 edge_features: int = None,
                 conv_units: List[int] = [128, 128, 128, 128, 128],
                 embedding_units: List[int] = [128, 256],
                 num_heads: int = 5,
                 ):
        nn.Module.__init__(self)
        
        self.in_features = in_features
        self.edge_features = edge_features
        self.conv_units = conv_units
        self.embedding_units = embedding_units
        self.num_heads = num_heads
        
        self.hparams.update({
            'in_features': in_features,
            'edge_features': edge_features,
            'conv_units': conv_units,
            'embedding_units': embedding_units,
            'num_heads': num_heads,
        })
        
        self.lay_emb = nn.Linear(in_features, conv_units[0])
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        prev_units = conv_units[0]
        for units in self.conv_units:
            
            lay_conv = GATv2Conv(
                in_channels=prev_units,
                out_channels=units,
                edge_dim=edge_features,
                concat=False,
                heads=self.num_heads,
                #residual=True,
            )
            lay_bn = nn.BatchNorm1d(units)
            
            self.conv_layers.append(lay_conv)
            self.bn_layers.append(lay_bn)
            prev_units = units
        
        self.lay_pool = SumAggregation()
        
        self.emb_layers = nn.ModuleList()
        for c, units in enumerate(self.embedding_units, start=1):
            if c == len(self.embedding_units):
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                )
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
                )
            self.emb_layers.append(lay)
            prev_units = units
            
        self.embedding_size = prev_units
            
    def forward(self, data: GraphData) -> dict:
        
        x = data.x
        
        x = self.lay_emb(x)
        
        for lay_conv, lay_bn in zip(self.conv_layers, self.bn_layers):
            
            x = lay_conv(
                x=x, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr
            )
            x = lay_bn(x)
            x = F.leaky_relu(x)
            
        node_embedding = x
        node_embedding = self.lay_pool(node_embedding, data.batch)
        
        graph_embedding = node_embedding
        for lay in self.emb_layers:
            graph_embedding = lay(graph_embedding)
            
        return {
            'node_embedding': node_embedding,
            'graph_embedding': graph_embedding,
        }
        
        
class SimCLRContrastiveGNN(AbstractGraphModel):
    
    def __init__(self,
                 in_features: int,
                 edge_features: int = None,
                 conv_units: List[int] = [256, 256, 256],
                 embedding_units: List[int] = [256, 512],
                 num_heads: int = 16,
                 projection_units: List[int] = [2048],
                 cluster_units: List[int] = [64, 128],
                 cluster_factor: float = 0.0,
                 cluster_depth: int = 3,
                 cluster_tau: float = 1.0,
                 contrastive_factor: float = 1.0,
                 contrastive_tau: float = 1.0,
                 learning_rate: float = 1e-5,
                 ):
        AbstractGraphModel.__init__(self)

        self.in_features = in_features
        self.edge_features = edge_features
        self.conv_units = conv_units
        self.embedding_units = embedding_units
        self.num_heads = num_heads
        self.projection_units = projection_units
        self.cluster_units = cluster_units
        self.cluster_factor = cluster_factor
        self.cluster_depth = cluster_depth
        self.cluster_tau = cluster_tau
        self.contrastive_factor = contrastive_factor
        self.contrastive_tau = contrastive_tau
        self.learning_rate = learning_rate

        self.hparams.update({
            'in_features': in_features,
            'edge_features': edge_features,
            'conv_units': conv_units,
            'embedding_units': embedding_units,
            'num_heads': num_heads,
            'projection_units': projection_units,
            'cluster_units': cluster_units,
            'cluster_factor': cluster_factor,
            'cluster_depth': cluster_depth,
            'cluster_tau': cluster_tau,
            'contrastive_factor': contrastive_factor,
            'contrastive_tau': contrastive_tau,
            'learning_rate': learning_rate,
        })

        self.encoder = GATv2Encoder(
            in_features=in_features,
            edge_features=edge_features,
            conv_units=conv_units,
            embedding_units=embedding_units,
            num_heads=num_heads,
        )
        
        # ~ projection head
        self.proj_layers = nn.ModuleList()
        prev_units = self.encoder.embedding_size
        for c, units in enumerate(self.projection_units, start=1):
            if c == len(self.projection_units):
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                )
                # lay = nn.Linear(prev_units, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
                )
            self.proj_layers.append(lay)
            prev_units = units
            
        # ~ clustering head
        
        self.cluster_level = self.cluster_depth
        self.num_clusters = 2 ** (cluster_depth + 1)
        cluster_units = cluster_units + [self.num_clusters - 1]
        
        self.cluster_layers = nn.ModuleList()
        prev_units = self.encoder.embedding_size
        for c, units in enumerate(cluster_units, start=1):
            if c == len(cluster_units):
                #lay = nn.Linear(prev_units, units)
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                )
                #print('TRUE')
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.GELU(),
                )
            self.cluster_layers.append(lay)
            prev_units = units
            
        self.logit_loss = LogitContrastiveLoss()    
        
    ## --- utility methods ---
    
    def forward_graph_loader(self, 
                             graph_loader: GraphDataLoader | ContrastiveGraphDataLoader,
                             ) -> List[dict]:
        
        results: List[dict] = []
            
        for data in graph_loader:
            
            if isinstance(graph_loader, ContrastiveGraphDataLoader):
                data = data['data']
            
            batch_size = int(torch.max(data.batch).item() + 1)
            batch_result: dict = self.forward(data)
            
            for index in range(batch_size):
                
                result = {}
                for key, value in batch_result.items():
                    value = value[index].detach().cpu().numpy()
                    result[key] = value
                    result[key.replace('graph_', '')] = value
                
                results.append(result)
                
        return results
                
            
    ## --- model implementation ---
            
    def forward(self, data: GraphData):
        
        result = self.encoder(data)
        graph_embedding = result['graph_embedding']
        
        graph_projection = graph_embedding
        for lay in self.proj_layers:
            graph_projection = lay(graph_projection)
            
        result['graph_projection'] = graph_projection
        
        clustering = graph_embedding
        for lay in self.cluster_layers:
            clustering = lay(clustering)
            
        clustering = F.sigmoid(clustering)
        
        result['graph_clustering'] = clustering
        result['graph_cluster_probas'] = self.probability_vec_with_level(clustering, self.cluster_depth + 1)
        result['graph_cluster_index'] = torch.argmax(result['graph_cluster_probas'], dim=1)
        
        return result
    
    def probability_vec_with_level(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        prob_vec = torch.tensor([], requires_grad=True).to(self.device)
        for u in torch.arange(2 ** level - 2, 2**(level + 1) - 2, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).to(self.device)
            while (u > 0):
                if u/2 > torch.floor(u/2):
                    # go left
                    u = torch.floor(u/2).long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # go right
                    u = torch.floor(u/2).long()
                    probability_u *= 1.0 - feature[:, u]
                    
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(dim=1)), dim=1)
        
        return prob_vec
    
    def training_step(self, graph_data_map: dict[str, GraphData], *args, eps: float = 1e-6, **kwargs):
        
        ## --- forward pass ---
        # The forward pass for the original graph object itself.
        graph_data: GraphData = graph_data_map['data']
        batch_size = torch.max(graph_data.batch).item() + 1
        result: dict = self.forward(graph_data)
        
        ## --- augmentations ---
        # Then we perform the separate forward passes for the augmented graph objects.
        
        graph_aug_1: GraphData = graph_data_map['aug_1']
        result_pos_1: dict = self.forward(graph_aug_1)
        
        graph_aug_2: GraphData = graph_data_map['aug_2']
        result_pos_2: dict = self.forward(graph_aug_2)
        
        ## --- contrastive loss ---
        # Using all of this information, as well as the loss function we can calculate 
        # the contrastive loss.
        
        z_0 = result['graph_projection']
        z_0 = F.normalize(z_0, p=2, dim=-1)
        
        z_1 = result_pos_1['graph_projection']
        z_1 = F.normalize(z_1, p=2, dim=-1)
        
        z_2 = result_pos_2['graph_projection']
        z_2 = F.normalize(z_2, p=2, dim=-1)
        
        z = torch.cat([z_0, z_1, z_2], dim=0)
        
        # We consider all other samples from the batch as the negative samples and therefore 
        # calculate the similarity between all sampled in the batch with each other (matrix)
        sim_neg = torch.mm(z, z.t())
        mask = (torch.ones_like(sim_neg) - torch.eye(3 * batch_size, device=self.device)).bool()
        sim_neg = sim_neg.masked_select(mask).view(3 * batch_size, -1)
        sim_neg_exp = torch.exp(sim_neg / self.contrastive_tau)
        
        # The positive samples to be pulled together are the pairs of augmented samples. We 
        # are using a kind of triangle here were we pull the original with each of the two 
        # augmented samples together but also the augmented samples with each other.
        sim_pos_1 = (z_0 * z_1).sum(dim=-1)
        sim_pos_2 = (z_1 * z_2).sum(dim=-1)
        sim_pos_3 = (z_2 * z_0).sum(dim=-1)
        sim_pos = torch.cat([sim_pos_1, sim_pos_2, sim_pos_3], dim=0)
        sim_pos_exp = torch.exp(sim_pos / self.contrastive_tau)
        #sim_pos_exp = torch.exp(sim_pos / 0.2)
        #loss = (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
        
        c = torch.mm(z_2, z_1.t())
        eye = torch.eye(c.size(0), device=self.device).float()
        
        l_pos = (1.0 - (c * eye)).pow(2).mean()
        l_neg = torch.max(torch.zeros_like(eye, device=self.device), c * (1.0 - eye)).pow(2).mean()
        loss = l_pos + 0.1 * l_neg
        
        self.log('loss_contr', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('sim_neg', sim_neg.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('sim_pos', sim_pos.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        
        ## --- clustering loss ---
        # If the clustering factor is set to a value greater than 0, we will also calculate the clustering 
        # loss and add it to overall loss value.
        
        if self.cluster_factor != 0:
            
            loss += self.cluster_factor * self.training_clustering(
                result=result,
                result_pos_1=result_pos_1,
                result_pos_2=result_pos_2,
                batch_size=batch_size,
            )
        
        return loss
    
    def probability_vec_with_level(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        prob_vec = torch.tensor([], requires_grad=True).to(self.device)
        
        # for u in torch.arange(2 ** level - 2, 2**(level + 1) - 2, dtype=torch.long):
           
        #     u = u + 1
        #     probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).to(self.device)
        #     while True:
                
        #         if u/2 > torch.floor(u/2):
        #             # go left
        #             u = torch.floor((u-0.1)/2).long()
        #             probability_u *= (feature[:, u])
                    
        #         elif u/2 == torch.floor(u/2):
        #             # go right
        #             u = torch.floor((u-0.1)/2).long()
        #             probability_u *= (1.0 - feature[:, u])
                    
        #         else:
        #             print("hello")
                    
        #         if u <= 0:
        #             break
                    
        #     prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(dim=1)), dim=1)
        
        for u in range(2 ** level - 2, 2**(level + 1) - 2):
            
            u = u + 1
            probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).to(self.device)
            
            while True:
                #print('u', u)
                if u % 2 != 0:
                    #print('left')
                    # go left
                    u = math.floor((u - 0.1) / 2)
                    probability_u *= (feature[:, u])
                else:
                    #print('right')
                    # go right
                    u = math.floor((u - 0.1) / 2)
                    probability_u *= (1.0 - feature[:, u])    
                    
                if u <= 0:
                    break
                    
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(dim=1)), dim=1)
                        
        return prob_vec
    
    def training_clustering(self, 
                            result: dict, 
                            result_pos_1: dict, 
                            result_pos_2: dict, 
                            batch_size: int, 
                            eps: float = 1e-8
                            ) -> torch.Tensor:
        
        cl_0 = result['graph_clustering']
        cl_1 = result_pos_1['graph_clustering']
        cl_2 = result_pos_2['graph_clustering']
        
        loss_reg = 0.0
        # cl_1_mean = cl_1.mean(dim=0)
        # cl_2_mean = cl_2.mean(dim=0)
        # loss_reg = (
        #     (cl_1_mean - 0.5).abs().mean() +
        #     (cl_2_mean - 0.5).abs().mean()
        #     # F.binary_cross_entropy(cl_1_mean, torch.ones_like(cl_1_mean) * 0.5) +
        #     # F.binary_cross_entropy(cl_2_mean, torch.ones_like(cl_2_mean) * 0.5)
        # )
        tau = self.cluster_tau

        loss_clust = 0.0
        clust_sim_pos = 0.0
        clust_sim_neg = 0.0
        
        for level in range(1, self.cluster_depth + 2):
            probs_0 = self.probability_vec_with_level(cl_0, level)
            probs_1 = self.probability_vec_with_level(cl_1, level)
            probs_2 = self.probability_vec_with_level(cl_2, level)
            
            # tree contrastive loss
            #sim_pos = torch.sqrt((probs_1 * probs_2) + eps).sum(dim=-1)
            sim_pos_0 = (torch.sqrt(probs_0 + eps) * torch.sqrt(probs_1 + eps)).sum(dim=-1)
            sim_pos_1 = (torch.sqrt(probs_1 + eps) * torch.sqrt(probs_2 + eps)).sum(dim=-1)
            sim_pos_2 = (torch.sqrt(probs_2 + eps) * torch.sqrt(probs_0 + eps)).sum(dim=-1)
            sim_pos = torch.cat([sim_pos_0, sim_pos_1, sim_pos_2], dim=0)
            sim_pos_exp = torch.exp(sim_pos / tau)
            #loss_clust -= sim_pos.mean()
            
            probs = torch.cat([probs_0, probs_1, probs_2], dim=0)
            probs_sqrt = torch.sqrt(probs + eps)
            #sim_neg = torch.sqrt((probs.unsqueeze(dim=0) * probs.unsqueeze(dim=1)) + eps).sum(dim=-1)
            sim_neg = torch.mm(probs_sqrt, probs_sqrt.t())
            mask = (torch.ones_like(sim_neg) - torch.eye(3 * batch_size, device=self.device)).bool()
            sim_neg = sim_neg.masked_select(mask).view(3 * batch_size, -1)
            sim_neg_exp = torch.exp(sim_neg / tau)
            #loss_clust += sim_neg.mean()
            
            probs_1_sqrt = torch.sqrt(probs_1 + eps)
            probs_2_sqrt = torch.sqrt(probs_2 + eps)
            
            c = torch.mm(probs_2_sqrt, probs_1_sqrt.t())
            eye = torch.eye(c.size(0), device=self.device).float()
            l_pos = (1.0 - (c * eye)).pow(2).mean()
            l_neg = torch.max(torch.ones_like(eye, device=self.device) * 0.1, c * (1.0 - eye)).pow(2).mean()
            #l_neg = (c * (1.0 - eye)).pow(2).mean()
            
            #loss_clust += sim_neg.mean() - sim_pos.mean()
            if level <= self.cluster_level + 1:
                
                loss_clust = (1 / (self.cluster_level+1)) * (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
                #loss_clust += (1 / (self.cluster_level+1)) * (l_pos + 0.1 * l_neg)
                #loss_clust = (1 / (self.cluster_level+1)) * self.logit_loss(c)
                clust_sim_pos += (1 / (self.cluster_level+1)) * sim_pos.mean()
                clust_sim_neg += (1 / (self.cluster_level+1)) * sim_neg.mean()
            
            # clust_sim_pos += sim_pos
            # clust_sim_neg += sim_neg
            
            # if level == self.cluster_depth + 1:
            #     loss_clust += (-torch.log(sim_pos_exp / (sim_neg_exp.sum(dim=-1)))).mean()
            #     clust_sim_pos += sim_pos.mean()
            #     clust_sim_neg += sim_neg.mean()
            
            # tree balancing regularization
            probs = torch.cat([probs_1, probs_2], dim=0).mean(dim=0)
            for leftnode in range(0, int((2**level)/2)):
                loss_reg -= (1 / 2**level) * (0.5 * torch.log(probs[2*leftnode]) + 0.5 * torch.log(probs[2*leftnode+1]))
                pass
            
        # clust_sim_pos_exp = torch.exp(clust_sim_pos / self.contrastive_tau)
        # clust_sim_neg_exp = torch.exp(clust_sim_neg / self.contrastive_tau)
        # loss_clust = (-torch.log(clust_sim_pos_exp / (clust_sim_neg_exp.sum(dim=-1)))).mean()
        
        self.log('cluster_level', self.cluster_level, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log('loss_clust', loss_clust, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('loss_reg', loss_reg, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('clust_sim_pos', clust_sim_pos.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('clust_sim_neg', clust_sim_neg.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss_clust + 0.1 * loss_reg
    
    def configure_optimizers(self):
        
        # return torch.optim.SGD(
        #     self.parameters(), 
        #     lr=self.learning_rate, 
        #     #momentum=0.9, 
        #     #weight_decay=1e-6
        # )
        
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            # l2 regularization on the weights. 
            #weight_decay=1e-6
        )
        
    def save(self, path: str) -> None:
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'SimCLRContrastiveGNN':
        """
        Load a model from the given path.
        """
        model = cls.load_from_checkpoint(path)
        model.eval()
        return model