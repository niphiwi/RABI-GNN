import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear, BatchNorm
import torch_geometric.transforms as T
import pytorch_lightning as pl

#from utils.funcational import WeightedMSELoss

class GNN(nn.Module):
    def __init__(self, seq_len=10, hidden_dim=256, embedding_dim=5, n_layers=5, n_heads=3, dropout=0.1):
        super(GNN, self).__init__()
        input_dim = seq_len
        output_dim = seq_len

        self.dropout = nn.Dropout(p=dropout)

        # First Layer
        self.embedding_layer = Linear(2*input_dim, embedding_dim) # embedding layer to encode known/unknown state
        self.first_layer = GATv2Conv(input_dim+embedding_dim, hidden_dim, heads=n_heads, edge_dim=3)

        # Inner Layers
        self.n_layers=n_layers
        self.inner_layers = nn.ModuleList()
        for _ in range(n_layers-2):
            self.inner_layers.append(
                GATv2Conv(hidden_dim*n_heads+embedding_dim, hidden_dim, heads=n_heads, edge_dim=3)
            )

        # Last Layer
        self.last_layer = GATv2Conv(hidden_dim*n_heads+embedding_dim, output_dim, heads=1, edge_dim=3)


        self.transform = T.Compose([
            T.Distance(norm=False),
            T.Cartesian(),
        ])
        
    def forward(self, data):
        """
        Args:

        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        known_mask = data.known.float()

        if self.training:
            # Add jitter to positions
            device = data.pos.device
            noise = torch.normal(mean=0, std=0.01, size=data.pos.size()).to(device)
            data.pos += noise

        # Embedding for known/unknown mask
        skip_connection = torch.concat([x, known_mask],dim=1)
        skip_connection = self.embedding_layer(skip_connection)

        # First layer
        x = torch.concat([x, skip_connection],dim=1)
        x = F.elu(self.first_layer(x, edge_index, edge_attr))

        # # Convert x and edge_attr to float, to 
        # x = x.float()
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr.float()
        
        # Inner layers
        for layer in self.inner_layers:
            if self.training:
                x = self.dropout(x)
            x = torch.concat([x, skip_connection],dim=1)
            x = F.elu(layer(x, edge_index, edge_attr))

        # Last layer
        x = torch.concat([x, skip_connection],dim=1)
        x = self.last_layer(x, edge_index, edge_attr)

        return x

class PLModule(pl.LightningModule):
    def __init__(self, hparams: dict = {}):
        super().__init__()

        # Saving hyperparameters
        self.save_hyperparameters(hparams, ignore=['model']) 
        self.lr = hparams["learning_rate"]
        self.weight_decay = hparams["weight_decay"]

        self.model = GNN(
            seq_len=hparams["seq_len"],
            hidden_dim=hparams["hidden_dim"],
            embedding_dim=hparams["embedding_dim"],
            n_layers=hparams["n_layers"],
            n_heads=hparams["n_heads"],
            dropout=hparams["dropout"],
        )

        self.loss_mse = nn.MSELoss()

    def forward(self, data):
        return self.model(data)
    
    def freeze_initial_layers(self):
        # Freeze first_layer and inner layers
        for param in self.model.first_layer.parameters():
            param.requires_grad = False
        for layer in self.model.inner_layers[:1]: 
            for param in layer.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_mse(y_hat, batch.y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True, logger=True, prog_bar=True, batch_size=len(batch.batch))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss_mse = self.loss_mse(y_hat, batch.y)
        self.log("val_loss", loss_mse, on_epoch=True, sync_dist=True, logger=True, prog_bar=True, batch_size=len(batch.batch))
        self.log("hp_metric", loss_mse)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss_mse = self.loss_mse(y_hat, batch.y)
        loss = torch.sqrt(loss_mse)
        self.log("test_loss_rmse", loss)