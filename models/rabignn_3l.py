import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm
import pytorch_lightning as pl

class GNN(nn.Module):
    def __init__(self, seq_len=10, hidden_dim=256, n_heads=4, dropout=0.3):
        super(GNN, self).__init__()
        self.seq_len = seq_len
        self.input_dim = seq_len
        self.output_dim = 1
        self.n_layers = 3
        self.dropout = dropout
 
        # Adjust the input dim to account for concatenation with known_mask
        adjusted_input_dim = self.input_dim + self.input_dim 

        self.first_layer = GATv2Conv(adjusted_input_dim, 
                                     hidden_dim, 
                                     heads=n_heads, 
                                     edge_dim=3)
        self.first_bn = BatchNorm(hidden_dim * n_heads)

        self.inner_layer = GATv2Conv(hidden_dim * n_heads, hidden_dim, heads=n_heads, edge_dim=3)
        self.inner_bn = BatchNorm(hidden_dim * n_heads)

        self.last_layer = GATv2Conv(hidden_dim*n_heads+adjusted_input_dim, 
                                    self.output_dim, 
                                    heads=1, 
                                    edge_dim=3)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        known_mask = data.known.float()

        x = torch.concat([x, known_mask], dim=1)

        skip_connection = x

        # First layer
        x = self.first_layer(x, edge_index, edge_attr)
        x = self.first_bn(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Inner layer
        x = self.inner_layer(x, edge_index, edge_attr)
        x = self.inner_bn(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Last layer
        x = torch.concat([x, skip_connection], dim=1)
        x = self.last_layer(x, edge_index, edge_attr)

        return x


class PLModule(pl.LightningModule):
    def __init__(self, hparams: dict = {}):
        super().__init__()

        # Saving hyperparameters
        self.save_hyperparameters() 
        self.lr = hparams.get("learning_rate", 1e-3)
        self.weight_decay = hparams.get("weight_decay", 0.0005)

        self.model = GNN(
            seq_len=hparams.get("seq_len", 10),
            hidden_dim=hparams.get("hidden_dim", 256),
            n_heads=hparams.get("n_heads", 2),
        )

        self.loss_mse = nn.MSELoss()

    def forward(self, data):
        return self.model(data)
    
    def freeze_for_transfer(self):
        # Freeze first_layer and inner layer
        for param in self.model.first_layer.parameters():
            param.requires_grad = False
        for param in self.model.first_bn.parameters():
            param.requires_grad = False
        for param in self.model.inner_layer.parameters():
            param.requires_grad = False
        for param in self.model.inner_bn.parameters():
            param.requires_grad = False

        # Set dropout to off
        # self.model.dropout.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch.y[:,-1].unsqueeze(1)
        y_hat = self.forward(batch)
        loss = self.loss_mse(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True, logger=True, prog_bar=True, batch_size=len(batch.batch))
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y[:,-1].unsqueeze(1)
        y_hat = self.forward(batch)
        loss_mse = self.loss_mse(y_hat, y)
        self.log("val_loss", loss_mse, on_epoch=True, sync_dist=True, logger=True, prog_bar=True, batch_size=len(batch.batch))
        self.log("hp_metric", loss_mse)

    def test_step(self, batch, batch_idx):
        y = batch.y[:,-1].unsqueeze(1)
        y_hat = self.forward(batch)
        loss_mse = self.loss_mse(y_hat, y)
        loss = torch.sqrt(loss_mse)
        self.log("test_loss_rmse", loss)