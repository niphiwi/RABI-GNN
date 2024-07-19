import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torchvision.transforms as transforms
import pytorch_lightning as pl

##--------
# DECODER

class DecoderNet(nn.Module):
    def __init__(self, inner_dims, seq_len=1):
        super().__init__() 
        self.inner_dims = inner_dims
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(seq_len, inner_dims[0], kernel_size=(2), stride=1, padding=0),  # [c,7,6]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=1, padding=0),       # [c,9,8]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),       # [c,12,11]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(3), stride=1, padding=(0)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(4), stride=1, padding=(0,1)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], 1, kernel_size=(4,3), stride=2, padding=(2,1)),     # [c,30,25]
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
    
class LightningDecoderNet(pl.LightningModule):
    def __init__(self, inner_dims, seq_len, learning_rate):
        super().__init__()        
        self.model = DecoderNet(inner_dims=inner_dims, seq_len=seq_len)
        self.inner_dims = inner_dims
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch  
        # .squeeze(2) to remove the seq_len dim:
        # [batch, seq_len, channel, width, height] -> [batch, seq_len, width, height]
        y_hat = self(X) 
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch  
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log("loss", loss, on_step=False, on_epoch=True)
        
        
class DecoderNet9L(nn.Module):
    def __init__(self, inner_dims, seq_len=1):
        super().__init__() 
        self.inner_dims = inner_dims
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(seq_len, inner_dims[0], kernel_size=(2), stride=1, padding=0),             # [c,7,6]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=1, padding=0),       # [c,9,8]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),       # [c,11,10]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(4), stride=1, padding=(0)),     # [c,14,13]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(4), stride=1, padding=(0)),     # [c,17,16]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], inner_dims[5], kernel_size=(4,3), stride=1, padding=(0)),   # [c,20,18]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[5], inner_dims[6], kernel_size=(5,4), stride=1, padding=(0,1)), # [c,24,19]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[6], inner_dims[7], kernel_size=(5), stride=1, padding=(1)),     # [c,26,21]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[7], 1, kernel_size=(5), stride=1, padding=(0)),                 # [c,30,25]
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded 

class LightningDecoderNet9L(pl.LightningModule):
    def __init__(self, inner_dims, seq_len, learning_rate):
        super().__init__()        
        self.model = DecoderNet9L(inner_dims=inner_dims, seq_len=seq_len)
        self.inner_dims = inner_dims
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch  
        # .squeeze(2) to remove the seq_len dim:
        # [batch, seq_len, channel, width, height] -> [batch, seq_len, width, height]
        y_hat = self(X.squeeze(1)) 
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch  
        y_hat = self(X.squeeze(2))
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X.squeeze(2))
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log("loss", {"loss":loss, "rmse": rmse})
        