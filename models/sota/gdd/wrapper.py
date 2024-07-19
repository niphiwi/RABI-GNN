from models.sota.gdd import architectures
import torch
import os


class GasDistributionDecoder():
    def __init__(self):
        
        model_path = os.path.join(os.path.dirname(__file__), "checkpoints/epoch=49-step=75900.ckpt")
        self.gdd = architectures.LightningDecoderNet.load_from_checkpoint(model_path).to('cpu')
        self.mean = 0
        self.std = 1

        self.gdd_train_max = 15.3748
    
    def set_normalization_params(self, normalization_params):
        self.mean = normalization_params['mean']
        self.std = normalization_params['std']

    def prepare_data(self, data):
        known_nodes = data.known[:,-1]
        x = data.x[known_nodes][:,-1].clone()
        x = x.view(6,5).unsqueeze(0)

        # de-normalize 
        x = x * self.std + self.mean

        # normalize to training set of gdd
        x = x / self.gdd_train_max
        return x

    def predict(self, data):
        x = self.prepare_data(data)
        pred = self.gdd(x)

        # de-normalize gdd normalization
        pred = pred * self.gdd_train_max

        # zscore normalize pred with dataset.normalization_params
        pred = (pred - self.mean) / self.std
        return pred.squeeze(0).detach()