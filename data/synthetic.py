import os.path as osp
from typing import Literal
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl

from utils.transforms import Add2DMask, Apply2DMask

def sequentialize_data(x, seq_len, sliding_step=4):
    r""" Splits the input data into torch.tensor of torch.Size([<number of sampels>, seq_len, 30, 25])
    Init args:
        x (torch.tensor): The data of the synthetic dataset of torch.Size([120, 210, 30, 25])
        seq_len (int): The length of the sequences.
    """

    # Get the number of sequences, that each source simulation generates
    sliding_step = sliding_step
    n_seq = x[0].unfold(0, seq_len, sliding_step).size()[0]

    # Create an empty tensor that is going to be filled up in the loop
    # Dims: [source simulations, sequences, images_per_sequence, width, height]
    new_x = torch.empty([x.size()[0], n_seq, seq_len, 30, 25])
    new_x = new_x.type_as(x)

    # Loop over the source simulations (360 = 30 positions * 12 wind sets)
    for i in range(x.shape[0]):
        sim = x[i]

        # Slice the continuous sequence into sequences of specific length (seq_len)
        sim = sim.unfold(0, seq_len, sliding_step)

        # Change ordering of dimensions
        sim = torch.permute(sim, (0, 3, 1, 2))
        new_x[i] = sim

    return new_x.reshape(-1, seq_len, 30, 25)  

def create_data_obj(imgs):
    """time-aggregation approach 
    Input args:
        imgs (torch.Tensor): tensor of torch.Size([<number of time steps>,30,25])"""
    
    rows, cols = torch.meshgrid(torch.arange(imgs.shape[1]), torch.arange(imgs.shape[2]), indexing="ij")
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    pos = torch.stack([rows, cols]).float().t()

    num_t = imgs.shape[0]
    
    y = imgs.view(num_t,-1)
    y = y.transpose(0,1) # reorder dimensions to torch.Size([num_nodes, num_timesteps])
    return Data(pos=pos, orig_pos=pos, y=y, ground_truth=y)

def create_graphs(data, transform):
    """ Creates a list of single graphs based on a sequence of images, containing all features from
    multiple time steps (time-aggregation approach). 
    """
    
    graph_list = []

    for source_pos in tqdm(range(data.shape[0]), desc="Build graphs..."):
        graph = create_data_obj(data[source_pos])
        graph_list.append(transform(graph))

    return graph_list

# ~~~~~~~~~~~
# DATASET
# ~~~~~~~~~~~
class SyntheticDataset(InMemoryDataset):
    """
    Creates a torch_geometric dataset for synthetic data.

    Args:
        root (str): The root directory for storing the dataset.
        type (Literal["train", "valid", "test"]): The type of the dataset ("train", "valid", or "test").
        seq_len (int): The desired sequence length.
        is_grid_n_rnd (bool): A flag for whether to apply grid masking with randomness.
        is_grid (bool): A flag for whether to apply grid masking.
    """
    def __init__(self, root: str, type: Literal["train", "valid", "test"], 
                 seq_len: int=10, sliding_step: int=4, radius: float=0.3, n_nodes: int=750, transform=None, temporary=False):
        self.type = type.lower()
        self.root = root
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.processed_path = osp.join(root, "processed", self.type, "data.pt")
        self.n_nodes = n_nodes
        self.radius = radius
        self.temporary = temporary

        self.normalization_params = {"mean": 2.1744942665100098, "std": 3.9583189487457275}

        if not self.temporary:
            self.pre_transform = T.Compose([
                T.NormalizeScale(),
                T.RadiusGraph(r=self.radius, loop=True, max_num_neighbors=200),
            ])
        else:
            self.pre_transform = T.Compose([
                T.NormalizeScale(),
            ])

        if transform is None:
            self.transform = T.Compose([
                T.Distance(norm=False),
                T.Cartesian(),
                Add2DMask(0.04, seq_len=10),
                Apply2DMask(),
            ])
        else:
            self.transform = transform

        super().__init__(root, transform=self.transform, pre_transform=self.pre_transform)
        self.load(self.processed_paths[0])

    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_radius{self.radius}_seqlen{self.seq_len}_slidingstep{self.sliding_step}", self.type)
        
    @property
    def raw_file_names(self):
        return [f"{self.type}.pt"]
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        pass
    
    def process(self):
        data = torch.load(self.raw_paths[0])
        data = sequentialize_data(data, self.seq_len, self.sliding_step)
        data_list = create_graphs(data, self.pre_transform)
        
        self.save(data_list, self.processed_paths[0])

    def get(self, idx):
        if not self.temporary:
            data = super().get(idx)
        else:
            data = super().get(idx)
            data = T.RadiusGraph(r=self.radius, loop=True, max_num_neighbors=200)(data)
            
        return data

# ~~~~~~~~~~~
# DATAMODULE
# ~~~~~~~~~~~
class SyntheticDataModule(pl.LightningDataModule):
    r"""DataModule that loads the synthetic dataset.
    """
    def __init__(self, seq_len: int=10, radius: float=0.3, batch_size: int=32, num_workers: int=0, shuffle: bool=False, n_nodes=750,
                 transform=None):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_nodes = n_nodes
        self.radius = radius
        self.transform = transform



    def setup(self, stage: str):
        # Train dataset
        self.train_dataset = SyntheticDataset(root="data/30x25", type="train", radius=self.radius, transform=self.transform)  
        # Val     
        self.val_dataset = SyntheticDataset(root="data/30x25", type="valid", radius=self.radius, transform=self.transform)  
        # Test
        self.test_dataset = SyntheticDataset(root="data/30x25", type="test", radius=self.radius, transform=self.transform)  
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)