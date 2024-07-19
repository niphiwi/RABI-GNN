import torch
from torch_geometric.transforms import BaseTransform, KNNGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import random

class AddMask:
    r"""Add a mask to mask random nodes. True denotes known nodes.""" 
    def __init__(self, masked_percentage):
        self.masked_percentage = masked_percentage
        
    def __call__(self, data):
        num_nodes = data.num_nodes
        num_masked_nodes = int(num_nodes * self.masked_percentage)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        masked_indices = torch.randperm(num_nodes)[:num_masked_nodes]
        mask[masked_indices] = True
        data.known = mask
        return data

class Add2DMask:
    r"""Add a mask to mask random nodes. True denotes known nodes.
    Mask is two dimensional, meaning that the temporal dimension is also taken into account.""" 
    def __init__(self, masked_percentage, seq_len):
        self.masked_percentage = masked_percentage
        self.seq_len = seq_len

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_masked_nodes = int(num_nodes * self.masked_percentage)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        masked_indices = torch.randperm(num_nodes)[:num_masked_nodes]
        mask[masked_indices] = True
        data.known = mask.repeat(self.seq_len,1).permute(1,0)
        return data    


class AddGridMask:
    r"""Add a mask to mask a regular grid of nodes. True denotes known nodes. 
    Args:
        n (int): The number of cells between a 'sensor' in x and y directions.
    """ 
    def __init__(self, n: int=5):
        n = n
        cell_size = 1
        x_min = 0
        y_min = 0

        # Create an empty image-like representation
        grid = torch.zeros((30, 25))

        # Check the cells, where grid sampling position is located
        for row in range(int(n/2), grid.shape[0], n): 
            for col in range(int(n/2), grid.shape[1], n):
                grid[row, col] = 1

        # Find positions of sensor grid
        marked_cells = torch.nonzero(grid)
        pos_x_grid = marked_cells[:,0] * cell_size + x_min
        pos_y_grid = marked_cells[:,1] * cell_size + y_min
        pos_grid = torch.stack([pos_x_grid, pos_y_grid], dim=1)
        self.sensor_positions = pos_grid

    def __call__(self, data):
        # Find matching positions of sensor grid in array from data graph 
        equality_matrix = (data.orig_pos[:, None] == self.sensor_positions).all(dim=2)
        matching_mask = equality_matrix.any(dim=1)
        data.ground_truth = data.y
        #data.x = data.x * matching_mask.view(-1,1)
        data.known = (torch.ones(750) * matching_mask).bool()
        data.known = data.known.repeat(data.y.shape[1],1).permute(1,0)

        return data

class ApplyMask(BaseTransform):
    r"""Apply mask to simulate sensor positions."""    
    def __call__(self, data: Data) -> Data:
        data.x = data.y * data.known.unsqueeze(1)
        return data
    
class Apply2DMask(BaseTransform):
    r"""Apply mask to simulate sensor positions."""    
    def __call__(self, data: Data) -> Data:
        data.x = data.y * data.known
        return data  

class SampleSubgraph(BaseTransform):
    r"""Creates a subgraph.
    Args:
        n (int): The number of nodes in the subgraph.    
    """
    def __init__(self, n):
        self.n = n
    
    def __call__(self, data: Data) -> Data:
        # randomly select nodes from the original graph
        selected_mask = random.sample(range(750), self.n)

        # extract the corresponding nodes and edges from the original graph
        selected_nodes = data.y[selected_mask,:]

        # create a new Data object from the selected nodes and edges
        data = Data(y=selected_nodes, ground_truth=data.y, pos=data.pos[selected_mask], orig_pos=data.orig_pos[selected_mask])
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n})'
    
class ConnectMissingToKnown(BaseTransform):
    r"""Create edge_index by connecting the missing nodes to all known nodes."""
    def __init__(self, undirected=True):
        self.undirected = undirected

    def __call__(self, data: Data) -> Data:
        known_indices = torch.nonzero(data.known, as_tuple=True)[0]
        missing_indices = torch.nonzero(~data.known, as_tuple=True)[0]
        if self.undirected:
            edge_indices = to_undirected(torch.cartesian_prod(missing_indices, known_indices).t())
        else:
            edge_indices = torch.cartesian_prod(known_indices, missing_indices).t()

        data.edge_index = edge_indices
        return data

class KNNtoMissingNodes(BaseTransform):
    r"""Create edge_index by connecting the missing nodes to all known nodes."""
    def __init__(self, k: int=10):
        self.k = k

    def filter_list_of_lists(self, l1, l2):
        """ Filter a list of list and only keep the elements that don't contain any elements from another list."""
        l3 = [[element for element in pair if element not in l2] for pair in l1]
        l3 = [pair for pair in l3 if len(pair)==2]
        return l3

    def create_edges_missing(self, data: Data) -> Data:
        """
        Create missing edges based on a KNNGraph.

        Args:
            data (Tensor): Input data tensor representing the graph.

        Returns:
            Tensor: Tensor representing the missing edges.

        """  
        data_knn = data.clone()
        edges_knn = KNNGraph(k=4, loop=False)(data_knn).edge_index
        edges_knn = torch.transpose(edges_knn, 0, 1)
        idx_known = torch.nonzero(data.known).flatten()
        edges_missing = torch.tensor(self.filter_list_of_lists(edges_knn.tolist(), idx_known.tolist()))
        edges_missing = torch.transpose(edges_missing, 0, 1)
        return edges_missing

    def __call__(self, data: Data) -> Data:
        edge_index_missing = self.create_edges_missing(data)
        edge_index = torch.cat([data.edge_index, edge_index_missing], dim=1)
        data.edge_index = edge_index

        return data