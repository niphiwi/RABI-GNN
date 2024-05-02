import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class CustomGATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super(CustomGATv2Conv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # Linear transformation
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # Learnable parameters for attention mechanism
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, known_mask):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        x = self.lin(x)

        # Start propagating messages
        return self.propagate(edge_index, x=x, known_mask=known_mask)

    def message(self, edge_index_i, x_i, x_j, size_i, known_mask):
        # Compute attention coefficients
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        
        alpha = (x_i * self.att_src + x_j * self.att_dst).sum(dim=-1)

        # Integrate known/unknown mask information into attention scores
        known_mask_i = known_mask[edge_index_i].unsqueeze(-1)  # Source node mask
        alpha = alpha * known_mask_i  # Modulate attention with known status

        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Dropout for attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out
