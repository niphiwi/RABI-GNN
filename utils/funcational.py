import torch
import torch.nn.functional as F

def selective_dropout(x, known_mask, dropout_prob, training):
    """
    Applies dropout to known nodes in the graph.

    Args:
    - x (Tensor): Node features.
    - known_mask (Tensor): A boolean mask indicating known nodes (True for known, False for unknown).
    - dropout_prob (float): Probability of an element to be zeroed.
    - training (bool): Apply dropout if True, else return original features.

    Returns:
    - Tensor: Node features after applying selective dropout.
    """
    if training:
        # Apply dropout only to the known nodes
        dropout_mask = (torch.rand_like(x[:, 0]) < dropout_prob) & known_mask
        x[dropout_mask] = 0.0
    return x

def create_sensor_proximity_mask(data, threshold):
    """
    Create a mask for missing nodes that are near a known node (sensor) in batched data.

    Args:
    - data (Data): PyTorch Geometric data object with `pos`, `known`, and `batch` attributes.
    - threshold (float): Distance threshold to consider a missing node as influenced by a sensor.

    Returns:
    - torch.Tensor: A mask of shape (num_nodes,) with `1` if the missing node is within
                    the threshold distance of any known node in the same batch, and `0` otherwise.
    """
    # Initialize the mask to zeros
    sensor_proximity_mask = torch.zeros(data.num_nodes, dtype=torch.float, device=data.pos.device)

    # Loop over each graph in the batch
    for batch_i in torch.unique(data.batch):
        batch_mask = (data.batch == batch_i)
        
        # Extract known and missing nodes for this batch
        known_idx = (data.known & batch_mask).nonzero(as_tuple=False).view(-1)
        missing_idx = (~data.known & batch_mask).nonzero(as_tuple=False).view(-1)

        # Calculate the distances between known and missing nodes within this batch
        dist = torch.norm(data.pos[known_idx].unsqueeze(1) - data.pos[missing_idx].unsqueeze(0), p=2, dim=-1)

        # Check if there's at least one known node within the threshold for each missing node
        min_dist, _ = torch.min(dist, dim=0)
        sensor_proximity_mask[missing_idx[min_dist <= threshold]] = 1.0

    return sensor_proximity_mask

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, concentration_weight=3.0, sensor_weight=2.0, threshold=0.6):
        """
        concentration_weight: The weight to put on errors in high concentration regions.
        sensor_weight: The weight to put on errors in proximity to sensors.
        threshold: The concentration threshold above which the concentration_weight is applied.
        """
        super(WeightedMSELoss, self).__init__()
        self.concentration_weight = concentration_weight
        self.sensor_weight = sensor_weight
        self.threshold = threshold

    def forward(self, prediction, target, sensor_mask):
        # Calculate the base mean squared error
        mse_loss = F.mse_loss(prediction, target, reduction='none')
        
        # Apply concentration weight: higher weight for higher concentration areas
        high_concentration_mask = target > self.threshold
        weighted_mse_loss = mse_loss * (high_concentration_mask.float() * self.concentration_weight + 1.0)
        
        # Apply sensor weight: higher weight for nodes close to sensors
        weighted_mse_loss = weighted_mse_loss * (sensor_mask.float() * self.sensor_weight + 1.0).unsqueeze(1)
        
        # You can add more custom behaviors to the loss function here
        
        # Return the mean loss
        return weighted_mse_loss.mean()