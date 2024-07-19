import torch

def set_random_node_to_unknown(data):
    # Set all sensors as known 
    data.x = data.y

    # Set a random sensor to unknown -> this sensor shall be predicted by the others
    n_nodes = data.known.shape[0]
    data.known = torch.ones(data.known.shape, dtype=torch.bool)
    random_index = torch.randint(0, n_nodes, (1,))
    data.known[random_index,:] = False
    data.x = data.y * data.known

    return data

def set_node_to_unknown(data, idx=None):
    # Set all sensors as known 
    data.x = data.y

    # Set a random sensor to unknown -> this sensor shall be predicted by the others
    n_nodes = data.known.shape[0]
    data.known = torch.ones(data.known.shape, dtype=torch.bool)
    if idx==None:
        idx = torch.randint(0, n_nodes, (1,))
    data.known[idx,:] = False
    data.x = data.y * data.known

    return data


def remove_node(data, node_to_remove):
    data = data.clone()

    # Remove node features
    data.x = torch.cat([data.x[:node_to_remove], data.x[node_to_remove+1:]])
    data.y = torch.cat([data.y[:node_to_remove], data.y[node_to_remove+1:]])
    data.pos = torch.cat([data.pos[:node_to_remove], data.pos[node_to_remove+1:]])
    data.id = torch.cat([data.id[:node_to_remove], data.id[node_to_remove+1:]])
    data.orig_pos = torch.cat([data.orig_pos[:node_to_remove], data.orig_pos[node_to_remove+1:]])
    data.known = torch.cat([data.known[:node_to_remove], data.known[node_to_remove+1:]])

    # Adjust edge indices
    edge_index = data.edge_index
    mask = (edge_index[0] != node_to_remove) & (edge_index[1] != node_to_remove)
    edge_index = edge_index[:, mask]

    # Adjust edge indices to account for the removed node
    edge_index[0][edge_index[0] > node_to_remove] -= 1
    edge_index[1][edge_index[1] > node_to_remove] -= 1
    data.edge_index = edge_index

    return data


def calc_mse_of_dropped(pred, data_original, dropped_id):
    # Find the index of the dropped node in the node list format
    dropped_idx = (data_original.id == dropped_id).nonzero()
    # Calculate MSE
    mse = torch.mean((data_original.x[dropped_idx] - pred[dropped_idx]) ** 2)
    return mse


def calc_abs_diff_of_dropped(pred, data_original, dropped_id):
    # Find the index of the dropped node in the node list format
    dropped_idx = (data_original.id == dropped_id).nonzero().squeeze(dim=1)   ## added .squeeze(dim=1)
    # Calculate MSE
    abs_diff = abs(data_original.x[:,-1][dropped_idx] - pred[dropped_idx])
    return abs_diff
