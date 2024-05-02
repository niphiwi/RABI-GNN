import torch
import torch_geometric.transforms as T

def connect_known_nodes(graph, r):
    """Connect all known nodes with each other (including self-loops), but not the missing nodes."""
    data = graph.clone()

    data = T.RadiusGraph(r=r, loop=True, max_num_neighbors=500)(data)
    edge_index = data.edge_index
    observed_node_mask = data.known

    edge_mask = observed_node_mask[edge_index[0]] & observed_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    
    return edge_index

def connect_k_missing_nodes(graph, k):
    data = graph.clone()

    # data = T.RadiusGraph(r=threshold, loop=True)(data)
    data = T.KNNGraph(k=k, loop=True)(data)
    edge_index = data.edge_index
    missing_node_mask = ~data.known

    edge_mask = missing_node_mask[edge_index[0]] & missing_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    
    return edge_index

def connect_r_missing_nodes(graph, r):
    data = graph.clone()

    data = T.RadiusGraph(r=r, loop=True)(data)
    edge_index = data.edge_index
    missing_node_mask = ~data.known

    edge_mask = missing_node_mask[edge_index[0]] & missing_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    
    return edge_index


def connect_known_to_missing(graph, threshold):
    """Connect known nodes to missing nodes (directed edges, i.e., one direction)."""
    data = graph.clone()
    
    # Check if 'batch' attribute is present in 'data'
    if data.batch != None:
        num_batches = torch.unique(data.batch).size(0)
    else:
        num_batches = 1

    # Initialize an empty tensor to store all edge indices
    concatenated_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)

    # Loop through each batch_i (or treat entire data as a single batch if 'batch' attribute is not present)
    for batch_i in range(num_batches):
        # If 'batch' attribute is available, filter based on batch_i
        if data.batch != None:
            batch_mask = data.batch == batch_i
        else:
            batch_mask = torch.ones(data.num_nodes, dtype=torch.bool)

        # Index of observed and unobserved nodes of this batch
        known_idx = torch.nonzero(data.known * batch_mask).flatten()
        missing_idx = torch.nonzero(~data.known * batch_mask).flatten()

        # Calculate edges with cartesian product
        edge_index = torch.cartesian_prod(known_idx, missing_idx).t()

        # Filter edges based on distance
        (row, col), pos = edge_index, data.pos
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        filtered_mask = (dist <= threshold).squeeze().t()
        filtered_edge_index = edge_index[:, filtered_mask]

        # Concatenate the current edge_index to the existing tensor
        concatenated_edge_index = torch.cat([concatenated_edge_index, filtered_edge_index], dim=1)

    return concatenated_edge_index

def connect_known_to_k_nearest_missing(graph, K):
    """Connect known nodes to K nearest neighbors."""
    data = graph.clone()
    
    if data.batch != None:
        num_batches = torch.unique(data.batch).size(0)
    else:
        num_batches = 1

    concatenated_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)

    for batch_i in range(num_batches):
        if data.batch != None:
            batch_mask = data.batch == batch_i
        else:
            batch_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            
        known_idx = torch.nonzero(data.known * batch_mask).flatten()
        missing_idx = torch.nonzero(~data.known * batch_mask).flatten()

        known_pos = data.pos[known_idx]
        missing_pos = data.pos[missing_idx]

        # Compute the pairwise distance matrix
        dists = torch.cdist(known_pos, missing_pos, p=2)

        # Check if K is too large, adjust if necessary
        max_K = min(missing_pos.size(1), K)
        
        try:
            # Find the indices of the K nearest neighbors
            knn_indices = dists.topk(K, dim=1, largest=False, sorted=True)[1]
        except RuntimeError:
            # If K is too large, use the maximum possible value
            knn_indices = dists.topk(max_K, dim=1, largest=False, sorted=True)[1]
        

        # Create edge indices
        row_indices = known_idx.view(-1, 1).repeat(1, K).flatten()
        col_indices = missing_idx[knn_indices].flatten()

        knn_edge_index = torch.stack([row_indices, col_indices])

        concatenated_edge_index = torch.cat([concatenated_edge_index, knn_edge_index], dim=1)

    return concatenated_edge_index