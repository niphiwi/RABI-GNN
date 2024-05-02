import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import random

def add_virtual_nodes_synth(data, total_sampled):
    """Add <total_sampled> amount of virtual sensors. Only used for synthetic data."""
    # Size of output
    height = 30
    width = 25

    # Create positions using meshgrid
    rows = torch.arange(height)
    cols = torch.arange(width)
    pos_y, pos_x = torch.meshgrid(rows, cols,  indexing="ij")

    # Reshape positions into a single tensor
    pos = torch.stack((pos_y.flatten(), pos_x.flatten()), dim=1).float()

    # Check which positions are not sampled yet
    sampled = data.pos.int()
    sampled_set = set(map(tuple, sampled.numpy()))
    pos_set = set(map(tuple, pos.numpy()))
    unsampled_set = pos_set - sampled_set
    unsampled = torch.tensor(list(unsampled_set))

    # Randomly sample a subset of unsampled nodes
    #unsampled = unsampled[random.sample(range(len(unsampled)), num_unsampled-len(sampled))]
    unsampled = torch.tensor(random.sample(unsampled.tolist(), k=total_sampled-len(sampled)))

    # Create features
    x = torch.cat((data.y, torch.zeros(len(unsampled), data.y.shape[1])), dim=0)
    mask = torch.cat((torch.ones(len(data.y), dtype=torch.bool), torch.zeros(len(unsampled), dtype=torch.bool)), dim=0)
    pos = torch.cat([sampled, unsampled])

    transform = T.Compose([
                #AddMask(.1),
                #ApplyMask(),
            ])

    return transform(Data(x=x, known=mask, pos=pos))

def sort_positions(data):
    # Combine both columns into a single number (e.g., 20.006, 23.004) for lexicographic sort
    combined = data.pos[:, 0] * 1e3 + data.pos[:, 1]
    sorted_indices = torch.argsort(combined)

    # Now, sort pos and pos2 using these indices
    data.pos = data.pos.index_select(0, sorted_indices)
    data.orig_pos = data.orig_pos.index_select(0, sorted_indices)
    data.x = data.x.index_select(0, sorted_indices)
    data.known = data.known.index_select(0, sorted_indices)

    return data

# def add_virtual_nodes(data, pos, cell_size, n_virtual, x_range=None, y_range=None):
#     """ Add virtual nodes to the graph. Nodes are added in-between the existing nodes
#     based on a discretized grid. Cell_size is governing the size of the discretized grid cells. """

#     # Define the range boundaries
#     if x_range == None:
#         x_min, x_max = pos[:,0].min(), pos[:,0].max()  # optional: subtract/add cell_size
#     else: 
#         x_min, x_max = x_range
#     if y_range == None:
#         y_min, y_max = pos[:,1].min(), pos[:,1].max()
#     else:
#         y_min, y_max = y_range

#     # Calculate the number of cells in each dimension
#     x_cells = int((x_max - x_min) / cell_size) + 1 # +1 to adjust indexing starting at zero
#     y_cells = int((y_max - y_min) / cell_size) + 1

#     # Create an empty grid as a PyTorch tensor
#     grid = torch.zeros(x_cells, y_cells, dtype=torch.int)

#     # Iterate through the points and update the cell counts
#     for x, y in pos:
#         # Calculate the cell indices for the current point
#         x_index = int((x - x_min)/ cell_size)
#         y_index = int((y - y_min)/ cell_size)
        
#         # Increment the count in the corresponding cell
#         grid[x_index, y_index] = 1

#     # Find cells with zero points
#     empty_cells = torch.nonzero(grid == 0)
#     pos_x_unknown = empty_cells[:,0] * cell_size + x_min
#     pos_y_unknown = empty_cells[:,1] * cell_size + y_min
#     pos_unknown = torch.stack([pos_x_unknown, pos_y_unknown], dim=1)

#     if n_virtual != None:
#         # randomly drop rows from pos_unknown so that len(pos_unknown) = n_virtual
#         indices_to_drop = random.sample(range(len(pos_unknown)), len(pos_unknown)-n_virtual)
#         filtered_pos_unknown = torch.tensor([row.tolist() for i, row in enumerate(pos_unknown) if i not in indices_to_drop])
#         pos_unknown = filtered_pos_unknown

#     # Build known tensor
#     seq_len = data.x.shape[1]
#     known = torch.cat([torch.ones([data.num_nodes, seq_len], dtype=torch.bool), 
#                        torch.zeros([pos_unknown.shape[0], seq_len], dtype=torch.bool)])

#     # Build feature tensor & replace nan (from missing robot measurements) with 0, as robot pos is now encoded in known tensor.
#     x = torch.nan_to_num(torch.cat([data.y,torch.zeros([pos_unknown.shape[0],10])]))

#     upscaled_data = Data(
#         x=x, 
#         pos=torch.cat([pos, pos_unknown]),
#         orig_pos=torch.cat([pos, pos_unknown]),
#         known=known,
#         )
    
#     return sort_positions(upscaled_data)

def add_virtual_nodes(data, cell_size, radius: float=0.3, x_range=None, y_range=None):
    seq_len = data.x.shape[1]
    # ~~~~~~~~~~~~
    # CREATE GRID
    # ~~~~~~~~~~~~
    # Define the range boundaries
    if x_range == None:
        x_min, x_max = data.pos[:,0].min(), data.pos[:,0].max()  # optional: subtract/add cell_size
    else: 
        x_min, x_max = x_range
    if y_range == None:
        y_min, y_max = data.pos[:,1].min(), data.pos[:,1].max()
    else:
        y_min, y_max = y_range

    # Calculate the number of cells in each dimension
    x_cells = int((x_max - x_min) / cell_size) + 1 # +1 to adjust indexing starting at zero
    y_cells = int((y_max - y_min) / cell_size) + 1

    # Create an empty grid as a PyTorch tensor
    grid = torch.full((x_cells, y_cells, seq_len), float('nan'))
    grid_id = torch.full((x_cells, y_cells), -1)
    # ~~~~~~~~~~~~
    # MAP data.y TO GRID
    # ~~~~~~~~~~~~

    # Assuming data.orig_pos correctly aligns with data.y
    for i, pos in enumerate(data.orig_pos):
        # Use torch.round() for proper rounding and convert to integer indices
        x, y = torch.round(pos[:2]/cell_size).int()  # Assuming the first two elements are x and y positions
        # Ensure the indices are within the grid bounds
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            # Assign the 10-dimensional measurement vector to the grid cell
            grid[x, y] = data.y[i]
            grid_id[x, y] = data.id[i]


    # ~~~~~~~~~~~~
    # GET POS FROM GRID
    # ~~~~~~~~~~~~
    rows = torch.arange(grid.shape[0])
    cols = torch.arange(grid.shape[1])
    cols_indices, rows_indices = torch.meshgrid(cols, rows, indexing='xy')
    pos = torch.stack((rows_indices.flatten(), cols_indices.flatten()), dim=1) * cell_size + torch.tensor([x_min, y_min]).float()

    # ~~~~~~~~~~~~
    # GET KNOWN TENSOR FROM GRID
    # ~~~~~~~~~~~~
    known = ~torch.isnan(grid).reshape(-1, seq_len)
    ids = grid_id.reshape(-1)

    # Change nans to zeros
    x = torch.nan_to_num(grid.reshape(-1, seq_len))

    # ~~~~~~~~~~~~
    # BUILD GRAPH
    # ~~~~~~~~~~~~

    upscaled_data = Data(
        x=x, 
        pos=pos,
        id=ids,
        orig_pos=pos,
        known=known,
        )
    
    upscaled_data = T.NormalizeScale()(upscaled_data)
    upscaled_data = T.RadiusGraph(r=radius, loop=True, max_num_neighbors=200)(upscaled_data)
    upscaled_data = T.Distance(norm=False)(upscaled_data)
    upscaled_data = T.Cartesian()(upscaled_data)
    
    return upscaled_data