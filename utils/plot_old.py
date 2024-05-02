import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

def plot_graph(graph, max_distance=None, vmin=0, vmax=1):
    """
    Plots a PyTorch Geometric data object.
    If max_distance is provided, the distances between nodes are drawn on the edges.
    """  
    
    # Convert PyTorch Geometric object to NetworkX object
    graph = to_networkx(graph, node_attrs=['x', 'y', 'pos'], edge_attrs=['edge_attr'])

    # Get the coordinates 
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Get the distances (edge labels)
    edge_labels = nx.get_edge_attributes(graph, 'edge_attr')
    if max_distance:
        edge_labels.update((x, int(y*max_distance)) for x, y in edge_labels.items())
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plot = nx.draw(graph, 
                   pos, 
                   with_labels=True, 
                   node_color=list(nx.get_node_attributes(graph, 'y').values()), 
                   cmap="coolwarm",
                   vmin=vmin, vmax=vmax
                  )

    return plot

def plot_directed_graph(graph, max_distance=None, vmin=0, vmax=1):
    """
    Plots a directed graph. Known nodes are colored red, unknown nodes are colored blue.    
    """  
    graph = graph.clone()
    graph.x = graph.x
    graph.y = graph.known
    # Convert PyTorch Geometric object to NetworkX object
    graph = to_networkx(graph, node_attrs=['x', 'y', 'pos'])

    # Get the coordinates 
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Create a directed graph for plotting
    directed_graph = nx.DiGraph(graph)

    plot = nx.draw_networkx(
                    directed_graph, 
                    pos, 
                    with_labels=True, 
                    node_color=list(nx.get_node_attributes(graph, 'y').values()), 
                    cmap="coolwarm",
                  )

    return plot


def plot_graph_with_values(graph, max_val, vmin=0, vmax=25):
    """
    Plots a PyTorch Geometric data object with y values as node labels.
    Input:
        graph 
        max_val -> maximum value of the dataset. used for scaling the y tensor
        vmin -> minimum value of colormap
        vmax -> maximum value of colormap (typically identical to max_val)
    Output:
        plot    
    """  

    # Convert PyTorch Geometric object to NetworkX object
    graph = to_networkx(graph, node_attrs=['x', 'y', 'coords'], edge_attrs=['edge_attr'])

    # Get the coordinates 
    pos = nx.get_node_attributes(graph, 'coords')

    # Get the y values
    y = nx.get_node_attributes(graph, 'y')
    for key in y:
        y[key] = round(y[key] * max_val,1)

    plot = nx.draw(graph, 
                   pos, 
                   with_labels=True, 
                   node_color=list(y.values()), 
                   cmap="coolwarm",
                   vmin=vmin, vmax=vmax,
                   labels=y,
                   node_size=1000,
                  )
    
    return plot


# class GridDataCreator():
#     """
#     Maps sparse data from a 1D array onto a 2D map.
#     """
#     def __init__(self, cell_size):
#         self.cell_size = cell_size

#     def _create_axes(self, coords):
#         """Create the x and y axes of of the grids"""
#         cell_size = self.cell_size

#         min_x = -cell_size
#         min_y = -cell_size

#         max_x = np.around((max(coords[:,0])/cell_size))*cell_size+cell_size
#         max_y = np.around((max(coords[:,1])/cell_size))*cell_size+cell_size

#         nx = int((max_x + cell_size)/cell_size)+1
#         ny = int((max_y + cell_size)/cell_size)+1

#         x = np.linspace(-cell_size, max_x, nx)
#         y = np.linspace(-cell_size, max_y, ny)

#         return x, y
    
#     def map(self, measurements, coords):
#         measurements = measurements.squeeze()
#         x_axis, y_axis = self._create_axes(coords)
        
#         m = torch.zeros(len(y_axis), len(x_axis))

#         for i in range(len(measurements)):
#             # Find the nearest cell center
#             cell_x = np.argmin(abs(x_axis - coords[:,0][i].numpy()))
#             cell_y = np.argmin(abs(y_axis - coords[:,1][i].numpy()))

#             m[int(cell_y), int(cell_x)] = measurements[i]

#         # flip the tensor
#         m = torch.flip(m, [0])
#         return m


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# this is specific to synthetic data
def graph_to_image(data, pos):
    """ Takes graph data with positions as input and returns it as an image. 
    Args:
        data: feature vector
        pos: spatial positions the features (i.e., sensor positions)
    """
    # Get the sensor measurements and positions
    measurements = data
    positions = pos.numpy()

    # Determine the dimensions of the image-like representation
    x_min, y_min = 0, 0 #np.min(positions, axis=0)
    y_max, x_max = 29,24 #np.max(positions, axis=0)
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)

    # Create an empty image-like representation
    image = np.empty((height, width))
    image.fill(np.nan)

    # Create a KDTree for efficient nearest neighbor search
    kdtree = cKDTree(positions)

    # Map each sensor measurement to its corresponding position in the image
    for i, measurement in enumerate(measurements):
        _, idx = kdtree.query(positions[i])
        y, x = positions[i] #- np.array([x_min, y_min])
        image[int(y), int(x)] = measurement

    return image

# This is a generalization of the graph_to_image function
def map_graph_to_image(x, pos, cell_size):
    """ Takes graph data with positions as input and returns it as an image. 
    Args:
        data: feature vector to be mapped
        pos: spatial positions the features (i.e., sensor positions)
        cell_size: size of the grid cells
    """
    # Determine the dimensions of the image-like representation
    x_min, y_min = 0,0
    [y_max, x_max], _ = torch.max(pos, axis=0)
    width = round(int(x_max - x_min + 2*cell_size)/cell_size)
    height = round(int(y_max - y_min + 2*cell_size)/cell_size)

    # Create an empty image-like representation
    image = np.empty((height, width))
    image.fill(np.nan)

    # Create a KDTree for efficient nearest neighbor search
    kdtree = cKDTree(pos.numpy())

    # Map each sensor measurement to its corresponding position in the image
    for i, measurement in enumerate(x):
        _, idx = kdtree.query(pos[i])
        y, x = (pos[i] - torch.tensor([x_min, y_min])) / cell_size
        image[int(y), int(x)] = measurement

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = plt.imshow(np.rot90(image, k=1))
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticks()*cell_size)
    ax.set_yticks(ax.get_yticks(), ax.get_yticks()*cell_size)

    return ax


def graph_to_2d_array(X, pos, cell_size):
    """ Takes graph data with positions as input and returns it as an 2d array 
    Args:
        X: feature vector to be mapped
        pos: spatial positions the features (i.e., sensor positions)
        cell_size: size of the grid cells
    """

    # Define the range boundaries
    x_min, x_max = pos[:,0].min() - cell_size, pos[:,0].max() + cell_size  # optional: subtract/add cell_size
    y_min, y_max = pos[:,1].min() - cell_size, pos[:,1].max() + cell_size

    # Calculate the number of cells in each dimension
    x_cells = int((x_max - x_min) / cell_size) + 1 # +1 to adjust indexing starting at zero
    y_cells = int((y_max - y_min) / cell_size) + 1

    # Create an empty grid as a PyTorch tensor
    grid = torch.zeros(x_cells, y_cells)

    i = 0
    # Iterate through the points and update the cell counts
    for x, y in pos:
        # Calculate the cell indices for the current point
        x_index = int((x - x_min)/ cell_size)
        y_index = int((y - y_min)/ cell_size)
        
        # Increment the count in the corresponding cell
        grid[x_index, y_index] = X[i]
        i += 1

    return grid


def map_graph_to_imagev2(X, pos, cell_size, grid_shape=None, ax=None, rot_k=0, vmin=None, vmax=None):
    """ Takes graph data with positions as input and returns it as an image. 
    Args:
        X: feature vector to be mapped
        pos: spatial positions the features (i.e., sensor positions)
        cell_size: size of the grid cells
    """
    
    if grid_shape == None:
        # Define the range boundaries
        x_min, x_max = pos[:,0].min() - cell_size, pos[:,0].max() + cell_size  # optional: subtract/add cell_size
        y_min, y_max = pos[:,1].min() - cell_size, pos[:,1].max() + cell_size

        # Calculate the number of cells in each dimension
        x_cells = int((x_max - x_min) / cell_size) + 1 # +1 to adjust indexing starting at zero
        y_cells = int((y_max - y_min) / cell_size) + 1

        # Create an empty grid as a PyTorch tensor
        grid = torch.zeros(x_cells, y_cells)
    else:
        grid = torch.zeros(grid_shape)
        x_min = 0
        y_min = 0
    grid.fill_(float('nan'))

    i = 0
    # Iterate through the points and update the cell counts
    for x, y in pos:
        # Calculate the cell indices for the current point
        x_index = int((x - x_min)/ cell_size)
        y_index = int((y - y_min)/ cell_size)
        
        # Increment the count in the corresponding cell
        grid[x_index, y_index] = X[i]
        i += 1

    if ax is None:
        # If ax is not provided, create a new subplot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Plot the grid on the specified ax
    im = ax.imshow(torch.rot90(grid, k=rot_k), vmin=vmin, vmax=vmax)

    return ax


def plot_input_pred_truth(data, pred, pred_pos, cell_size):
    """ Optimized for synthetic dataset. """
    fig, axs = plt.subplots(1, 3)    

    # Plot input
    map_graph_to_imagev2(data.x[data.known][:,-1], data.orig_pos[data.known], cell_size=cell_size, grid_shape=[30,25], ax=axs[0], vmin=0, vmax=1)
    axs[0].set_title('Known Sensors')
    axs[0].axis('off')

    # Plot prediction
    map_graph_to_imagev2(pred[:,-1], pred_pos, cell_size=cell_size, grid_shape=[30,25], ax=axs[1], vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    # Plot ground_truth
    axs[2].imshow(data.ground_truth[:,-1].view([30,25]), vmin=0, vmax=1)
    axs[2].set_title('Ground Truth')
    axs[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()


# def plot_input_pred_truth(graph, pred, pred_pos, cell_size):
#     """ Optimized for synthetic dataset. """
#     fig, axs = plt.subplots(1, 3)

#     # Plot pred
#     axs[0].imshow(graph_to_2d_array(graph.x[:,-1], graph.pos, cell_size), vmin=0, vmax=1)
#     axs[0].set_title('Masked Sensor Input')
#     axs[0].axis('off')
    
#     axs[1].imshow(graph_to_2d_array(pred[:,-1], pred_pos, cell_size), vmin=0, vmax=1)
#     axs[1].set_title('Prediction')
#     axs[1].axis('off')
    
#     # Plot ground_truth
#     axs[2].imshow(graph.ground_truth[:,-1].view([30,25]), vmin=0, vmax=1)
#     axs[2].set_title('Ground Truth')
#     axs[2].axis('off')

#     # Adjust spacing between subplots
#     plt.tight_layout()

#     # Display the plots
#     plt.show()
    

def plot_graphs(graph, pred, pred_pos):
    fig, axs = plt.subplots(1, 3)

    # Plot pred
    axs[0].imshow(graph_to_image(graph.x[:,-1], graph.pos), vmin=0, vmax=1)
    axs[0].set_title('Masked Sensor Input')
    axs[0].axis('off')
    
    axs[1].imshow(graph_to_image(pred[:,-1], pred_pos), vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    
    # Plot ground_truth
    axs[2].imshow(graph.ground_truth[:,-1].view([30,25]), vmin=0, vmax=1)
    axs[2].set_title('Ground Truth')
    axs[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()