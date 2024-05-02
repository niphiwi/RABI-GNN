import torch
import matplotlib.pyplot as plt

# def visualize_feature_as_image(X, pos, cell_size, grid_shape=None, ax=None, rot_k=0, vmin=None, vmax=None):
#     """ Takes graph data with positions as input and returns it as an image. 
#     Args:
#         X: feature vector to be mapped
#         pos: spatial positions the features (i.e., sensor positions)
#         cell_size: size of the grid cells
#     """
    
#     if grid_shape == None:
#         # Define the range boundaries
#         x_min, x_max = pos[:,0].min() - cell_size, pos[:,0].max() + cell_size  # optional: subtract/add cell_size
#         y_min, y_max = pos[:,1].min() - cell_size, pos[:,1].max() + cell_size

#         # Calculate the number of cells in each dimension
#         x_cells = int((x_max - x_min) / cell_size) + 1 # +1 to adjust indexing starting at zero
#         y_cells = int((y_max - y_min) / cell_size) + 1

#         # Create an empty grid as a PyTorch tensor
#         grid = torch.zeros(x_cells, y_cells)
#     else:
#         grid = torch.zeros(grid_shape)
#         x_min = 0
#         y_min = 0
#     grid.fill_(float('nan'))

#     i = 0
#     # Iterate through the points and update the cell counts
#     for x, y in pos:
#         # Calculate the cell indices for the current point
#         x_index = int((x - x_min)/ cell_size)
#         y_index = int((y - y_min)/ cell_size)
        
#         # Increment the count in the corresponding cell
#         grid[x_index, y_index] = X[i]
#         i += 1

#     if ax is None:
#         # If ax is not provided, create a new subplot
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)

#     # Plot the grid on the specified ax
#     im = ax.imshow(torch.rot90(grid, k=rot_k), vmin=vmin, vmax=vmax)

#     return ax

def visualize_feature_as_image(X, pos, cell_size=1, grid_shape=None, ax=None, rot_k=1, vmin=None, vmax=None):
    """Visualizes a specific feature dimension as an image based on spatial positions.
    
    Args:
        X: Feature vector to be mapped (last dimension selected).
        pos: Spatial positions of the features.
        cell_size: Size of the grid cells.
        grid_shape: Optional shape of the output grid. If None, it's calculated.
        ax: Matplotlib axis to plot on. If None, a new figure is created.
        rot_k: Number of times the image is rotated by 90 degrees.
        vmin, vmax: Colorbar range.
    """
    # Define the range boundaries
    if grid_shape is None:
        x_min, x_max = pos[:,0].min() - cell_size, pos[:,0].max() + cell_size
        y_min, y_max = pos[:,1].min() - cell_size, pos[:,1].max() + cell_size
        x_cells = int((x_max - x_min) / cell_size) + 1
        y_cells = int((y_max - y_min) / cell_size) + 1
        grid_sum = torch.full((x_cells, y_cells), 0.0)
        grid_count = torch.full((x_cells, y_cells), 0)
    else:
        grid_sum = torch.full(grid_shape, 0.0)
        grid_count = torch.full(grid_shape, 0)
        x_min = 0
        y_min = 0

    # Map features to the grid
    for (x, y), value in zip(pos, X):
        x_index = int((x - x_min) / cell_size)
        y_index = int((y - y_min) / cell_size)
        if 0 <= x_index < grid_sum.shape[0] and 0 <= y_index < grid_sum.shape[1]:
            grid_sum[x_index, y_index] += value
            grid_count[x_index, y_index] += 1

    # Calculate averages where counts are non-zero
    grid_avg = torch.where(grid_count > 0, grid_sum / grid_count, torch.full_like(grid_sum, float('nan')))

    # Create a plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the grid
    im = ax.imshow(torch.rot90(grid_avg, k=rot_k), vmin=vmin, vmax=vmax, cmap='viridis')
    #plt.colorbar(im, ax=ax)  # Optionally add a colorbar
    #ax.set_title("Feature Visualization")

    return ax

def plot_data(input_data, pred, pred_pos, cell_size, file_prefix=None, ground_truth=None, rot_k=0, scale=1):
    """
    Plot input, prediction, and optionally ground truth. Save to file if file_prefix is provided.
    
    Args:
    - input_data: The input data (graph) to be plotted.
    - pred: The prediction data to be plotted.
    - pred_pos: The positions associated with the prediction data.
    - cell_size: The size of each cell in the plot.
    - file_prefix: (Optional) Prefix for the filename if the plots are to be saved.
    - ground_truth: (Optional) The ground truth data to be plotted.
    - rot_k: (Optional) How many 90° turns the plotted images shall be rotated.
    - scale: (Optinonal) Scaling factor of matplotlib subplots.

    The function dynamically adjusts based on the presence of ground_truth.
    """

    num_plots = 2 if ground_truth is None else 3
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 2 * scale, 3 * scale))

    # Function to handle subplot creation and saving
    def create_subplot(data, pos, ax_index, title, save_suffix=None, is_ground_truth=False):
        if is_ground_truth:
            axs[ax_index].imshow(data.view([30, 25]), vmin=None, vmax=None)
        else:
            visualize_feature_as_image(data, pos, cell_size=cell_size, ax=axs[ax_index], rot_k=rot_k, vmin=None, vmax=None)
        axs[ax_index].set_title(title)
        axs[ax_index].axis('off')
        if file_prefix is not None and save_suffix is not None:
            fig.savefig(f'{file_prefix}_{save_suffix}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)


    # Plot input (masked by known data)
    masked_input = input_data.x[input_data.known[:, -1]]
    masked_pos = input_data.orig_pos[input_data.known[:, -1]]
    create_subplot(masked_input[:,-1], masked_pos, 0, 'Known Sensors', 'input' if file_prefix else None)
    
    # Plot prediction
    create_subplot(pred[:,-1], pred_pos, 1, 'Prediction', 'pred' if file_prefix else None)

    if ground_truth is not None:
        # Plot ground truth if available
        create_subplot(ground_truth[:, -1], pred_pos, 2, 'Ground Truth', 'truth' if file_prefix else None, is_ground_truth=True)


    plt.tight_layout()
    if file_prefix is None:
        plt.show()
    plt.close(fig)

# def plot_input_pred_truth(data, pred, pred_pos, cell_size, file_prefix=None):
#     """ Optimized for synthetic dataset. Save to file, if file_prefix is provided. """

#     if file_prefix is not None:
#         # Plot input
#         fig, ax = plt.subplots()
#         visualize_feature_as_image(data.x[data.known[:,-1]][:,-1], data.orig_pos[data.known[:,-1]], cell_size=cell_size, grid_shape=[30,25], ax=ax, vmin=0, vmax=1)
#         ax.axis('off')
#         plt.tight_layout()
#         fig.savefig(f'{file_prefix}_input.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#         plt.close(fig)

#         # Plot prediction
#         fig, ax = plt.subplots()
#         visualize_feature_as_image(pred[:,-1], pred_pos, cell_size=cell_size, grid_shape=[30,25], ax=ax, vmin=0, vmax=1)
#         ax.axis('off')
#         plt.tight_layout()
#         fig.savefig(f'{file_prefix}_pred.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#         plt.close(fig)

#         # Plot ground_truth
#         fig, ax = plt.subplots()
#         ax.imshow(data.ground_truth[:,-1].view([30,25]), vmin=0, vmax=1)
#         ax.axis('off')
#         plt.tight_layout()
#         fig.savefig(f'{file_prefix}_truth.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#         return

#     fig, axs = plt.subplots(1, 3)    

#     # Plot input
#     visualize_feature_as_image(data.x[data.known[:,-1]][:,-1], data.orig_pos[data.known[:,-1]], cell_size=cell_size, grid_shape=[30,25], ax=axs[0], vmin=0, vmax=1)
#     axs[0].set_title('Known Sensors')
#     axs[0].axis('off')

#     # Plot prediction
#     visualize_feature_as_image(pred[:,-1], pred_pos, cell_size=cell_size, grid_shape=[30,25], ax=axs[1], vmin=0, vmax=1)
#     axs[1].set_title('Prediction')
#     axs[1].axis('off')

#     # Plot ground_truth
#     axs[2].imshow(data.ground_truth[:,-1].view([30,25]), vmin=0, vmax=1)
#     axs[2].set_title('Ground Truth')
#     axs[2].axis('off')

#     # Adjust spacing between subplots
#     plt.tight_layout()


def plot_input_pred(input_data, pred, pred_pos, cell_size, file_prefix=None):
    """ Plot input and prediction. Also works for data, where no ground_truth is provided. """

    if file_prefix is not None:
        # Plot input
        fig, ax = plt.subplots()
        visualize_feature_as_image(input_data.x[:,-1], input_data.orig_pos, cell_size=cell_size, ax=ax, rot_k=1, vmin=0, vmax=1)
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(f'{file_prefix}_input.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Plot prediction
        fig, ax = plt.subplots()
        visualize_feature_as_image(pred[:,-1], pred_pos, cell_size=cell_size, ax=ax, rot_k=1, vmin=0, vmax=1)
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(f'{file_prefix}_pred.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    fig, axs = plt.subplots(1, 2)    

    # Plot input
    visualize_feature_as_image(input_data.y[:,-1], input_data.orig_pos, cell_size=cell_size, ax=axs[0], rot_k=1, vmin=0, vmax=1)
    axs[0].set_title('Sensors')
    axs[0].axis('off')

    # Plot prediction
    visualize_feature_as_image(pred[:,-1], pred_pos, cell_size=cell_size, ax=axs[1], rot_k=1, vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

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
    

def plot_hrm(input_data, pred, pred_pos, cell_size, file_prefix=None, ground_truth=None, scale=1):
    """
    Plot input, prediction, and optionally ground truth. Save to file if file_prefix is provided.
    
    Args:
    - input_data: The torch_geometric data object to be plotted.
    - pred: The prediction feature vector to be plotted.
    - pred_pos: The positions associated with the prediction data.
    - cell_size: The size of each cell in the plot.
    - file_prefix: (Optional) Prefix for the filename if the plots are to be saved.
    - ground_truth: (Optional) The ground truth data to be plotted.
    - rot_k: (Optional) How many 90° turns the plotted images shall be rotated.
    - scale: (Optinonal) Scaling factor of matplotlib subplots.

    The function dynamically adjusts based on the presence of ground_truth.
    """

    num_plots = 2
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 2 * scale, 3 * scale))

    # Function to handle subplot creation and saving
    def create_subplot(data, pos, ax_index, title, save_suffix=None, is_ground_truth=False):
        visualize_feature_as_image(data, pos, cell_size=cell_size, ax=axs[ax_index], vmin=None, vmax=None)
        axs[ax_index].set_title(title)
        axs[ax_index].axis('off')
        if file_prefix is not None and save_suffix is not None:
            fig.savefig(f'{file_prefix}_{save_suffix}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)


    # Plot input (masked by known data)
    masked_input = input_data.x[input_data.known[:,-1]]
    masked_pos = input_data.orig_pos[input_data.known[:,-1]]
    create_subplot(masked_input[:,-1], masked_pos, 0, 'Known Sensors', 'input' if file_prefix else None)
    
    # Plot prediction
    create_subplot(pred[:,-1], pred_pos, 1, 'Prediction', 'pred' if file_prefix else None)

    if ground_truth is not None:
        # Plot ground truth if available
        create_subplot(ground_truth[:, -1], pred_pos, 2, 'Ground Truth', 'truth' if file_prefix else None, is_ground_truth=True)

    plt.tight_layout()
    if file_prefix is None:
        plt.show()
    plt.close(fig)