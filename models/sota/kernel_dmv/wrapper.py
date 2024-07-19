from models.sota.kernel_dmv.td_kernel_dmvw import TDKernelDMVW
import torch

class KernelDMV:
    def __init__(self, x_range, y_range, cell_size, kernel_size=None, evaluation_radius=None):
        self.x_range = x_range
        self.y_range = y_range
        self.cell_size = cell_size
        if kernel_size:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = 2.5*cell_size

        if evaluation_radius:
            self.evaluation_radius = evaluation_radius
        else:
            self.evaluation_radius = 5*self.kernel_size
        
        self.wind_scale = 0
        self.time_scale = 0
        self.mean = 0
        self.std = 1

        self.kernel = TDKernelDMVW(self.x_range[0] , self.y_range[0], 
                              self.x_range[1] , self.y_range[1], 
                              self.cell_size,
                              self.kernel_size,
                              self.wind_scale,
                              self.time_scale,
                              confidence_scale=1,
                              real_time=False,
                              low_confidence_calculation_zero=True,
                              evaluation_radius=self.evaluation_radius)

    def set_normalization_params(self, mean, std):
        self.mean = mean
        self.std = std

    def set_measurements(self, sensor_positions, measurements):
        x = sensor_positions[:,0].numpy()
        y = sensor_positions[:,1].numpy()
        concentration_normalized = measurements.numpy()
        concentration = concentration_normalized * self.std + self.mean

        num_measurements = len(sensor_positions)
        wind_speed = torch.zeros(num_measurements).numpy()
        wind_direction = torch.zeros(num_measurements).numpy()
        timestamp = range(num_measurements)
        
        self.kernel.set_measurements(x=x, y=y, 
                                     concentration=concentration, 
                                     timestamp=timestamp, 
                                     wind_speed=wind_speed,
                                     wind_direction=wind_direction)

    def set_measurements_old(self, data):
        """ Deprecated, old behavior. """
        x = data.orig_pos[:,0].numpy()
        y = data.orig_pos[:,1].numpy()
        concentration_normalized = data.y[:,-1].numpy()
        concentration = concentration_normalized * self.std + self.mean

        wind_speed = torch.zeros(len(data.orig_pos)).numpy()
        wind_direction = torch.zeros(len(data.orig_pos)).numpy()
        timestamp = range(len(data.orig_pos))
        
        self.kernel.set_measurements(x=x, y=y, 
                                     concentration=concentration, 
                                     timestamp=timestamp, 
                                     wind_speed=wind_speed,
                                     wind_direction=wind_direction)

    def predict(self):
        """ Calculate the KDM based on specified positions and return mean map."""
        self.kernel.calculate_maps()
        
        normalized = (self.kernel.mean_map - self.mean)/self.std

        return torch.tensor(normalized)    