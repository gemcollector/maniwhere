import numpy as np

class RandomCamera:
    """Class to control the the camera settings.
    """
    def __init__(self, initial_point, circle_point, ):
        pass
    
    def randomize(self):
        angles = np.random.uniform(low=self.actual_min,
                                   high=self.actual_max)
        initial_fai = np.arctan2(
            self.initial_point[2] - self.circle_center[2],
            self.initial_point[0] - self.circle_center[0],
        )
        r = np.linalg.norm(self.initial_point - self.circle_center)
        rxy = np.cos(np.deg2rad(angles[1]) + initial_fai) * r
        delta_x = np.cos(np.deg2rad(angles[0])) * rxy
        delta_y = np.sin(np.deg2rad(angles[0])) * rxy
        delta_z = np.sin(np.deg2rad(angles[1]) + initial_fai) * r
        self.cur_val = np.array([
            self.circle_center[0] + delta_x,
            self.circle_center[1] + delta_y,
            self.circle_center[2] + delta_z,
        ])