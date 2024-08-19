import numpy as np
import math

class RandAttr(object):
    """Class for a env parameter to randomize.
    """
    def __init__(self, name, default_val, min_val=None, max_val=None, discrete_vals=None, 
                 mode='uniform', initial_point=None, circle_center=None, schedule=True):
        """Initialize a new random attribute, either provide all discrete possible values(`descrete_vals`),
        or provide a continuous range(`min_val`, `max_val`). `default_val` is always needed.
        """

        assert mode in ['uniform', 'circle']

        if discrete_vals is None and (min_val is None or max_val is None):
            raise ValueError("Either discrete_vals or [min_val, max_val] is needed.")
        
        self.name = name
        self.mode = mode
        self.schedule = schedule
        self.default_val = np.array(default_val)
        if mode == 'circle':
            self.initial_point = np.array(initial_point)
            self.circle_center = np.array(circle_center)
        if discrete_vals is None:
            self.discrete_vals = None
            self.min_val = np.array(min_val)
            self.max_val = np.array(max_val)
        else:
            self.discrete_vals = np.array(discrete_vals)
            self.min_val = self.max_val = None
        
        # TODO: unify all array or scalar!
        self.actual_max = self.actual_min = self.default_val  # start at no randomization
        self.reset()

    def randomize(self):
        if self.mode == 'uniform':
            if self.discrete_vals is not None:
                self.cur_val = np.random.choice(self.discrete_vals)
            else:
                self.cur_val = np.random.uniform(low=self.actual_min,
                                                high=self.actual_max)
        elif self.mode == 'circle':
            rand_vals = np.random.uniform(low=self.actual_min,
                                       high=self.actual_max)
            angles, dist_ceof = rand_vals[:2], rand_vals[2]
            initial_theta = np.arctan2(
                self.initial_point[1] - self.circle_center[1],
                self.initial_point[0] - self.circle_center[0],
            )
            initial_fai = np.arctan2(
                self.initial_point[2] - self.circle_center[2],
                np.linalg.norm(self.initial_point[:2] - self.circle_center[:2]),
            )
            r = np.linalg.norm(self.initial_point - self.circle_center) * dist_ceof
            rxy = np.cos(np.deg2rad(angles[1]) + initial_fai) * r
            # rxy = np.linalg.norm(self.initial_point[:2] - self.circle_center[:2])
            delta_x = np.cos(np.deg2rad(angles[0]) + initial_theta) * rxy
            delta_y = np.sin(np.deg2rad(angles[0]) + initial_theta) * rxy
            delta_z = np.sin(np.deg2rad(angles[1]) + initial_fai) * r
            self.cur_val = np.array([
                self.circle_center[0] + delta_x,
                self.circle_center[1] + delta_y,
                self.circle_center[2] + delta_z,
            ])
        return self.cur_val

    def reset(self):
        if self.mode == 'uniform':
            self.cur_val = self.default_val
        elif self.mode == 'circle':
            self.cur_val = self.initial_point
        return self.cur_val
    
    def to_string(self):
        """deprecated
        """
        if isinstance(self.cur_val, np.ndarray):
            return ' '.join('{:.6f}'.format(val) for val in self.cur_val)
        else:
            return '{:.6f}'.format(self.cur_val)

class _Scheduler(object):
    def __init__(self, rand_attrs):
        self._rand_attrs = [attr for rand_attr in rand_attrs 
                            for attr in (rand_attr if isinstance(rand_attr, list) else [rand_attr])]

    def step(self):
        """Adjust DR range. Should be called per episode.
        """
        raise NotImplementedError

class ConstantScheduler(_Scheduler):
    """A scheduler that keeps the DR range a constant ratio of the max bound.
        The range follows formula: bound = max_bound * ratio 
    """
    def __init__(self, rand_attrs, gamma):
        super().__init__(rand_attrs)
        self._gamma = gamma
    
    def step(self):
        for rand_attr in self._rand_attrs:
            if rand_attr.schedule:
                rand_attr.actual_min = rand_attr.default_val * (1.0 - self._gamma) + rand_attr.min_val * self._gamma
                rand_attr.actual_max = rand_attr.default_val * (1.0 - self._gamma) + rand_attr.max_val * self._gamma
            else:
                rand_attr.actual_min, rand_attr.actual_max = rand_attr.min_val, rand_attr.max_val
                
class ExpScheduler(_Scheduler):
    """A scheduler that gradually increase DR attribute range.
        The range follows formula: bound = max_bound * (1.0 - gamma ^ x) 
    """
    def __init__(self, rand_attrs, gamma, latency=0):
        super().__init__(rand_attrs)
        self._gamma = gamma
        self._gamma_exp = 1.0
        self._latency = latency

    def step(self):
        if self._latency > 0:
            self._latency -= 1
        else:
            self._gamma_exp *= self._gamma
        
        for rand_attr in self._rand_attrs:
            if rand_attr.schedule:
                rand_attr.actual_min = rand_attr.default_val * self._gamma_exp + rand_attr.min_val * (1.0 - self._gamma_exp)
                rand_attr.actual_max = rand_attr.default_val * self._gamma_exp + rand_attr.max_val * (1.0 - self._gamma_exp)
            else:
                rand_attr.actual_min, rand_attr.actual_max = rand_attr.min_val, rand_attr.max_val
            
class SigmoidScheduler(_Scheduler):
    """A scheduler that gradually increase DR attribute range.
        The range follows formula: bound = max_bound * (1.0 / (1 + e^(-gamma * x))) 
    """
    def __init__(self, rand_attrs, gamma, latency=0):
        super().__init__(rand_attrs)
        self._gamma = gamma
        self._latency = latency
        self._step = 0

    def step(self):
        coff = 1.0 / (1 + np.pow(math.e, -self._gamma * (self._step - self._latency)))
        self._step += 1
        
        for rand_attr in self._rand_attrs:
            if rand_attr.schedule:
                rand_attr.actual_min = rand_attr.default_val * (1.0 - coff) + rand_attr.min_val * coff
                rand_attr.actual_max = rand_attr.default_val * (1.0 - coff) + rand_attr.max_val * coff
            else:
                rand_attr.actual_min, rand_attr.actual_max = rand_attr.min_val, rand_attr.max_val