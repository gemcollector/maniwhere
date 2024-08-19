import numpy as np
from dm_control.rl.control import specs

from ..randomize.wrapper import RandEnvironment

class DofWrapper(RandEnvironment):
    """A wrapper to change dof of an environment.
    """
    def __init__(self, env: RandEnvironment, dof_setting):
        """Initialize a DofWrapper instance.

        Args:
            env (RandEnvironment): environment to be wrapped
            dof_setting (array-like): specify which dof to be enabled and disabled.
                `None` stand for enable, `number` for disable. For example, for a 6-dof env, 
                dof_setting=[None, None, None, 1, 1, 1.5] can fix last three dof.
        """
        self._env = env
        self._dof_setting = np.asarray(dof_setting)

        wrapped_action_spec = self._env.action_spec()
        
        if wrapped_action_spec.shape != self._dof_setting.shape:
            raise ValueError('Dof setting shape mismatch original action space shape.')

        self._free_action_idxs = [i for i, val in enumerate(self._dof_setting) if val is None]

        action_min = wrapped_action_spec.minimum[self._free_action_idxs]
        action_max = wrapped_action_spec.maximum[self._free_action_idxs]
        self._action_spec = specs.BoundedArray(
            shape=action_min.shape,
            dtype=np.float32,
            minimum=action_min,
            maximum=action_max,
            name='action'
        )

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        target_action = self._dof_setting.copy()
        target_action[self._free_action_idxs] = action
        target_action = target_action.astype(np.float32)
        return self._env.step(target_action)

    def randomize(self):
        self._env.randomize()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)
