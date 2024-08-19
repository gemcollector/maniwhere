from dm_control.suite import base
import numpy as np

class BaseTask(base.Task):
    def __init__(self, random=None, action_delay=0):
        super().__init__(random)
        self._action_buffer = []
        self._action_delay = action_delay
        self._cur_delay = 0

    def initialize_episode(self, physics):
        physics.reset_pose()
        self._action_buffer = []
        self._cur_delay = 0
        super().initialize_episode(physics)

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        self._action_buffer.append(action)
        if self._cur_delay < self._action_delay:
            self._cur_delay += 1
            self.current_action = physics.data.ctrl  # hold still
        else:
            self.current_action = self._action_buffer[0]
            self._action_buffer = self._action_buffer[1:]
        physics.set_control(self.current_action)

    def randomize(self, min_delay, max_delay):
        self._action_delay = np.random.randint(min_delay, max_delay + 1)

    def _rescale_action(self, physics, base_action=None, target_min=-1.0, target_max=1.0):
        """Rescale difference between current_action and base_action.
            Only for reward computation.
        """
        minimum = self.action_spec(physics).minimum
        maximum = self.action_spec(physics).maximum
        scale = (target_max - target_min) * np.ones_like(self.current_action) / (maximum - minimum)
        if base_action:
            return scale * (self.current_action - base_action)
        else:
            return target_min + scale * (self.current_action - minimum)