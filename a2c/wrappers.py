import numpy as np
import gymnasium as gym

class FilterAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, buttons_to_exclude):
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self.button_filter = np.ones(12, np.float32)
        buttons = self.env.unwrapped.buttons

        for button_to_exclude in buttons_to_exclude:
            button_index = buttons.index(button_to_exclude)
            self.button_filter[button_index] = 0.

    def action(self, action):
        return action * self.button_filter