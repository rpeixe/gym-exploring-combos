import numpy as np
import gymnasium as gym

# Wrap the environment with the custom action filter wrapper
class FilterActionWrapper(gym.ActionWrapper):
    def __init__(self, env, button_to_exclude):
        super(FilterActionWrapper, self).__init__(env)
        self.button_to_exclude = button_to_exclude
        self.buttons = self.env.unwrapped.buttons
        self.button_index = self.buttons.index(self.button_to_exclude)
        self.valid_actions = self._create_valid_actions()

        # Update the action space to reflect only the valid actions
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))

    def _create_valid_actions(self):
        valid_actions = []
        for action in range(self.env.action_space.n):
            action_array = self.env.action_space.sample()
            action_array[self.button_index] = 0
            valid_actions.append(action_array)
        return valid_actions

    def action(self, action):
        # Ensure the action does not press the excluded button
        filtered_action = np.copy(self.valid_actions[action])
        return filtered_action