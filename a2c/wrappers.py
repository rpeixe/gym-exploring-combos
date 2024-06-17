import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

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

class ActionMemory(gym.Wrapper):
    class ActionHistory:
        def __init__(self, actions_to_keep):
            self.actions_to_keep = actions_to_keep
            self.reset_history()
        
        def reset_history(self):
            self._deque = deque(np.zeros((self.actions_to_keep, 12), np.float32), self.actions_to_keep)
        
        def add_action(self, action):
            assert action.shape == self._deque[0].shape
            self._deque.appendleft(action)
        
        def get_history(self):
            return np.array(self._deque, np.float32)

    def __init__(self, env, memory):
        gym.utils.RecordConstructorArgs.__init__(self)
        super(ActionMemory, self).__init__(env)

        self.action_history = self.ActionHistory(memory)
        self.observation_space = spaces.Dict({
            'image': env.observation_space,
            'action_history': spaces.Box(low=0., high=1., shape=(memory, 12), dtype=np.float32)
        })

    def reset(self, seed, options):
        observation, info = self.env.reset(seed=seed, options=options)
        self.action_history.reset_history()
        action_history = self.action_history.get_history()
        return {'image': observation, 'action_history': action_history}, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.action_history.add_action(action)
        action_history = self.action_history.get_history()
        return {'image': observation, 'action_history': action_history}, reward, terminated, truncated, info
