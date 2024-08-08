import retro
import os
from gymnasium.wrappers import TimeLimit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'custom_integrations')

def make_custom_env(
          render = True,
          game_state = 'Training.RyuVSKen.Wall.CustomCombo',
          game_scenario = 'scenario.training.time',
          time_limit = 600):
    
    retro.data.Integrations.add_custom_path(INTEGRATIONS_DIR)

    render_mode = 'human' if render else 'rgb_array'
    env = retro.make('StreetFighterAlpha3-GbAdvance',
                     state=game_state,
                     scenario=game_scenario,
                     inttype=retro.data.Integrations.ALL,
                     render_mode=render_mode)
    
    env = TimeLimit(env, time_limit)

    return env
