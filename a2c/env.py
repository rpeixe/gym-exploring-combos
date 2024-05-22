import retro
import os
from gymnasium.wrappers import TimeLimit, GrayScaleObservation, ResizeObservation, FrameStack
from wrappers import FilterActionWrapper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'custom_integrations')

def make_env(render = True):
        # Create the retro environment for SF Alpha 3
        retro.data.Integrations.add_custom_path(INTEGRATIONS_DIR)
        env = retro.make('StreetFighterAlpha3-GbAdvance', state='Training.RyuVSKen.Wall', scenario='scenario.training.time', inttype=retro.data.Integrations.ALL)
        env.render_mode = 'human' if render else None
        env = TimeLimit(env, 600)

        button_to_exclude = 'SELECT'
        env = FilterActionWrapper(env, button_to_exclude)
        return env