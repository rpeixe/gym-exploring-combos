import retro
import os
from gymnasium.wrappers import TimeLimit, GrayScaleObservation, ResizeObservation, FrameStack
from wrappers import FilterAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'custom_integrations')

def make_env(render = True, game_state = 'Training.RyuVSKen.Wall', 
game_scenario = 'scenario.training.time', time_limit = 600):
        # Create the retro environment for SF Alpha 3
        retro.data.Integrations.add_custom_path(INTEGRATIONS_DIR)
        env = retro.make('StreetFighterAlpha3-GbAdvance', state=game_state, scenario=game_scenario, inttype=retro.data.Integrations.ALL)
        env.render_mode = 'human' if render else None
        env = TimeLimit(env, time_limit)

        env = FilterAction(env, ['SELECT'])
        return env