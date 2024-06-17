import retro
import os
from gymnasium.wrappers import TimeLimit, GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from wrappers import FilterAction, ActionMemory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'custom_integrations')

def make_custom_env(
          render = True,
          action_memory = True,
          game_state = 'Training.RyuVSKen.Wall',
          game_scenario = 'scenario.training.time',
          time_limit = 600):
    
    retro.data.Integrations.add_custom_path(INTEGRATIONS_DIR)

    render_mode = 'human' if render else 'rgb_array'
    env = retro.make('StreetFighterAlpha3-GbAdvance',
                     state=game_state,
                     scenario=game_scenario,
                     inttype=retro.data.Integrations.ALL,
                     render_mode=render_mode)
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (80, 120))
    if action_memory:
        env = ActionMemory(env, 1)
    env = TimeLimit(env, time_limit)
    env = FilterAction(env, ['SELECT'])

    return env

def make_custom_vec_env(n_envs = 1, **kwargs):
      vec_env = make_vec_env(
         make_custom_env,
         n_envs=n_envs,
         env_kwargs=kwargs,
         vec_env_cls=SubprocVecEnv)
      vec_env = VecFrameStack(vec_env, 5, {"image": "last", "action_history": "first"})
      vec_env = VecTransposeImage(vec_env)
      return vec_env
