from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from env import make_env
import argparse
import os

def main(args):
      # Create a vectorized environment
      vec_env = make_vec_env(
           make_env,
           n_envs=args.n_envs,
           env_kwargs={"render": args.render},
           vec_env_cls=SubprocVecEnv)
      vec_env = VecFrameStack(vec_env, 4)
      
      # Create the A2C model with TensorBoard logging enabled
      model = A2C(
         'CnnPolicy',
         vec_env,
         learning_rate=0.0007,
         n_steps=5,
         gamma=0.99,
         ent_coef=0.01,
         vf_coef=0.5,
         max_grad_norm=0.5,
         verbose=args.verbose
         )

      # Train and save the model
      time_steps = args.timesteps/args.qnt_saves
      print(time_steps)
      for i in range(args.qnt_saves):
         model.learn(total_timesteps=time_steps)
         model_save_path = f"{args.model_save_path}/a2c_sf6a_{i*time_steps:.0f}"
         model.save(model_save_path)

      # Save the final model
      model_save_path = f"{args.model_save_path}/a2c_sf6a_final"
      model.save(model_save_path)

      vec_env.close()

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Train an A2C model on a custom environment")

      # Optional arguments with defaults
      parser.add_argument("-r", "--render", action='store_true', help="Choose to render the environment")
      parser.add_argument("--n_envs", type=int, default=1, help="Number of environments")
      parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: info)")
      parser.add_argument("-t", "--timesteps", type=int, default=10000, help="Total timesteps of training")
      parser.add_argument("--qnt_saves", type=int, default=5, help="Quantity of models will save")
      parser.add_argument("--model_save_path", type=str, default="trained_models", help="Path to save the trained model")
      
      args = parser.parse_args()
      main(args)