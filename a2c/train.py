import argparse
import os
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from env import make_env
from callbacks import TensorboardCallbackTrain

def main(args):
      # Create a vectorized environment
      vec_env = make_vec_env(
         make_env,
         n_envs=args.n_envs,
         env_kwargs={"render": args.render},
         vec_env_cls=SubprocVecEnv)
      vec_env = VecFrameStack(vec_env, 4)
      
      # Dir to TensorBoard visualisation
      for i in range(10):
        log_dir = os.path.join("logs/train", f"train{i+1}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
        
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
         verbose=args.verbose,
         tensorboard_log=log_dir
         )

      time_steps = args.timesteps/args.saves
      
      # Callbacks for TensorBoard visualisation
      checkpoint_callback = CheckpointCallback(
         save_freq=max(int(time_steps), 1),
         save_path="./trained_models",
         name_prefix="a2c_stf6",
         verbose=2
         )
      
      eval_callback = EvalCallback(
         vec_env,
         eval_freq=500,
         log_path="./trained_models/results",
         best_model_save_path="./trained_models",
         verbose=1
         )
      
      tensorboard_callback = TensorboardCallbackTrain()
      callback = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])
      
      # Train, save the model and make Tensorboard visualisation
      model.learn(total_timesteps=args.timesteps, callback=callback)

      vec_env.close()

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Train an A2C model on a custom environment")

      # Optional arguments with defaults
      parser.add_argument("-r", "--render", action='store_true', help="Choose to render the environment")
      parser.add_argument("--n_envs", type=int, default=1, help="Number of environments")
      parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: info)")
      parser.add_argument("-t", "--timesteps", type=int, default=10000, help="Total timesteps of training")
      parser.add_argument("--saves", type=int, default=5, help="Quantity of models will save")
      
      args = parser.parse_args()
      main(args)