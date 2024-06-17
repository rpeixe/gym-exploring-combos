import argparse
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from env import make_custom_vec_env
from callbacks import TensorboardCallbackTrain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args):
    # Create a vectorized environment
    vec_env = make_custom_vec_env(n_envs=args.n_envs, render=args.render, action_memory=args.no_action_memory)
    eval_vec_env = make_custom_vec_env(n_envs=args.n_envs, render=args.render, action_memory=args.no_action_memory)
    
    # Dir to TensorBoard visualisation
    for i in range(10):
        log_dir = os.path.join(SCRIPT_DIR, f"logs/train/train{i+1}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
        
    policy_kwargs = {
        "optimizer_class": RMSpropTFLike,
        "optimizer_kwargs": {"eps": 1e-5}
    }

    model = A2C(
        'MultiInputPolicy',
        vec_env,
        ent_coef=0.01,
        vf_coef=0.25,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=args.verbose,
        tensorboard_log=log_dir
        )

    time_steps = args.timesteps/args.saves
    
    # Callbacks for TensorBoard visualisation
    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(time_steps), 1),
        save_path=os.path.join(SCRIPT_DIR, "trained_models"),
        name_prefix="a2c_sfa3",
        verbose=2
        )
    
    eval_callback = EvalCallback(
        eval_vec_env,
        n_eval_episodes=1,
        eval_freq=500,
        log_path=os.path.join(SCRIPT_DIR, "trained_models/results"),
        best_model_save_path=os.path.join(SCRIPT_DIR, "trained_models"),
        deterministic=True,
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
    parser.add_argument("-no_am", "--no_action_memory", action='store_false', help="Choose to not feed the last actions as input")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: info)")
    parser.add_argument("-t", "--timesteps", type=int, default=10000, help="Total timesteps of training")
    parser.add_argument("--saves", type=int, default=5, help="Quantity of models will save")
    parser.add_argument("--seed", type=int, default=0, help="Define the random seed the model will use")
    
    args = parser.parse_args()
    main(args)
