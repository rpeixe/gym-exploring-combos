import argparse
import os
import optuna
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from env import make_custom_vec_env
from callbacks import TensorboardCallbackTrain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(trial, args):
    # Create a vectorized environment
    vec_env = make_custom_vec_env(n_envs=args.n_envs, render=args.render, action_memory=args.no_action_memory)
    eval_vec_env = make_custom_vec_env(n_envs=1, render=args.render, action_memory=args.no_action_memory)

    policy_kwargs = {
        "optimizer_class": RMSpropTFLike,
        "optimizer_kwargs": {"eps": 1e-5}
    }

    # Hyperparameters to optimize or preselected, don't save when optimizing
    if args.n_optmization_trials > 0:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1, log=True)
        vf_coef = trial.suggest_float('vf_coef', 1e-8, 1, log=True)
        
        path_models = None
        path_results = None
        log_dir = None
    else:
        learning_rate = 0.0007 # 0.000219 # 0.0007
        n_steps = 200
        gamma = 0.99
        ent_coef = 0.001 # 1.262e-7 # 0.01
        vf_coef = 0.5 # 0.000265 # 0.25

        path_models=os.path.join(SCRIPT_DIR, "trained_models")
        path_results=os.path.join(SCRIPT_DIR, "trained_models/results")

        # Dir to TensorBoard visualisation
        for i in range(10):
            log_dir = os.path.join(SCRIPT_DIR, f"logs/train/train{i+1}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                break

    model = A2C(
        'MultiInputPolicy',
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=args.verbose,
        tensorboard_log=log_dir
    )

    # Callbacks for TensorBoard visualisation
    time_steps = args.timesteps/args.saves
    callbackList = []

    if args.n_optmization_trials <= 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(int(time_steps), 1),
            save_path=path_models,
            name_prefix="a2c_sfa3",
            verbose=2
            )
        callbackList.append(checkpoint_callback)
        
    eval_callback = EvalCallback(
        eval_vec_env,
        n_eval_episodes=1,
        eval_freq=10000,
        log_path=path_results,
        best_model_save_path=path_models,
        deterministic=True,
        verbose=args.verbose
        )
    callbackList.append(eval_callback)
    
    tensorboard_callback = TensorboardCallbackTrain()
    callbackList.append(tensorboard_callback)

    callback = CallbackList(callbackList)

    # Train, save the model and make Tensorboard visualisation
    model.learn(total_timesteps=args.timesteps, callback=callback)

    eval_vec_env.close()
    vec_env.close()

    return eval_callback.last_mean_reward 

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
    parser.add_argument("-o", "--n_optmization_trials", type=int, default=0, help="Trials of hyperparameter optimization (0: preselect parms, 0>: optimize params)")

    args = parser.parse_args()

    if args.n_optmization_trials > 0:
        study = optuna.create_study(direction='maximize')   
        try:
            study.optimize(lambda trial: main(trial, args),n_trials=args.n_optmization_trials)
        except KeyboardInterrupt:
            pass
        print("Best hyperparameters: ", study.best_params)
        print("Best reward: ", study.best_value)
    else:
       main(None, args)
