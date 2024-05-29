import argparse
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from env import make_env

def main(args):
        vec_env = make_vec_env(
            make_env,
            n_envs=1,
            env_kwargs={"render": True},
            vec_env_cls=SubprocVecEnv)
        vec_env = VecFrameStack(vec_env, 4)
        vec_env.reset()

        # Loading model treined
        model = A2C.load(args.model_load_path, env=vec_env)

        print("Starting test")

        # Test the model treined
        total_rewards = []
        best_reward = 0
        best_action = []
        for episode in range(args.num_episodes):
            obs = vec_env.reset()
            done = False
            episode_reward = 0
            episode_action = []

            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = vec_env.step(action)

                episode_action.append(action[0])
                episode_reward += reward[0]

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_action = episode_action

            print(f"Episode {episode + 1}")
            print(f"\treward: {episode_reward}")
            #print(f"\taction: {episode_action}")
            total_rewards.append(episode_reward)

        # Print results from test
        print(f"Reward reults from {args.num_episodes} episodes:")
        avg_reward = sum(total_rewards) / len(total_rewards)
        std_reward = (sum((x - avg_reward) ** 2 for x in total_rewards) / len(total_rewards)) ** 0.5
        print(f"avg_reward: {avg_reward:.2f}")
        print(f"std_reward: {std_reward:.2f}")
        print(f"\nBest reward: {best_reward}")
        print(f"Best action: {best_action}\n")

        vec_env.close()

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run test from A2C model")

        parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to test")
        parser.add_argument("--model_load_path", type=str, default="trained_models/a2c_sf6a_final", help="Path to load the trained model")
        
        args = parser.parse_args()
        main(args)
