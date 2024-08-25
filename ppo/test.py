import argparse
import os
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from env import make_custom_vec_env
from callbacks import TensorboardCallbackTest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args):
    vec_env = make_custom_vec_env(render=True, action_memory=args.no_action_memory)

    # Loading trained model
    model = PPO.load(args.load_path, env=vec_env)

    # Callback for TensorBoard visualisation
    for i in range(10):
        log_dir = os.path.join(SCRIPT_DIR, f"logs/test/test{i+1}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
    callback = TensorboardCallbackTest(log_dir)

    print("Starting test")

    # Test the trained model
    best_reward = 0
    total_rewards = []
    best_action = []
    for episode in range(args.episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0
        episode_action = []

        while not done:
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = vec_env.step(action)

            episode_action.append(action[0])
            episode_reward += reward[0]

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_action = episode_action

        print(f"Episode {episode + 1}")
        print(f"\treward: {episode_reward}")
        total_rewards.append(episode_reward)

        # Log episode reward to TensorBoard
        callback.log_episode_reward(episode_reward, episode)

    # Print results from test
    print(f"Reward results from {args.episodes} episodes:")
    avg_reward = sum(total_rewards) / len(total_rewards)
    std_reward = (sum((x - avg_reward) ** 2 for x in total_rewards) / len(total_rewards)) ** 0.5
    print(f"avg_reward: {avg_reward:.2f}")
    print(f"std_reward: {std_reward:.2f}")
    print(f"\nBest reward: {best_reward}")
    #print(f"Best action: {best_action}\n")
    with open('best_combo.txt', 'w') as file:
        file.write(str(best_action))
    
    # Log final metrics to TensorBoard
    callback.log_final_metrics(avg_reward, std_reward, best_reward)

    vec_env.close()
    callback.writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test from A2C model")

    parser.add_argument("-no_am", "--no_action_memory", action='store_false', help="Choose to not feed the last actions as input")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Total episodes of test")
    parser.add_argument("-d", "--deterministic", type=bool, default=False, help="Set predict deterministic of test")
    parser.add_argument("--load_path", type=str, default=os.path.join(SCRIPT_DIR, "trained_models/best_model.zip"), help="Path to load the trained model")
    
    args = parser.parse_args()
    main(args)
