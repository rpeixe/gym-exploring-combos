from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from env import make_env

def main():
        env = make_env(False)

        # Create a vectorized environment without rendering
        vec_env = make_vec_env(lambda: env, n_envs=1)

        # Create the A2C model
        model = A2C('CnnPolicy', vec_env, learning_rate=0.0007, n_steps=5, gamma=0.99, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, verbose=1)

        # Train the model with callbacks
        model.learn(total_timesteps=10000)

        # Save the model
        model.save("a2c_sf6a_final")
        vec_env.close()

if __name__ == "__main__":
     main()
