from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from env import make_env

def main():
    vec_env = make_vec_env(make_env)

    model = A2C.load("a2c_sf6a_final", env=vec_env)

    print("Starting test")
    done = False
    obs = vec_env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
    vec_env.close()

if __name__ == "__main__":
     main()
