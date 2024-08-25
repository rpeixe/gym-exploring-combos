from env import make_custom_env
import time
import numpy as np

with open("best_combo.txt") as file:
    inp = file.read()

moves = eval(inp, {"array": np.array, "float32": np.float32})
env = make_custom_env(render=True)
obs = env.reset()
done = False
output = ''
total_reward = 0
time.sleep(5)
for move in moves:
    obs, reward, terminated, truncated, info = env.step(move)
    total_reward += reward
    time.sleep(0.0166)

print(total_reward)