from env import make_custom_env
from utils import decode_individual
import time


finp = open(".teste/rep1/_gen300/84.ind")
ind = finp.read()
moves = decode_individual(ind)

env = make_custom_env(render=True)
obs = env.reset()
done = False
output = ''
total_reward = 0
for move in moves:
    obs, reward, terminated, truncated, info = env.step(move)
    total_reward += reward
    time.sleep(0.01)

print(total_reward)