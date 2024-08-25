from env import make_custom_env
from utils import decode_individual
import time


finp = open(".teste2/rep1/_gen227/110.ind")
ind = finp.read()
moves = decode_individual(ind)

env = make_custom_env(render=True)
_obs = env.reset()
_done = False
combos = []
new_combo = []
combo_time = 0
idle_time = 0
current_combo_hits = 0
current_combo_index = 0
time.sleep(3)
for move in moves:
    _obs, _reward, _terminated, _truncated, info = env.step(move)
    if info['combo'] < current_combo_hits:
        # Acabou o combo
        current_combo_index += 1
        new_combo.append(combo_time)
        combo_time = 0
        current_combo_hits = 0
    elif current_combo_hits > 0:
        # No meio de um combo
        combo_time += 1
        current_combo_hits = info['combo']
    elif info['combo'] > 0:
        # Comecou um combo
        if current_combo_index > 0:
            # Nao e o primeiro combo
            new_combo.append(idle_time)
            combos.append(new_combo)
            new_combo = []
        new_combo.append(idle_time)
        idle_time = 0
        combo_time += 1
        current_combo_hits = info['combo']
    else:
        # Nao esta em combo
        idle_time += 1
    time.sleep(0.01)
if len(new_combo) < 1:
    new_combo.append(idle_time)
if len(new_combo) < 2:
    new_combo.append(combo_time)
new_combo.append(idle_time)
combos.append(new_combo)


print(max(combos))
print(combos)