#import evaluate as ev
import os
import subprocess
import shutil
import time
import re
import math

from env import make_custom_env
from utils import decode_individual

def calcFit(path):
    env = make_custom_env(render=False)

    vet = os.listdir(path)
    #print os.path.splitext(vet[0])
    index = 0
    for i in range(0, len(vet)):
        if(os.path.splitext(vet[i])[1] != ".ind"):
            continue
        if(os.path.exists(path + "/" + os.path.splitext(vet[i])[0] + ".fit")):
            continue
        inp = path + "/" + vet[i]
        out = open(path + "/" + os.path.splitext(vet[i])[0] + ".fit", 'w')
        out_best = open(path + "/" + os.path.splitext(vet[i])[0] + ".best", 'w')


        finp = open(inp)
        ind = finp.read()
        moves = decode_individual(ind)

        _obs = env.reset()
        _done = False
        combos = []
        new_combo = []
        combo_time = 0
        idle_time = 0
        current_combo_hits = 0
        current_combo_index = 0
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
        if len(new_combo) < 1:
            new_combo.append(idle_time)
        if len(new_combo) < 2:
            new_combo.append(combo_time)
        new_combo.append(idle_time)
        combos.append(new_combo)

        bigger = combos.index(max(combos, key = lambda combo: combo[1]))
        best = combos[bigger][1]

        ffit = best

        for i in range(len(combos)):
            if(i<bigger):
                ffit+= combos[i][1] / math.pow(1+combos[i][2],2) 
            elif(i>bigger):
                ffit+= combos[i][1] / math.pow(1+combos[i][0],2)

        out.write(str(int(ffit*10000)))
        out.close()
        out_best.write(str(int(best)))
        out_best.close()
        #shutil.copyfile(fgpath, os.path.join(path, str(index) + '.out'))

        #os.remove(fgpath)
        index = index + 1


        #print("Simulation Ended")

        #out.write(str(err))
        #out.close()
    env.close()

if __name__ == "__main__":
    calcFit("test","Game")
