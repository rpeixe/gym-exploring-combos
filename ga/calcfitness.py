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


        finp = open(inp)
        ind = finp.read()
        moves = decode_individual(ind)

        obs = env.reset()
        done = False
        output = ''
        for move in moves:
            obs, reward, terminated, truncated, info = env.step(move)
            if reward:
                output += '1'
            else:
                output += '0'


        fg = output # frames in combo e.g. 000111111111111111111100000111110000000
        #print "Opened fg_output"



        h2 = [m.group(0) for m in re.finditer(r"(\d)\1*", fg)]

        bigger=0
        for i in range(0,len(h2)):
            if(len(h2[i]) > len(h2[bigger]) and h2[i][0]=='1'):
               bigger = i

        ffit = 0.0
        for i in range(0,len(h2)):
            if(i==bigger and h2[i][0]=='1'):
                ffit+=len(h2[i])
            elif(i<bigger and h2[i][0]=='1'):
                ffit+= len(h2[i]) / math.pow(1+len(h2[i+1]),2) 
            elif(i>bigger and h2[i][0]=='1'):
                ffit+= len(h2[i]) / math.pow(1+len(h2[i-1]),2)

        out.write(str(int(ffit*10000)))
        out.close()
        #shutil.copyfile(fgpath, os.path.join(path, str(index) + '.out'))

        #os.remove(fgpath)
        index = index + 1


        #print("Simulation Ended")

        #out.write(str(err))
        #out.close()

if __name__ == "__main__":
    calcFit("test","Game")
