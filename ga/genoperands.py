import os
import random
import calcfitness as cf
import sys
import shutil

moves  = [
        '1lp', '1mp', '1hp', '1lk', '1mk', '1hk', '1-',
        '2lp', '2mp', '2hp', '2lk', '2mk', '2hk', '2-',
        '3lp', '3mp', '3hp', '3lk', '3mk', '3hk', '3-',
        '4lp', '4mp', '4hp', '4lk', '4mk', '4hk', '4-',
        '5lp', '5mp', '5hp', '5lk', '5mk', '5hk', '5-',
        '6lp', '6mp', '6hp', '6lk', '6mk', '6hk', '6-',
        '7lp', '7mp', '7hp', '7lk', '7mk', '7hk', '7-',
        '8lp', '8mp', '8hp', '8lk', '8mk', '8hk', '8-',
        '9lp', '9mp', '9hp', '9lk', '9mk', '9hk', '9-',
        ]

def reproduce(father1, father2, path, i, k, indSize,
              mutationProb, crossProb, maxMutations):
    currentGen = path+str(i)
    nextGen = path+str(i+1)

    fat1 = open(os.path.join(currentGen,father1)).read().split()
    fat2 = open(os.path.join(currentGen,father2)).read().split()
    son1 = fat1
    son2 = fat2

    sonout1 = open(os.path.join(nextGen,str(k)+'.ind'),'w')
    sonout2 = open(os.path.join(nextGen,str(k+1)+'.ind'),'w')

    rollCross = random.randint(0,100)
    rollMut = random.randint(0,100)
    numMut = random.randint(1,maxMutations)

    if(rollCross <= crossProb):
        cp = random.randint(1,indSize-1) #crossover point
        son1 = fat1[0:cp] + fat2[cp:]
        son2 = fat2[0:cp] + fat1[cp:]
    if(rollMut <= mutationProb):
        for i in range(0,numMut):
            ind1 = random.randint(3,indSize-1)
            ind2 = random.randint(3,indSize-1)
            son1[ind1] = random.choice(moves)
            son2[ind2] = random.choice(moves)


    for i in range(0, indSize):
        sonout1.write(son1[i]+' ')
        sonout2.write(son2[i]+' ')

    sonout1.close()
    sonout2.close()


def bubblesort(vlist,smartGen):
    length = len(vlist) - 1
    s = False

    while not s:
        s = True
        for i in range(length):
            a = int((open(os.path.join(smartGen,str(vlist[i])+'.fit'),'r')).read())
            b = int((open(os.path.join(smartGen,str(vlist[i+1])+'.fit'),'r')).read())
            if a < b:
                s = False
                vlist[i], vlist[i+1] = vlist[i+1], vlist[i]
    return vlist

def smartmutation(father, path, gamepath, gen, segsize, h):
    currentGen = path+str(gen)
    nextGen = path+str(gen+1)
    smartGen = os.path.join(currentGen,"search")
    if(os.path.exists(smartGen)):
        shutil.rmtree(smartGen)    
    os.makedirs(smartGen)

    fat = open(os.path.join(currentGen,father+'.ind')).read().split()
    index=0
    start = random.randint(0,len(fat)-segsize)
    for i in range(start,start+segsize):
        for m in moves:
            if (fat[i] != m):
                son = list(fat)
                son[i] = m
                sonout = open(os.path.join(smartGen,str(index)+'.ind'),'w')
                index = index+1
                for k in range(0, len(son)):
                    sonout.write(son[k]+' ')
                sonout.close()
    sonout = open(os.path.join(smartGen,str(index)+'.ind'),'w')
    for k in range(0, len(fat)):
        sonout.write(fat[k]+' ')
    sonout.close()

    cf.calcFit(smartGen,gamepath)
    vetind = range(0,index)
    bubblesort(vetind,smartGen)

    shutil.copyfile(smartGen + "/" + str(vetind[0])+".ind", nextGen + "/" + str(h)+".ind")
    shutil.copyfile(smartGen + "/" + str(vetind[0])+".fit", nextGen + "/" + str(h)+".fit")
    shutil.copyfile(smartGen + "/" + str(vetind[1])+".ind", nextGen + "/" + str(h+1)+".ind")
    shutil.copyfile(smartGen + "/" + str(vetind[1])+".fit", nextGen + "/" + str(h+1)+".fit")
    

if __name__=='__main__':
    print(smartmutation(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),0))
