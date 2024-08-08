import random
#import gentree as gt
import os


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

def regnextGen(basepath, gennum, startindex,regval, indsize):
    path = basepath + str(gennum)
    for index in range(startindex,regval+startindex):
        ofile = open(path + "/" + str(index)+".ind",'w')
        for i in range (0,indsize):
            if(i<3):
                ofile.write('5- ')
            else:
                ofile.write(random.choice(moves))
                ofile.write(' ')
        ofile.close()
        index = index + 1

		
def genPopulation(path,popsize,indsize):
    if(not os.path.exists(path)):
        os.makedirs(path)

    for index in range(0,popsize):
        ofile = open(path + "/" + str(index)+".ind",'w')
        for i in range (0,indsize):
            if(i<3):
                ofile.write('5- ')
            else:
                ofile.write(random.choice(moves))
                ofile.write(' ')
        ofile.close()
        index = index + 1
    

def premadegenPopulation(path, popsize, indsize):
    print("\rPopulation generated from premade combos")
    if(not os.path.exists(path)):
        os.makedirs(path)

    for index in range(0,popsize):
        s = []
        while len(s)<indsize:
            r = random.randint(0,45)
            s.extend(open('premade//'+str(r)+'.txt','r').read().split())
        ofile = open(path + "/" + str(index)+".ind",'w')
        for i in range(0,len(s)):
            if(i<indsize):
                ofile.write(s[i]+' ')
            else:
                break
        ofile.close()
        index = index + 1

if __name__ == "__main__":
    genPopulation("test",3,30)
