import matplotlib.pyplot as plt
import numpy as np

test_dir = ".teste/"

best_fitness_list = []
med_fitness_list = []

for gen in range(300):
    gen_dir = "_gen" + str(gen) + "/"

    best_fitness = 0
    med_fitness = 0

    for rep in range(1, 11):
        rep_dir = 'rep' + str(rep) + "/"

        with open(test_dir + rep_dir + gen_dir + "_bestfitness.txt") as file:
            value = float(file.read()) / 10000
            best_fitness += value

        with open(test_dir + rep_dir + gen_dir + "_medfitness.txt") as file:
            value = float(file.read()) / 10000
            med_fitness += value
    
    best_fitness /= 10
    med_fitness /= 10

    best_fitness_list.append(best_fitness)
    med_fitness_list.append(best_fitness)

# plot
fig, ax = plt.subplots()

x = np.arange(300)
ax.plot(x, best_fitness_list)
ax.grid()

plt.savefig('graphs/best_fitness')

fig, ax = plt.subplots()

x = np.arange(300)
ax.plot(x, med_fitness_list)
ax.grid()

plt.savefig('graphs/med_fitness')