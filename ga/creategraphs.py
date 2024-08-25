import matplotlib.pyplot as plt
import numpy as np

test_dir = ".teste2/"
num_gens = 1000
num_reps = 1

best_fitness_list = []
med_fitness_list = []
best_combo_list = []
med_combo_list = []

for gen in range(num_gens):
    gen_dir = "_gen" + str(gen) + "/"

    best_fitness = 0
    med_fitness = 0
    best_combo = 0
    med_combo = 0

    for rep in range(1, num_reps+1):
        rep_dir = 'rep' + str(rep) + "/"

        with open(test_dir + rep_dir + gen_dir + "_bestfitness.txt") as file:
            value = float(file.read()) / 10000
            best_fitness += value

        with open(test_dir + rep_dir + gen_dir + "_medfitness.txt") as file:
            value = float(file.read()) / 10000
            med_fitness += value

        with open(test_dir + rep_dir + gen_dir + "_bestcombo.txt") as file:
            value = float(file.read())
            best_combo += value

        with open(test_dir + rep_dir + gen_dir + "_medbestcombo.txt") as file:
            value = float(file.read())
            med_combo += value
    
    best_fitness /= num_reps
    med_fitness /= num_reps
    best_combo /= num_reps
    med_combo /= num_reps

    best_fitness_list.append(best_fitness)
    med_fitness_list.append(med_fitness)
    best_combo_list.append(best_combo)
    med_combo_list.append(med_combo)

# plot
fig, ax = plt.subplots()

x = np.arange(num_gens)
ax.plot(x, best_fitness_list)
ax.grid()

plt.savefig('graphs2/best_fitness')

fig, ax = plt.subplots()

x = np.arange(num_gens)
ax.plot(x, med_fitness_list)
ax.grid()

plt.savefig('graphs2/med_fitness')

# plot
fig, ax = plt.subplots()

x = np.arange(num_gens)
ax.plot(x, best_combo_list)
ax.grid()

plt.savefig('graphs2/best_combo')

fig, ax = plt.subplots()

x = np.arange(num_gens)
ax.plot(x, med_combo_list)
ax.grid()

plt.savefig('graphs2/med_combo')