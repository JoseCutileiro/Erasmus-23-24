# Needed this for a game balance feature

import random
import matplotlib.pyplot as plt

def throw(num_dices):
    ret = []
    for i in range(num_dices):
        ret += [random.randint(0,5),]
    return ret

def repeat(N,num_dices):
    ret = []
    for i in range(N):
        r = throw(num_dices)
        print(r)
        s = sum(r)
        ret += [s,]
    return ret

def plot_results(results,num_dices):
    plt.hist(results, bins=range(1, 6 * num_dices), align='left', edgecolor='black', linewidth=1.2)
    plt.xlabel('Dice Roll')
    plt.ylabel('Frequency')
    plt.title('Dice Roll Results')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


N = 10000
num_dices = 10
results = repeat(N,num_dices)
plot_results(results,num_dices)