import matplotlib.pyplot as plt
import numpy as np
import re

def normalize(arr):
    arr = np.log10(arr)
    arr -= arr.min()
    arr /= arr.max()
    return arr

def get_lifetimes():
    lifetimes_file = 'lifetimes.out'

    x = []
    y = []

    with open(lifetimes_file, 'r') as file:
        text = file.read()
        lines = [x for x in re.split('Frame \d+\n', text) if x != '']
        total = lines[0]
        print(total)
        mols = [x[:-1] for x in lines[1:]]
        # print(mols)
        for index, mol in enumerate(mols):
            parts = mol.split(': ')
            if parts[0] == 'C2 H6 1(C-C) 6(H-C)': # C1 H4 4(H-C), C2 H6 1(C-C) 6(H-C), H2 1(H-H)
                x.append(index)
                y.append(int(parts[1]))
        plt.plot(x, y)
        plt.show(block=True)

    # # print(lifetimes)
    # print('Total Number of Molecules:', total_molecules)
    #
    # print('Percentages:')
    # for molecule in sorted(lifetimes.keys(), key=lambda x: -lifetimes[x]):
    #     percentage = lifetimes[molecule] / total_molecules * 100
    #     print('%s: %f' % (molecule, percentage))
    #     percentages.append(percentage)
    # percentages = normalize(percentages)
    # print('Normalized:')
    # for index, percentage in enumerate(percentages):
    #     molecule = list(lifetimes.keys())[index]
    #     print('%s: %f' % (molecule, percentage))
    #     lifetimes_normalized[molecule] = percentage

    # plt.hist(percentages, 30, facecolor='green', alpha=0.75)
    # plt.show(block=True)
    # return lifetimes_normalized

get_lifetimes()
