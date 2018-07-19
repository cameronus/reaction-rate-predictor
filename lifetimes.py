import matplotlib.pyplot as plt
import numpy as np

def normalize(arr):
    arr = np.log10(arr)
    arr -= arr.min()
    arr /= arr.max()
    return arr

def get_lifetimes():
    lifetimes_file = 'lifetimes.out'
    lifetimes = {}
    lifetimes_normalized = {}
    total_molecules = 0
    percentages = []

    with open(lifetimes_file, 'r') as file:
        lines = [x for x in file.read().split('\n') if x != '']
        for line in lines:
            parts = line.split(': ')
            if parts[0] == 'Total Number of Molecules':
                total_molecules += int(parts[1])
                continue
            if parts[0] not in lifetimes:
                lifetimes[parts[0]] = 0
            lifetimes[parts[0]] += int(parts[1])

    # print(lifetimes)
    print('Total Number of Molecules:', total_molecules)

    print('Percentages:')
    for molecule in sorted(lifetimes.keys(), key=lambda x: -lifetimes[x]):
        percentage = lifetimes[molecule] / total_molecules * 100
        print('%s: %f' % (molecule, percentage))
        percentages.append(percentage)
    # percentages = normalize(percentages)
    # print('Normalized:')
    # for index, percentage in enumerate(percentages):
    #     molecule = list(lifetimes.keys())[index]
    #     print('%s: %f' % (molecule, percentage))
    #     lifetimes_normalized[molecule] = percentage

    # plt.hist(percentages, 30, facecolor='green', alpha=0.75)
    # plt.show(block=True)
    return lifetimes_normalized

get_lifetimes()
