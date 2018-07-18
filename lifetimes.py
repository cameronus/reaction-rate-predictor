import matplotlib.pyplot as plt
import numpy as np

lifetimes_file = 'lifetimes.out'
lifetimes = {}
total_molecules = 0
percentages = []

def normalize(arr):
    arr = np.log10(arr)
    arr -= arr.min()
    arr /= arr.max()
    return arr

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

print(lifetimes)
print('Total Number of Molecules:', total_molecules)

print('Percentages:')
for molecule in lifetimes:
    percentage = lifetimes[molecule] / total_molecules * 100
    print('%s: %f' % (molecule, percentage))
    percentages.append(percentage)
percentages = normalize(percentages)
print('Normalized:')
for index, percentage in enumerate(percentages):
    print('%s: %f' % (list(lifetimes.keys())[index], percentage))

plt.hist(percentages, 30, normed=1, facecolor='green', alpha=0.75)
plt.show(block=True)
