import numpy as np
from itertools import groupby
from collections import Counter
import glob
import math

data_dir = 'methane-data'

reactions = []
numerators = []
denominators = []

rxns = {}

for reaction_file, numer_file, denom_file in zip(sorted(glob.iglob(data_dir + '/reactdict_NPT*.txt')), sorted(glob.iglob(data_dir + '/numer*.txt')), sorted(glob.iglob(data_dir + '/denom*.txt'))):
    print(reaction_file, numer_file, denom_file)
    with open(reaction_file, 'r') as file: reactions += [reaction.split(': ')[1] for reaction in file.read().split('\n')[:-1]]
    with open(numer_file, 'r') as file: numerators = np.concatenate([numerators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    with open(denom_file, 'r') as file: denominators = np.concatenate([denominators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    print(len(reactions), len(numerators), len(denominators))

print('----------------------------------------')

counts = Counter(reactions)
for rxn in reactions:
    rxns[rxn] = [0, 0]
for index, rxn in enumerate(reactions):
    if (counts[rxn] == 6) and numerators[index] != 0.0 and denominators[index] != 0.0:
        rxns[rxn][0] += numerators[index]
        rxns[rxn][1] += denominators[index]

for rxn, rate in rxns.items():
    if rate[0] > 0 and rate[1] > 0:
        # print(sum(rate[0]), sum(rate[1]))
        rate = (rate[0] / rate[1]) / 0.012
        print(rxn)
        print(rate)
