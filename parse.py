import numpy as np
from itertools import groupby
from collections import Counter
import glob
import math

data_dir = 'methane-data'

for reaction_file, rate_file in zip(sorted(glob.iglob(data_dir + '/reactdict_NPT*.txt')), sorted(glob.iglob(data_dir + '/ktrue*.txt'))):
    print(reaction_file, rate_file)
    with open(reaction_file, 'r') as file: reactions = file.read().split('\n')[:-1]
    with open(rate_file, 'r') as file: rates = np.fromstring(file.read(), dtype=np.float64, sep='\n')
    # print(len(reactions))
    # print(len(rates))
    for reaction in reactions:
        reaction = reaction.split(':')
        number = reaction[0]
        # print(number)
        normalized = ' '.join(reaction[1].split())
        sides = [[list(filter(None, r.split(' '))) for r in side.split('+')] for side in normalized.split('=>')]
        # print(sides)
        bond_sides = []
        for side in sides:
            added = Counter()
            for r in side:
                idx = math.ceil(len(r) / 2)
                chems = r[:idx]
                bonds = r[idx:]
                bond_counts = {}
                for bond in bonds:
                    parts = bond.split('(')
                    bond_counts[parts[1][:-1]] = int(parts[0])
                # print(chems)
                added.update(bond_counts)
                # print(bond_counts)
            bond_sides.append(added)
        print(number)
        print(bond_sides[0])
        print(bond_sides[1])

# [[[[chem], {bonds}], [[chem], {bonds}]], [[[chem], {bonds}], [[chem], {bonds}]]]
