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
    print(len(reactions))
    print(len(rates))
    
